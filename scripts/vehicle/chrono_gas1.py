"""G-AS1 (VEHICLE_PROPOSAL.md section 11.15): active chassis on a rough road.

Task: the lane changes of ``chrono_rollover_control.py`` (HMMWV, 800 kg roof load, 0.6 m corridor), now on a random
rough road (``supermimic.vehicles.road``, 4 cm RMS plus potholes; the wheels briefly leave the ground) with four
active suspension force actuators (``chrono_worker`` option ``active_susp``). Six actions: steering, acceleration and
the four normalised corner forces. Every controller knows the road (preview). Reward: the rollover task's reward
minus a comfort term on the body's vertical acceleration and a small actuator-energy cost.

Controllers:
* ``mpc_passive``: the roll-model MPC of ``chrono_roll_mpc.py`` (identified on the flat-road data), suspension passive;
* ``hierarchical``: the same MPC for the path plus corner skyhook with road-preview feedforward for the suspension
  (``chrono_gas1_env_check.py``), the industrial layered structure;
* ``ssf_passive`` / ``ssf_preview``: the SSF speed limiter with passive or preview-skyhook suspension;
* ``supermimic``: one policy for all six actions, trained through a learned world model (``chrono_lift_check.LiftWM``)
  whose inputs are the six actions and the change in road height under each wheel over the step, computed the same
  way (``Road.height`` at the wheel positions) in Chrono data and in imagination.

    SUPERMIMIC_CHRONO_PYTHON=chrono_env/bin/python PYTHONPATH=src python scripts/vehicle/chrono_gas1.py \\
        --lift-data logs/chrono_lift_data.npz --road-dir logs/roads_gas1 --out logs/chrono_gas1.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chrono_rollover_control as crc
from chrono_gas1_env_check import SIDE, preview_ff, skyhook
from chrono_lift_check import CFG as LIFT_CFG
from chrono_lift_check import LiftWM, batch, episode_plan, predict, windows
from chrono_roll_mpc import RollMPC, identify

from supermimic.vehicles.chrono_env import ChronoPool
from supermimic.vehicles.road import road_files

EXTRA = ["az", *[f"dz{i}" for i in range(4)], *[f"dzd{i}" for i in range(4)], *[f"fa{i}" for i in range(4)]]
STATE_G = crc.STATE + EXTRA
IXG = {k: i for i, k in enumerate(STATE_G)}
CFG = {**LIFT_CFG, "active_susp": True}
PREVIEW_D = [0.3, 0.6, 1.2, 2.4]  # [m] road preview distances ahead of each wheel (policy features)
W_COMFORT, W_ENERGY = 0.2, 0.01
K_FF = 30.0  # preview feedforward gain of the hierarchical controller (chrono_gas1_env_check: best of 15 / 30)
DT, DS = crc.DT, crc.DS


# ---------------------------------------------------------------------- road and wheels


class Roads:
    """The roads of a run: numpy ``Road`` objects for Chrono-side controllers and one tensor for imagination."""

    def __init__(self, seeds, out_dir, geo):
        self.items = road_files(seeds, out_dir)
        self.road0 = self.items[0][0]
        self.Z = torch.as_tensor(np.stack([r.z for r, _ in self.items]), dtype=torch.float32)
        self.geo = geo
        self.dx = torch.tensor([geo["a"], geo["a"], -geo["b"], -geo["b"]], dtype=torch.float32)
        self.dy = torch.as_tensor(SIDE * geo["track"] / 2, dtype=torch.float32)

    def cfg(self, k):
        road, path = self.items[k]
        return road.chrono_cfg(path)

    def height(self, rid, x, y):
        """Bilinear road height (torch, differentiable in x, y); rid (B,), x, y (B, ...)."""
        r0 = self.road0
        rows, cols = self.Z.shape[1:]
        sx, sy = r0.spacing
        u = ((x - (r0.center[0] - r0.size[0] / 2)) / sx).clamp(0, cols - 1.001)
        v = (((r0.center[1] + r0.size[1] / 2) - y) / sy).clamp(0, rows - 1.001)
        c0, w0 = u.floor().long(), v.floor().long()
        fu, fv = u - c0, v - w0
        R = rid.view(-1, *([1] * (x.dim() - 1))).expand_as(c0)
        Z = self.Z
        return ((1 - fv) * ((1 - fu) * Z[R, w0, c0] + fu * Z[R, w0, c0 + 1]) + fv * ((1 - fu) * Z[R, w0 + 1, c0] + fu * Z[R, w0 + 1, c0 + 1]))

    def wheels(self, pose):
        c, s = torch.cos(pose[:, 2:3]), torch.sin(pose[:, 2:3])
        return pose[:, 0:1] + c * self.dx - s * self.dy, pose[:, 1:2] + s * self.dx + c * self.dy

    def under_wheels(self, rid, pose):
        return self.height(rid, *self.wheels(pose))


# ---------------------------------------------------------------------- task (torch)


def reward_g(x, a, a_prev, e_lat, e_head, v_ref):
    r, fail, ltr = crc.reward(x, a, a_prev, e_lat, e_head, v_ref)
    return r - W_COMFORT * (x[:, IXG["az"]] / 2) ** 2 - W_ENERGY * (a[:, 2:6] ** 2).sum(-1), fail, ltr


def features_g(x, pose, s, pid, v_ref, a_prev, paths, wm, roads, rid):
    ob, e_lat, e_head = crc.features(x, pose, s, pid, v_ref, a_prev[:, :2], paths, wm)
    wx, wy = roads.wheels(pose)
    z_now = roads.height(rid, wx, wy)
    d = torch.tensor(PREVIEW_D)
    c, sn = torch.cos(pose[:, 2])[:, None, None], torch.sin(pose[:, 2])[:, None, None]
    ahead = roads.height(rid, wx[..., None] + c * d, wy[..., None] + sn * d)
    return torch.cat([ob, a_prev[:, 2:6], ((ahead - z_now[..., None]) / 0.05).flatten(1)], -1), e_lat, e_head


class Policy6(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 6))

    def forward(self, obs):
        u = torch.tanh(self.net(obs))
        return torch.cat([u[:, :1], crc.A_MIN + (u[:, 1:2] + 1) * 0.5 * (crc.A_MAX - crc.A_MIN), u[:, 2:]], -1)


def kinematic_pose(pose, x):
    vx, vy, r = x[:, IXG["vx"]], x[:, IXG["vy"]], x[:, IXG["r"]]
    yaw = pose[:, 2] + r * DT
    return torch.stack([pose[:, 0] + (vx * torch.cos(yaw) - vy * torch.sin(yaw)) * DT, pose[:, 1] + (vx * torch.sin(yaw) + vy * torch.cos(yaw)) * DT, yaw], -1)


def imagine_g(wm, policy, starts, paths, H, roads, corridor_w: float = 0.0, fail_pen: float = 0.0, ltr_w: float = 0.0):
    """Training objective (not the evaluation reward): the task reward, optionally minus a corridor barrier
    corridor_w x (excess of |lateral error| over 0.4 m / 0.1 m)^2 (the MPC's corridor term), minus fail_pen x the
    probability of failing at each step (in Chrono a failure ends the episode, which the 1.5 s horizon undercounts),
    and minus ltr_w x (excess of |LTR| over 0.6 / 0.1)^2 (the MPC's load-transfer limit)."""
    Xh, Ah, pose, s, pid, v_ref, rid = starts
    x, a_prev = Xh[-1], Ah[-1][:, :6]
    c = torch.ones(x.shape[0])
    J = torch.zeros(x.shape[0])
    for _ in range(H):
        ob, _, _ = features_g(x, pose, s, pid, v_ref, a_prev, paths, wm, roads, rid)
        a = policy(ob)
        pose1 = kinematic_pose(pose, x)
        u = torch.cat([a, roads.under_wheels(rid, pose1) - roads.under_wheels(rid, pose)], -1)
        x1 = wm.step_hist(Xh, torch.cat([Ah, u[None]]))
        Xh, Ah = torch.cat([Xh[1:], x1[None]]), torch.cat([Ah[1:], u[None]])
        pose = kinematic_pose(pose, x1)
        _, _, pyaw, _ = crc.path_at(paths, pid, s)
        s = s + (x1[:, IXG["vx"]] * torch.cos(pose[:, 2] - pyaw) - x1[:, IXG["vy"]] * torch.sin(pose[:, 2] - pyaw)) * DT
        _, e_lat, e_head = crc.features(x1, pose, s, pid, v_ref, a[:, :2], paths, wm)
        rew, fail, ltr = reward_g(x1, a, a_prev, e_lat, e_head, v_ref)
        rew = rew - corridor_w * (torch.relu(e_lat.abs() - 0.4) / 0.1) ** 2 - ltr_w * (torch.relu(ltr - 0.6) / 0.1) ** 2
        J = J + c * (rew - fail_pen * fail)
        c = c * (1 - fail)
        x, a_prev = x1, a
    return J


# ---------------------------------------------------------------------- Chrono episodes


def road_input(roads, rid, poses):
    """(T+1, 3) poses -> (T, 4) change in road height under each wheel over each step."""
    z = roads.under_wheels(torch.full((len(poses),), rid), torch.as_tensor(np.asarray(poses), dtype=torch.float32)).numpy()
    return np.diff(z, axis=0)


class Hierarchical:
    """Path: the roll-model MPC; suspension: passive, or corner skyhook + road-preview feedforward."""

    def __init__(self, mpc, roads, preview: bool):
        self.mpc, self.roads, self.preview = mpc, roads, preview

    def reset(self, n):
        if self.mpc is not None:
            self.mpc.reset(n)

    def act(self, i, o, path, j, v_ref, a_prev, rid):
        a = self.mpc.act(i, o, path, j, v_ref, a_prev[:2]) if self.mpc is not None else crc.baseline_action("ssf_limiter", o, path, j, v_ref)
        f = np.clip(skyhook(o, 1.0) + preview_ff(o, self.roads.geo, self.roads.items[rid][0], K_FF), -1, 1) if self.preview else np.zeros(4)
        return [*a, *f.tolist()]


def drive_g(pool, cases, roads, controller, policy=None, wm=None, noise=0.0, rng=None, max_steps=700):
    """cases: (path at the origin, v_ref, road index). Like ``chrono_rollover_control.drive`` with six actions."""
    n = len(cases)
    if hasattr(controller, "reset"):
        controller.reset(n)
    obs = pool.reset([{**CFG, "v": v, **roads.cfg(k)} for _, v, k in cases], ids=range(n))
    paths = [crc.placed(p, o["x"], o["y"], o["yaw"]) for (p, _, _), o in zip(cases, obs)]
    tp = torch.as_tensor(np.stack([np.pad(p, ((0, 2400 - len(p)), (0, 0)), mode="edge") for p in paths]), dtype=torch.float32)
    logs = [{"x": [[o[k] for k in STATE_G]], "pose": [[o["x"], o["y"], o["yaw"]]], "a": [], "s": [], "r": [], "ltr": [], "lat": [], "az": [],
             "tilt": [o["tilt"]], "end": "completed", "v_ref": v, "rid": k} for o, (_, v, k) in zip(obs, cases)]
    hint, a_prev, active = [0] * n, [[0.0] * 6 for _ in range(n)], list(range(n))
    for _ in range(max_steps):
        acts, ids = [], []
        for i in list(active):
            o, path = obs[i], paths[i]
            hint[i] = crc.nearest(path, o, hint[i])
            if hint[i] >= len(cases[i][0]) - int(70 / DS):
                active.remove(i)
                continue
            if policy is None:
                a = controller.act(i, o, path, hint[i], cases[i][1], a_prev[i], cases[i][2])
            else:
                with torch.no_grad():
                    ob, _, _ = features_g(torch.tensor([[o[k] for k in STATE_G]], dtype=torch.float32), torch.tensor([[o["x"], o["y"], o["yaw"]]], dtype=torch.float32),
                                          torch.tensor([hint[i] * DS], dtype=torch.float32), torch.tensor([i]), torch.tensor([cases[i][1]], dtype=torch.float32),
                                          torch.tensor([a_prev[i]], dtype=torch.float32), tp, wm, roads, torch.tensor([cases[i][2]]))
                    a = policy(ob)[0].tolist()
            if noise > 0:
                a = [max(-1.0, min(1.0, a[0] + noise * rng.standard_normal())), max(crc.A_MIN, min(crc.A_MAX, a[1] + 3 * noise * rng.standard_normal())),
                     *[max(-1.0, min(1.0, f + 4 * noise * rng.standard_normal())) for f in a[2:6]]]
            acts.append(a)
            ids.append(i)
            logs[i]["s"].append(hint[i] * DS)
        if not ids:
            break
        new = pool.step([a[0] for a in acts], [a[1] for a in acts], ids=ids, f_cmd=[a[2:6] for a in acts])
        for i, a, o in zip(ids, acts, new):
            obs[i] = o
            L, path = logs[i], paths[i]
            xi = torch.tensor([[o[k] for k in STATE_G]], dtype=torch.float32)
            j = crc.nearest(path, o, hint[i])
            lat = -math.sin(path[j, 2]) * (o["x"] - path[j, 0]) + math.cos(path[j, 2]) * (o["y"] - path[j, 1])
            head = math.atan2(math.sin(o["yaw"] - path[j, 2]), math.cos(o["yaw"] - path[j, 2]))
            r, _f, ltr = reward_g(xi, torch.tensor([a]), torch.tensor([a_prev[i]]), torch.tensor([lat]), torch.tensor([head]), torch.tensor([L["v_ref"]]))
            L["x"].append(xi[0].tolist())
            L["pose"].append([o["x"], o["y"], o["yaw"]])
            L["a"].append(a)
            L["r"].append(float(r))
            L["ltr"].append(float(ltr))
            L["lat"].append(lat)
            L["az"].append(o["az"])
            L["tilt"].append(o["tilt"])
            a_prev[i] = a
            if not o["ok"] or o["tilt"] > 0.35 or abs(lat) > crc.LAT_FAIL:
                L["end"] = "rolled" if o["tilt"] > 0.35 or not o["ok"] else "off_path"
                active.remove(i)
    for L, p in zip(logs, paths):
        L["path"] = p
        L["u"] = np.concatenate([np.array(L["a"]).reshape(-1, 6), road_input(roads, L["rid"], L["pose"])], -1) if L["a"] else np.zeros((0, 10))
        lift = np.array(L["ltr"]) > 0.98
        L["metrics"] = {"end": L["end"], "return": float(np.sum(L["r"])), "steps": len(L["a"]), "mean_abs_lat": float(np.mean(np.abs(L["lat"]))) if L["lat"] else None,
                        "rms_az": float(np.sqrt(np.mean(np.square(L["az"])))) if L["az"] else None, "two_wheel_lift_s": float(lift.sum() * DT),
                        "max_tilt_deg": float(np.degrees(np.max(L["tilt"]))), "mean_speed": float(np.mean([x[0] for x in L["x"]])), "v_ref": L["v_ref"]}
    return logs


def evaluate_g(pool, cases, roads, controller, workers, **kw):
    logs = []
    for i in range(0, len(cases), workers):
        logs += drive_g(pool, cases[i:i + workers], roads, controller, **kw)
    m = [L["metrics"] for L in logs]
    return {"rolled": sum(x["end"] == "rolled" for x in m), "off_path": sum(x["end"] == "off_path" for x in m), "episodes": len(m),
            "mean_return": float(np.mean([x["return"] for x in m])), "mean_abs_lat": float(np.mean([x["mean_abs_lat"] for x in m if x["mean_abs_lat"] is not None])),
            "rms_az": float(np.mean([x["rms_az"] for x in m if x["rms_az"] is not None])), "two_wheel_lift_s": float(sum(x["two_wheel_lift_s"] for x in m)),
            "mean_speed": float(np.mean([x["mean_speed"] for x in m])), "per_episode": m}, logs


def collect_random(pool, roads, n, seed, steps, workers, log):
    """Random-excitation data on the training roads: open-loop steering / acceleration plans (chrono_lift_check), and
    suspension forces that are either slow random noise or preview skyhook plus smaller noise."""
    rng = np.random.default_rng(seed)
    eps = []
    for start in range(0, n, workers):
        k = min(workers, n - start)
        plans = [episode_plan(rng, steps)[1] for _ in range(k)]
        rids = [int(rng.integers(0, len(roads.items))) for _ in range(k)]
        mode = [rng.random() < 0.5 for _ in range(k)]  # True: preview skyhook + noise
        obs = pool.reset([{**CFG, "v": float(rng.uniform(8.0, 17.0)), **roads.cfg(r)} for r in rids], ids=range(k))
        L = [{"x": [[o[s] for s in STATE_G]], "pose": [[o["x"], o["y"], o["yaw"]]], "a": [], "tilt": [o["tilt"]]} for o in obs]
        ou = np.zeros((k, 4))
        active = list(range(k))
        for t in range(steps):
            ou += -ou * DT / 0.3 + 0.5 * math.sqrt(2 * DT / 0.3) * rng.standard_normal((k, 4))  # OU noise, std 0.5, 0.3 s
            ids, acts = [], []
            for i in active:
                o = obs[i]
                f = (np.clip(skyhook(o, 1.0) + preview_ff(o, roads.geo, roads.items[rids[i]][0], K_FF) + 0.4 * ou[i], -1, 1) if mode[i] else np.clip(ou[i], -1, 1))
                acts.append([*plans[i][t], *f.tolist()])
                ids.append(i)
            if not ids:
                break
            new = pool.step([a[0] for a in acts], [a[1] for a in acts], ids=ids, f_cmd=[a[2:] for a in acts])
            for i, a, o in zip(ids, acts, new):
                obs[i] = o
                L[i]["x"].append([o[s] for s in STATE_G])
                L[i]["pose"].append([o["x"], o["y"], o["yaw"]])
                L[i]["a"].append(a)
                L[i]["tilt"].append(o["tilt"])
                if not o["ok"] or abs(o["y"]) > 17.0 or o["x"] > 225.0:
                    active.remove(i)
        for i in range(k):
            if len(L[i]["a"]) >= 12:
                eps.append({"x": np.array(L[i]["x"]), "actions": np.concatenate([np.array(L[i]["a"]), road_input(roads, rids[i], L[i]["pose"])], -1),
                            "tilt": np.array(L[i]["tilt"])})
        if (start // workers) % 20 == 0:
            log(f"  {len(eps)}/{n} random episodes")
    return eps


def to_episodes(logs):
    return [{"x": np.array(L["x"]), "actions": L["u"], "tilt": np.array(L["tilt"])} for L in logs if len(L["a"]) >= 12]


def sample_starts_g(logs, n, rng, hist):
    eligible = [(e, t) for e, L in enumerate(logs) for t in range(max(1, hist - 1), len(L["a"]))]
    sel = [eligible[i] for i in rng.integers(0, len(eligible), n)]
    paths = torch.as_tensor(np.stack([np.pad(L["path"], ((0, 2400 - len(L["path"])), (0, 0)), mode="edge") for L in logs]), dtype=torch.float32)

    def f(key, off):
        return torch.tensor(np.array([logs[e][key][t + off] for e, t in sel]), dtype=torch.float32)

    Xh = torch.stack([f("x", -i) for i in reversed(range(hist))])
    Ah = torch.stack([f("u", -i) for i in reversed(range(1, hist))])
    return (Xh, Ah, f("pose", 0), f("s", 0), torch.tensor([e for e, _ in sel]), torch.tensor([logs[e]["v_ref"] for e, _ in sel], dtype=torch.float32),
            torch.tensor([logs[e]["rid"] for e, _ in sel])), paths


def wm_update(wm, opt, eps, k, n, rng):
    idx = windows(eps, k, wm.hist)
    loss = None
    for _ in range(n):
        Xb, Ab = batch(eps, [idx[i] for i in rng.integers(0, len(idx), 512)], k, wm.hist)
        loss = (((predict(wm, Xb, Ab) - Xb[wm.hist:]) / wm.std) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt.step()
    return float(loss)


# ---------------------------------------------------------------------- main


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lift-data", required=True, help="flat-road random-excitation data (chrono_lift_check --cache) for the MPC's identification")
    ap.add_argument("--road-dir", default="logs/roads_gas1")
    ap.add_argument("--train-roads", type=int, default=64)
    ap.add_argument("--random-episodes", type=int, default=400)
    ap.add_argument("--random-cache", default="logs/chrono_gas1_random.npz")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--episodes-per-iter", type=int, default=8)
    ap.add_argument("--wm-pretrain", type=int, default=4000)
    ap.add_argument("--bc-updates", type=int, default=600, help="behaviour-clone updates before the SuperMimic loop")
    ap.add_argument("--wm-updates", type=int, default=300)
    ap.add_argument("--policy-updates", type=int, default=150)
    ap.add_argument("--horizon", type=int, default=75)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--eval-paths", type=int, default=8)
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--corridor-weight", type=float, default=0.0, help="training objective: corridor barrier beyond 0.4 m (imagination only)")
    ap.add_argument("--fail-penalty", type=float, default=0.0, help="training objective: penalty per unit failure probability (imagination only)")
    ap.add_argument("--ltr-weight", type=float, default=0.0, help="training objective: load-transfer barrier beyond 0.6 (imagination only)")
    ap.add_argument("--extra-eval-paths", type=int, default=0, help="a second held-out evaluation set: this many paths x 2 speeds on new roads")
    ap.add_argument("--save-dir", default=None, help="save policy and world model at every evaluation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--python", default=None)
    ap.add_argument("--out", default="logs/chrono_gas1.json")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    def log(m):
        print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    train_rng, eval_rng = np.random.default_rng(args.seed + 1), np.random.default_rng(args.seed + 2000)
    report = {"args": {k: v for k, v in vars(args).items() if k != "python"}, "curve": [], "chrono_steps": {}}
    with ChronoPool(args.workers, args.python) as pool:
        pool.reset([{**CFG, "v": 5.0}], ids=[0])
        geo = pool.info(ids=[0])[0]
        roads = Roads([*range(args.train_roads), *range(5000, 5000 + args.eval_paths), *range(6000, 6000 + args.extra_eval_paths)], args.road_dir, geo)
        eval_cases = []
        for e in range(args.eval_paths):
            p, _ = crc.lane_change_path(eval_rng)
            eval_cases += [(p, v, args.train_roads + e) for v in crc.EVAL_SPEEDS]
        extra_rng, extra_cases = np.random.default_rng(args.seed + 3000), []
        for e in range(args.extra_eval_paths):
            p, _ = crc.lane_change_path(extra_rng)
            extra_cases += [(p, v, args.train_roads + args.eval_paths + e) for v in crc.EVAL_SPEEDS]

        def failed_traces(logs):
            return [{"case": i, "v_ref": L["v_ref"], "end": L["end"], "lat": L["lat"], "vx": [x[0] for x in L["x"]], "steer": [a[0] for a in L["a"]],
                     "ltr": L["ltr"], "tilt": L["tilt"]} for i, L in enumerate(logs) if L["end"] != "completed"]
        ident = identify(args.lift_data)

        def show(name, ev):
            log(f"{name:>13}: rolled {ev['rolled']}/{ev['episodes']}, off path {ev['off_path']}, return {ev['mean_return']:.1f}, |lat| {ev['mean_abs_lat']:.3f} m, "
                f"RMS az {ev['rms_az']:.2f} m/s2, lift {ev['two_wheel_lift_s']:.1f} s, speed {ev['mean_speed']:.1f} m/s")

        base = {}
        for name, ctl in (("ssf_passive", Hierarchical(None, roads, False)), ("ssf_preview", Hierarchical(None, roads, True)),
                          ("mpc_passive", Hierarchical(RollMPC(ident), roads, False)), ("hierarchical", Hierarchical(RollMPC(ident), roads, True))):
            base[name], _ = evaluate_g(pool, eval_cases, roads, ctl, args.workers)
            show(name, base[name])
            if extra_cases:
                base[name + "_extra"], _ = evaluate_g(pool, extra_cases, roads, ctl, args.workers)
                show(name + " [extra]", base[name + "_extra"])
            report["baselines"] = base
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, indent=1))
        if args.baselines_only:
            return 0
        cache = Path(args.random_cache)
        if cache.exists():
            rand_eps = list(np.load(cache, allow_pickle=True)["eps"])
        else:
            log("collecting random-excitation episodes on the training roads")
            rand_eps = collect_random(pool, roads, args.random_episodes, args.seed + 7, 300, args.workers, log)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, eps=np.array(rand_eps, dtype=object))
        report["chrono_steps"]["random_excitation"] = int(sum(len(ep["actions"]) for ep in rand_eps))
        X = np.concatenate([ep["x"] for ep in rand_eps])
        D = np.concatenate([np.diff(ep["x"], axis=0) for ep in rand_eps])
        wm = LiftWM(len(STATE_G), 10, [512, 512, 512], X.mean(0), X.std(0) + 1e-6, D.std(0) + 1e-6, hist=2)
        wopt = torch.optim.Adam(wm.parameters(), lr=1e-3)
        for g in range(args.wm_pretrain // 500):
            log(f"  world model {(g + 1) * 500} updates, 8-step loss {wm_update(wm, wopt, rand_eps, 8, 500, rng):.4f}")
        for gr in wopt.param_groups:
            gr["lr"] = 3e-4
        # task data and behaviour clone: the SSF limiter with preview skyhook (cheap, decent), with noise
        task_logs = []
        for _ in range(2):
            cases = [(crc.lane_change_path(train_rng)[0], float(train_rng.uniform(*crc.TRAIN_SPEEDS)), int(train_rng.integers(0, args.train_roads)))
                     for _ in range(args.episodes_per_iter)]
            for i in range(0, len(cases), args.workers):
                task_logs += drive_g(pool, cases[i:i + args.workers], roads, Hierarchical(None, roads, True), noise=0.05, rng=rng)
        st, paths = sample_starts_g(task_logs, 2, rng, wm.hist)
        policy = Policy6(features_g(st[0][-1], st[2], st[3], st[4], st[5], st[1][-1][:, :6], paths, wm, roads, st[6])[0].shape[-1])
        popt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        eligible = [(e, t) for e, L in enumerate(task_logs) for t in range(1, len(L["a"]))]
        tp = torch.as_tensor(np.stack([np.pad(L["path"], ((0, 2400 - len(L["path"])), (0, 0)), mode="edge") for L in task_logs]), dtype=torch.float32)
        for _ in range(args.bc_updates):
            sel = [eligible[i] for i in rng.integers(0, len(eligible), 512)]
            x = torch.tensor(np.array([task_logs[e]["x"][t] for e, t in sel]), dtype=torch.float32)
            pose = torch.tensor(np.array([task_logs[e]["pose"][t] for e, t in sel]), dtype=torch.float32)
            s = torch.tensor([task_logs[e]["s"][t] for e, t in sel], dtype=torch.float32)
            a_prev = torch.tensor(np.array([task_logs[e]["a"][t - 1] for e, t in sel]), dtype=torch.float32)
            tgt = torch.tensor(np.array([task_logs[e]["a"][t] for e, t in sel]), dtype=torch.float32)
            ob, _, _ = features_g(x, pose, s, torch.tensor([e for e, _ in sel]), torch.tensor([task_logs[e]["v_ref"] for e, _ in sel], dtype=torch.float32),
                                  a_prev, tp, wm, roads, torch.tensor([task_logs[e]["rid"] for e, _ in sel]))
            loss = (((policy(ob) - tgt) / torch.tensor([1.0, 1.5, 1.0, 1.0, 1.0, 1.0])) ** 2).mean()
            popt.zero_grad()
            loss.backward()
            popt.step()
        for gr in popt.param_groups:
            gr["lr"] = 3e-4
        ev, _ = evaluate_g(pool, eval_cases, roads, None, args.workers, policy=policy, wm=wm)
        show("behaviour clone", ev)
        report["behaviour_clone"] = ev
        report["curve"].append({"iteration": 0, **{k: v for k, v in ev.items() if k != "per_episode"}})
        for it in range(1, args.iterations + 1):
            cases = [(crc.lane_change_path(train_rng)[0], float(train_rng.uniform(*crc.TRAIN_SPEEDS)), int(train_rng.integers(0, args.train_roads)))
                     for _ in range(args.episodes_per_iter)]
            for i in range(0, len(cases), args.workers):
                task_logs += drive_g(pool, cases[i:i + args.workers], roads, None, policy=policy, wm=wm, noise=0.05, rng=rng)
            wl = wm_update(wm, wopt, rand_eps + to_episodes(task_logs) * 4, 8, args.wm_updates, rng)
            imag = []
            for _ in range(args.policy_updates):
                st, paths = sample_starts_g(task_logs, args.batch, rng, wm.hist)
                J = imagine_g(wm, policy, st, paths, args.horizon, roads, args.corridor_weight, args.fail_penalty, args.ltr_weight)
                loss = -J.mean() / args.horizon
                popt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                popt.step()
                imag.append(float(J.mean()) / args.horizon)
            row = {"iteration": it, "task_steps": sum(len(L["a"]) for L in task_logs), "wm_loss": wl, "imagined_reward_per_step": float(np.mean(imag[-20:]))}
            if it % 5 == 0 or it == args.iterations:
                ev, ev_logs = evaluate_g(pool, eval_cases, roads, None, args.workers, policy=policy, wm=wm)
                row.update({k: v for k, v in ev.items() if k != "per_episode"})
                report["final"] = {**ev, "failed_traces": failed_traces(ev_logs)}
                show(f"iteration {it}", ev)
                if extra_cases:
                    ex, _ = evaluate_g(pool, extra_cases, roads, None, args.workers, policy=policy, wm=wm)
                    row["extra"] = {k: v for k, v in ex.items() if k != "per_episode"}
                    report["final_extra"] = ex
                    show(f"iteration {it} [extra]", ex)
                if args.save_dir:
                    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                    torch.save({"policy": policy.state_dict(), "wm": wm.state_dict()}, Path(args.save_dir) / f"it{it}.pt")
            else:
                log(f"iteration {it}: imagined reward/step {row['imagined_reward_per_step']:.3f}, wm loss {wl:.4f}, task steps {row['task_steps']}")
            report["curve"].append(row)
            report["chrono_steps"]["task"] = row["task_steps"]
            Path(args.out).write_text(json.dumps(report, indent=1))
    log(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
