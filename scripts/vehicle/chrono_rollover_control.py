"""SuperMimic's recipe on a task whose dynamics a single-track model cannot describe: rollover-safe evasive driving
with a top-heavy vehicle in Project Chrono (VEHICLE_PROPOSAL.md section 11.12).

Vehicle: Chrono's HMMWV with an 800 kg roof load (static stability factor about 0.8: it lifts two wheels and rolls
over in hard manoeuvres), rigid ground, TMeasy tyres. Task: follow a single or double lane change (offset 2.5-4 m
over 16-26 m) at a reference speed of 10-18 m/s (evaluation: 13 and 17 m/s); some combinations cannot be driven at the reference speed without
lifting two wheels, so a good controller slows down in time. Reward per step (the same formula scores every
controller in Chrono): path following (lateral and heading), speed up to the reference, minus penalties on the lateral
load transfer ratio LTR = |F_left - F_right| / F_total above 0.75 (1 = one side off the ground), on roll above
0.08 rad and on steering rate; an episode fails when the roll exceeds 0.35 rad (about 20 deg, past recovery here) or
the car is more than 0.6 m off the path (out of its lane: the lane change avoids an obstacle).

Controllers, all driving the same Chrono vehicle through the same commands (normalised steering, acceleration):
* ``pure_pursuit``: geometric path following (preview 0.6 s) at the reference speed (no rollover awareness);
* ``ssf_limiter``: the same steering plus the classical rollover speed limit: never faster than
  sqrt(0.7 g SSF / kappa) for the largest path curvature within the next 2.5 s (SSF = 0.83 from the vehicle's
  track and CoG height);
* ``supermimic``: a policy trained by backpropagation through a learned world model (``chrono_lift_check.LiftWM``,
  which predicts the tyre normal loads) over 1.5 s imagined rollouts, in an online loop: drive in Chrono, refit the
  world model, improve the policy in imagination. The policy starts as a behaviour clone of ``ssf_limiter``.

    SUPERMIMIC_CHRONO_PYTHON=chrono_env/bin/python PYTHONPATH=src python scripts/vehicle/chrono_rollover_control.py \\
        --lift-data logs/chrono_lift_data.npz --out docs/results/vehicle_chrono_rollover_control.json
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

from chrono_lift_check import CFG as LIFT_CFG
from chrono_lift_check import STATE, LiftWM, batch, windows

from supermimic.vehicles.chrono_env import ChronoPool

DT, DS = 0.02, 0.25
K_STEER, WHEELBASE, SSF = 0.49, 3.31, 0.83
A_MIN, A_MAX = -2.0, 1.0
LAT_FAIL = 0.6  # [m] leaving the lane (an obstacle beside the path) ends the episode
OFFSET, TRANSITION = (2.5, 4.0), (16.0, 26.0)  # lane-change offset and transition length ranges [m]
EVAL_SPEEDS, TRAIN_SPEEDS = (13.0, 17.0), (10.0, 18.0)
LOOK = 0.6  # [s] pure-pursuit preview: tight enough to really follow the lane change
PREVIEW = [4.0, 8.0, 12.0, 16.0, 24.0, 32.0]
IX = {k: i for i, k in enumerate(STATE)}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lift-data", required=True, help="random-excitation data of chrono_lift_check.py (its --cache)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--episodes-per-iter", type=int, default=8)
    ap.add_argument("--wm-pretrain", type=int, default=4000)
    ap.add_argument("--wm-updates", type=int, default=300)
    ap.add_argument("--policy-updates", type=int, default=150)
    ap.add_argument("--horizon", type=int, default=75)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--eval-paths", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--python", default=None)
    ap.add_argument("--out", default="logs/chrono_rollover_control.json")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------- paths


def lane_change_path(rng, lead=12.0, tail=80.0):
    """(P, 4) array x, y, yaw, kappa every DS metres, starting at the origin heading +x."""
    A = rng.uniform(*OFFSET) * rng.choice([-1, 1])
    Lc = rng.uniform(*TRANSITION)
    double = rng.random() < 0.5
    hold = rng.uniform(8.0, 16.0)
    length = lead + Lc + (hold + Lc if double else 0) + tail
    xs = np.arange(0, length, DS / 2)

    def ramp(u):
        return 0.5 * (1 - np.cos(np.pi * np.clip(u, 0, 1)))

    y = A * ramp((xs - lead) / Lc)
    if double:
        y -= A * ramp((xs - lead - Lc - hold) / Lc)
    seg = np.hypot(np.diff(xs), np.diff(y))
    s = np.concatenate([[0], np.cumsum(seg)])
    s_new = np.arange(0, s[-1], DS)
    X, Y = np.interp(s_new, s, xs), np.interp(s_new, s, y)
    yaw = np.unwrap(np.arctan2(np.gradient(Y), np.gradient(X)))
    kappa = np.gradient(yaw) / DS
    return np.stack([X, Y, yaw, kappa], -1), {"offset": float(A), "length": float(Lc), "double": bool(double)}


def placed(path, x0, y0, yaw0):
    c, s = math.cos(yaw0), math.sin(yaw0)
    out = path.copy()
    out[:, 0] = x0 + c * path[:, 0] - s * path[:, 1]
    out[:, 1] = y0 + s * path[:, 0] + c * path[:, 1]
    out[:, 2] = path[:, 2] + yaw0
    return out


# ---------------------------------------------------------------------- task (torch, shared by imagination and Chrono)


def path_at(paths, pid, s):
    """paths (N, P, 4); pid (B,); s (B,) or (B, m) arc length -> interpolated x, y, yaw, kappa."""
    P = paths.shape[1]
    u = (s / DS).clamp(0, P - 1.001)
    i0 = u.floor().long()
    f = (u - i0).unsqueeze(-1)
    pp = paths[pid.view(-1, *([1] * (s.dim() - 1))).expand_as(i0), i0]
    pn = paths[pid.view(-1, *([1] * (s.dim() - 1))).expand_as(i0), i0 + 1]
    v = pp + f * (pn - pp)
    return v[..., 0], v[..., 1], v[..., 2], v[..., 3]


def features(x, pose, s, pid, v_ref, a_prev, paths, wm):
    X, Y, yaw = pose[:, 0], pose[:, 1], pose[:, 2]
    px, py, pyaw, _ = path_at(paths, pid, s)
    e_lat = -torch.sin(pyaw) * (X - px) + torch.cos(pyaw) * (Y - py)
    e_head = torch.atan2(torch.sin(yaw - pyaw), torch.cos(yaw - pyaw))
    d = torch.tensor(PREVIEW, dtype=x.dtype)
    qx, qy, _, _ = path_at(paths, pid, s[:, None] + d[None])
    dx, dy = qx - X[:, None], qy - Y[:, None]
    c, sn = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    bx, by = c * dx + sn * dy, -sn * dx + c * dy
    z = (x - wm.mean) / wm.std
    return torch.cat([z, a_prev, torch.stack([e_lat, e_head], -1), bx / 10, by / 10, (v_ref / 15)[:, None]], -1), e_lat, e_head


def reward(x, a, a_prev, e_lat, e_head, v_ref):
    fz = x[:, 12:16].clamp_min(0.0)
    ltr = ((fz[:, 0] + fz[:, 2]) - (fz[:, 1] + fz[:, 3])).abs() / fz.sum(-1).clamp_min(1e3)
    roll = x[:, IX["roll_b"]]
    r = (torch.exp(-(e_lat / 0.4) ** 2) + 0.3 * torch.exp(-(e_head / 0.15) ** 2) + 0.6 * (x[:, IX["vx"]] / v_ref).clamp(max=1.0)
         - 4.0 * torch.relu(ltr - 0.75) - 10.0 * torch.relu(roll.abs() - 0.08) - 0.1 * (a[:, 0] - a_prev[:, 0]) ** 2)
    fail = torch.sigmoid((roll.abs() - 0.35) / 0.03) + torch.sigmoid((e_lat.abs() - LAT_FAIL) / 0.1)
    return r, fail.clamp(0, 1), ltr


class Policy(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 2))

    def forward(self, obs):
        u = torch.tanh(self.net(obs))
        return torch.stack([u[:, 0], A_MIN + (u[:, 1] + 1) * 0.5 * (A_MAX - A_MIN)], -1)


# ---------------------------------------------------------------------- controllers in Chrono


def baseline_action(kind, o, path, j, v_ref):
    look = max(3.0, LOOK * max(o["vx"], 1.0))
    k = min(len(path) - 1, j + int(look / DS))
    dx, dy = path[k, 0] - o["x"], path[k, 1] - o["y"]
    alpha = math.atan2(dy, dx) - o["yaw"]
    steer = math.atan2(2 * WHEELBASE * math.sin(alpha), max(math.hypot(dx, dy), 1e-3)) / K_STEER
    v_t = v_ref
    if kind == "ssf_limiter":
        ahead = path[j: j + int(2.5 * max(o["vx"], 1.0) / DS) + 1, 3]
        kmax = float(np.abs(ahead).max()) if len(ahead) else 0.0
        if kmax > 1e-4:
            v_t = min(v_ref, math.sqrt(0.7 * 9.81 * SSF / kmax))
    return [max(-1.0, min(1.0, steer)), max(A_MIN, min(A_MAX, 1.5 * (v_t - o["vx"])))]


def nearest(path, o, hint):
    lo, hi = max(0, hint - 40), min(len(path), hint + 200)
    d2 = (path[lo:hi, 0] - o["x"]) ** 2 + (path[lo:hi, 1] - o["y"]) ** 2
    return lo + int(np.argmin(d2))


def drive(pool, cases, controller, policy=None, wm=None, noise=0.0, rng=None, max_steps=700):
    """cases: list of (path (P, 4) at the origin, v_ref); one Chrono episode each, len(cases) <= workers.
    Returns per-episode logs (states, actions, poses, progress, rewards, metrics)."""
    n = len(cases)
    obs = pool.reset([{**LIFT_CFG, "v": v} for _, v in cases], ids=range(n))
    paths = [placed(p, o["x"], o["y"], o["yaw"]) for (p, _), o in zip(cases, obs)]
    tp = torch.as_tensor(np.stack([np.pad(p, ((0, 2400 - len(p)), (0, 0)), mode="edge") for p in paths]), dtype=torch.float32)
    logs = [{"x": [[o[k] for k in STATE]], "pose": [[o["x"], o["y"], o["yaw"]]], "a": [], "s": [], "r": [], "ltr": [], "lat": [],
             "tilt": [o["tilt"]], "end": "completed", "v_ref": v} for o, (_, v) in zip(obs, cases)]
    hint = [0] * n
    a_prev = [[0.0, 0.0] for _ in range(n)]
    active = list(range(n))
    for _ in range(max_steps):
        acts, ids = [], []
        for i in list(active):
            o, path = obs[i], paths[i]
            hint[i] = nearest(path, o, hint[i])
            if hint[i] >= len(cases[i][0]) - int(70 / DS):  # reached the end of the manoeuvre (+ lead-out)
                active.remove(i)
                continue
            if controller in ("pure_pursuit", "ssf_limiter"):
                a = baseline_action(controller, o, path, hint[i], cases[i][1])
            else:
                with torch.no_grad():
                    x = torch.tensor([[o[k] for k in STATE]], dtype=torch.float32)
                    pose = torch.tensor([[o["x"], o["y"], o["yaw"]]], dtype=torch.float32)
                    ob, _, _ = features(x, pose, torch.tensor([hint[i] * DS], dtype=torch.float32), torch.tensor([i]),
                                        torch.tensor([cases[i][1]], dtype=torch.float32), torch.tensor([a_prev[i]], dtype=torch.float32), tp, wm)
                    a = policy(ob)[0].tolist()
            if noise > 0:
                a = [max(-1.0, min(1.0, a[0] + noise * rng.standard_normal())), max(A_MIN, min(A_MAX, a[1] + 3 * noise * rng.standard_normal()))]
            acts.append(a)
            ids.append(i)
            logs[i]["s"].append(hint[i] * DS)
        if not ids:
            break
        new = pool.step([a[0] for a in acts], [a[1] for a in acts], ids=ids)
        for i, a, o in zip(ids, acts, new):
            obs[i] = o
            L = logs[i]
            xi = torch.tensor([[o[k] for k in STATE]], dtype=torch.float32)
            path = paths[i]
            j = nearest(path, o, hint[i])
            lat = -math.sin(path[j, 2]) * (o["x"] - path[j, 0]) + math.cos(path[j, 2]) * (o["y"] - path[j, 1])
            head = math.atan2(math.sin(o["yaw"] - path[j, 2]), math.cos(o["yaw"] - path[j, 2]))
            r, _fail, ltr = reward(xi, torch.tensor([a]), torch.tensor([a_prev[i]]), torch.tensor([lat]), torch.tensor([head]), torch.tensor([L["v_ref"]]))
            L["x"].append(xi[0].tolist())
            L["pose"].append([o["x"], o["y"], o["yaw"]])
            L["a"].append(a)
            L["r"].append(float(r))
            L["ltr"].append(float(ltr))
            L["lat"].append(lat)
            L["tilt"].append(o["tilt"])
            a_prev[i] = a
            if not o["ok"] or o["tilt"] > 0.35 or abs(lat) > LAT_FAIL:
                L["end"] = "rolled" if o["tilt"] > 0.35 or not o["ok"] else "off_path"
                active.remove(i)
    for L, p in zip(logs, paths):
        L["path"] = p
        lift = np.array(L["ltr"]) > 0.98
        L["metrics"] = {"end": L["end"], "return": float(np.sum(L["r"])), "steps": len(L["a"]), "mean_abs_lat": float(np.mean(np.abs(L["lat"]))) if L["lat"] else None,
                        "max_ltr": float(np.max(L["ltr"])) if L["ltr"] else None, "two_wheel_lift_s": float(lift.sum() * DT),
                        "max_tilt_deg": float(np.degrees(np.max(L["tilt"]))), "mean_speed": float(np.mean([x[0] for x in L["x"]])), "v_ref": L["v_ref"]}
    return logs


def evaluate(pool, cases, controller, args, **kw):
    out = []
    for i in range(0, len(cases), args.workers):
        out += drive(pool, cases[i:i + args.workers], controller, **kw)
    m = [L["metrics"] for L in out]
    return {"rolled": sum(q["end"] == "rolled" for q in m), "off_path": sum(q["end"] == "off_path" for q in m), "episodes": len(m),
            "mean_return": float(np.mean([q["return"] for q in m])), "mean_abs_lat": float(np.mean([q["mean_abs_lat"] for q in m])),
            "two_wheel_lift_s": float(np.sum([q["two_wheel_lift_s"] for q in m])), "mean_speed": float(np.mean([q["mean_speed"] for q in m])),
            "max_tilt_deg": float(np.max([q["max_tilt_deg"] for q in m])), "per_episode": m}, out


# ---------------------------------------------------------------------- learning


def wm_update(wm, opt, eps, k, n, rng):
    idx = windows(eps, k)
    std = wm.std
    loss = None
    for _ in range(n):
        sel = [idx[i] for i in rng.integers(0, len(idx), 512)]
        Xb, Ab = batch(eps, sel, k)
        pred = wm.rollout(Xb[1], Xb[0], Ab[1:], Ab[0])
        loss = (((pred - Xb[2:]) / std) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt.step()
    return float(loss)


def to_episodes(logs):
    return [{"x": np.array(L["x"]), "actions": np.array(L["a"]) if L["a"] else np.zeros((0, 2)), "tilt": np.array(L["tilt"])} for L in logs if len(L["a"]) >= 12]


def imagine(wm, policy, starts, paths, H):
    x_prev, x, a_prev, pose, s, pid, v_ref = starts
    c = torch.ones(x.shape[0])
    J = torch.zeros(x.shape[0])
    for _ in range(H):
        ob, _, _ = features(x, pose, s, pid, v_ref, a_prev, paths, wm)
        a = policy(ob)
        x1 = wm.step(x, x_prev, a, a_prev)
        vx, vy, r = x1[:, 0], x1[:, 1], x1[:, 3]
        yaw = pose[:, 2] + r * DT
        X = pose[:, 0] + (vx * torch.cos(yaw) - vy * torch.sin(yaw)) * DT
        Y = pose[:, 1] + (vx * torch.sin(yaw) + vy * torch.cos(yaw)) * DT
        _, _, pyaw, _ = path_at(paths, pid, s)
        s = s + (vx * torch.cos(yaw - pyaw) - vy * torch.sin(yaw - pyaw)) * DT
        pose = torch.stack([X, Y, yaw], -1)
        _, e_lat, e_head = features(x1, pose, s, pid, v_ref, a, paths, wm)
        rew, fail, _ = reward(x1, a, a_prev, e_lat, e_head, v_ref)
        J = J + c * rew
        c = c * (1 - fail)
        x_prev, x, a_prev = x, x1, a
    return J


def sample_starts(task_logs, n, rng):
    eligible = [(e, t) for e, L in enumerate(task_logs) for t in range(1, len(L["a"]))]
    sel = [eligible[i] for i in rng.integers(0, len(eligible), n)]
    paths = torch.as_tensor(np.stack([np.pad(L["path"], ((0, 2400 - len(L["path"])), (0, 0)), mode="edge") for L in task_logs]), dtype=torch.float32)

    def f(key, off):
        return torch.tensor(np.array([task_logs[e][key][t + off] for e, t in sel]), dtype=torch.float32)

    pid = torch.tensor([e for e, _ in sel])
    return (f("x", -1), f("x", 0), f("a", -1), f("pose", 0), f("s", 0), pid, torch.tensor([task_logs[e]["v_ref"] for e, _ in sel], dtype=torch.float32)), paths


def main(argv=None) -> int:
    args = parse(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    def log(m):
        print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    d = np.load(args.lift_data, allow_pickle=True)
    rand_eps = list(d["train"])
    X = np.concatenate([ep["x"] for ep in rand_eps])
    D = np.concatenate([np.diff(ep["x"], axis=0) for ep in rand_eps])
    wm = LiftWM(len(STATE), 2, [512, 512, 512], X.mean(0), X.std(0) + 1e-6, D.std(0) + 1e-6)
    wopt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    log(f"world model: pretraining on {sum(len(ep['x']) for ep in rand_eps)} random-excitation steps")
    for g in range(args.wm_pretrain // 500):
        loss = wm_update(wm, wopt, rand_eps, 8, 500, rng)
        log(f"  {(g + 1) * 500} updates, 8-step loss {loss:.4f}")
    for gr in wopt.param_groups:
        gr["lr"] = 3e-4
    train_rng, eval_rng = np.random.default_rng(args.seed + 1), np.random.default_rng(args.seed + 2000)
    eval_cases = []
    for _ in range(args.eval_paths):
        p, _info = lane_change_path(eval_rng)
        eval_cases += [(p, v) for v in EVAL_SPEEDS]
    report = {"args": {k: v for k, v in vars(args).items() if k != "python"}, "chrono_steps": {"random_excitation": int(sum(len(ep["actions"]) for ep in rand_eps))}, "curve": []}
    with ChronoPool(args.workers, args.python) as pool:
        base = {}
        for c in ("pure_pursuit", "ssf_limiter"):
            base[c], _logs = evaluate(pool, eval_cases, c, args)
            log(f"{c:>13}: rolled {base[c]['rolled']}/{base[c]['episodes']}, off path {base[c]['off_path']}, return {base[c]['mean_return']:.1f}, "
                f"|lat| {base[c]['mean_abs_lat']:.3f} m, lift {base[c]['two_wheel_lift_s']:.1f} s, speed {base[c]['mean_speed']:.1f} m/s")
        report["baselines"] = base
        # task data: the limiter (and a noisy limiter) on training paths
        task_logs = []
        for _ in range(2):
            cases = [(lane_change_path(train_rng)[0], float(train_rng.uniform(*TRAIN_SPEEDS))) for _ in range(args.episodes_per_iter)]
            for i in range(0, len(cases), args.workers):
                task_logs += drive(pool, cases[i:i + args.workers], "ssf_limiter", noise=0.05, rng=rng)
        obs_dim = features(torch.zeros(1, len(STATE)), torch.zeros(1, 3), torch.zeros(1), torch.zeros(1, dtype=torch.long), torch.ones(1),
                           torch.zeros(1, 2), torch.zeros(1, 10, 4), wm)[0].shape[-1]
        policy = Policy(obs_dim)
        popt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        # behaviour clone of the limiter
        _st, paths = sample_starts(task_logs, 1, rng)
        eligible = [(e, t) for e, L in enumerate(task_logs) for t in range(1, len(L["a"]))]
        for _ in range(600):
            sel = [eligible[i] for i in rng.integers(0, len(eligible), 512)]
            x = torch.tensor(np.array([task_logs[e]["x"][t] for e, t in sel]), dtype=torch.float32)
            pose = torch.tensor(np.array([task_logs[e]["pose"][t] for e, t in sel]), dtype=torch.float32)
            s = torch.tensor([task_logs[e]["s"][t] for e, t in sel], dtype=torch.float32)
            a_prev = torch.tensor(np.array([task_logs[e]["a"][t - 1] for e, t in sel]), dtype=torch.float32)
            tgt = torch.tensor(np.array([task_logs[e]["a"][t] for e, t in sel]), dtype=torch.float32)
            ob, _, _ = features(x, pose, s, torch.tensor([e for e, _ in sel]), torch.tensor([task_logs[e]["v_ref"] for e, _ in sel], dtype=torch.float32), a_prev, paths, wm)
            loss = (((policy(ob) - tgt) / torch.tensor([1.0, 1.5])) ** 2).mean()
            popt.zero_grad()
            loss.backward()
            popt.step()
        for gr in popt.param_groups:
            gr["lr"] = 3e-4
        ev, _ = evaluate(pool, eval_cases, "supermimic", args, policy=policy, wm=wm)
        log(f"  behaviour clone: rolled {ev['rolled']}, return {ev['mean_return']:.1f}, |lat| {ev['mean_abs_lat']:.3f}, lift {ev['two_wheel_lift_s']:.1f} s, speed {ev['mean_speed']:.1f}")
        report["behaviour_clone"] = ev
        report["curve"].append({"iteration": 0, "task_steps": sum(len(L["a"]) for L in task_logs), **{k: v for k, v in ev.items() if k != "per_episode"}})
        for it in range(1, args.iterations + 1):
            cases = [(lane_change_path(train_rng)[0], float(train_rng.uniform(*TRAIN_SPEEDS))) for _ in range(args.episodes_per_iter)]
            for i in range(0, len(cases), args.workers):
                task_logs += drive(pool, cases[i:i + args.workers], "supermimic", policy=policy, wm=wm, noise=0.05, rng=rng)
            wl = wm_update(wm, wopt, rand_eps + to_episodes(task_logs) * 4, 8, args.wm_updates, rng)
            imag = []
            for _ in range(args.policy_updates):
                st, paths = sample_starts(task_logs, args.batch, rng)
                J = imagine(wm, policy, st, paths, args.horizon)
                loss = -J.mean() / args.horizon
                popt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                popt.step()
                imag.append(float(J.mean()) / args.horizon)
            row = {"iteration": it, "task_steps": sum(len(L["a"]) for L in task_logs), "wm_loss": wl, "imagined_reward_per_step": float(np.mean(imag[-20:]))}
            if it % 5 == 0 or it == args.iterations:
                ev, _ = evaluate(pool, eval_cases, "supermimic", args, policy=policy, wm=wm)
                row.update({k: v for k, v in ev.items() if k != "per_episode"})
                report["final"] = ev
                log(f"iteration {it}: rolled {ev['rolled']}/{ev['episodes']}, off path {ev['off_path']}, return {ev['mean_return']:.1f}, |lat| {ev['mean_abs_lat']:.3f} m, "
                    f"lift {ev['two_wheel_lift_s']:.1f} s, speed {ev['mean_speed']:.1f} m/s; imagined reward/step {row['imagined_reward_per_step']:.3f}, wm loss {wl:.4f}")
            else:
                log(f"iteration {it}: imagined reward/step {row['imagined_reward_per_step']:.3f}, wm loss {wl:.4f}, task steps {row['task_steps']}")
            report["curve"].append(row)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, indent=1))
    report["chrono_steps"]["task"] = sum(len(L["a"]) for L in task_logs)
    Path(args.out).write_text(json.dumps(report, indent=1))
    log(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
