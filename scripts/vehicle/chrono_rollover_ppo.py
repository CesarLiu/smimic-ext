"""Model-free counterpart of ``chrono_rollover_control.py``: PPO on the same task, directly in Chrono, from the same
starting point (a behaviour clone of the SSF speed limiter), with a budget of Chrono steps equal to what SuperMimic
used there (its random-excitation data plus its task data) or a multiple of it (VEHICLE_PROPOSAL.md section 11.12).

Same observation features, reward, action bounds, training paths and speeds, and the same held-out evaluation; the
policy's mean is the same network, with a learned Gaussian standard deviation per action.

    SUPERMIMIC_CHRONO_PYTHON=chrono_env/bin/python PYTHONPATH=src python scripts/vehicle/chrono_rollover_ppo.py \\
        --lift-data logs/chrono_lift_data.npz --steps 140000 --out docs/results/vehicle_chrono_rollover_ppo.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chrono_lift_check import CFG as LIFT_CFG
from chrono_lift_check import STATE
from chrono_rollover_control import (
    A_MAX,
    A_MIN,
    DS,
    EVAL_SPEEDS,
    LAT_FAIL,
    TRAIN_SPEEDS,
    Policy,
    drive,
    evaluate,
    features,
    lane_change_path,
    nearest,
    placed,
    reward,
    sample_starts,
)

from supermimic.vehicles.chrono_env import ChronoPool


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lift-data", required=True)
    ap.add_argument("--steps", type=int, default=140000, help="Chrono control steps of PPO training")
    ap.add_argument("--rollout", type=int, default=512, help="steps per worker between PPO updates")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-paths", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=35000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--python", default=None)
    ap.add_argument("--out", default="logs/chrono_rollover_ppo.json")
    return ap.parse_args(argv)


class Env:
    """The task over a ChronoPool: auto-reset with a new training path and speed; observation features as the policy
    sees them in ``chrono_rollover_control``."""

    def __init__(self, pool, n, norm, rng):
        self.pool, self.n, self.norm, self.rng = pool, n, norm, rng
        self.obs, self.paths, self.vref, self.hint, self.a_prev, self.end_idx = [None] * n, [None] * n, [0.0] * n, [0] * n, [[0.0, 0.0]] * n, [0] * n
        for i in range(n):
            self.reset(i)

    def reset(self, i):
        p, _ = lane_change_path(self.rng)
        v = float(self.rng.uniform(*TRAIN_SPEEDS))
        o = self.pool.reset([{**LIFT_CFG, "v": v}], ids=[i])[0]
        self.obs[i], self.paths[i], self.vref[i], self.hint[i], self.a_prev[i] = o, placed(p, o["x"], o["y"], o["yaw"]), v, 0, [0.0, 0.0]
        self.end_idx[i] = len(p) - int(70 / DS)

    def features(self):
        tp = torch.as_tensor(np.stack([np.pad(p, ((0, 2400 - len(p)), (0, 0)), mode="edge") for p in self.paths]), dtype=torch.float32)
        for i in range(self.n):
            self.hint[i] = nearest(self.paths[i], self.obs[i], self.hint[i])
        x = torch.tensor([[o[k] for k in STATE] for o in self.obs], dtype=torch.float32)
        pose = torch.tensor([[o["x"], o["y"], o["yaw"]] for o in self.obs], dtype=torch.float32)
        s = torch.tensor([h * DS for h in self.hint], dtype=torch.float32)
        ob, _, _ = features(x, pose, s, torch.arange(self.n), torch.tensor(self.vref, dtype=torch.float32), torch.tensor(self.a_prev, dtype=torch.float32), tp, self.norm)
        return ob

    def step(self, acts):
        acts = [[max(-1.0, min(1.0, a[0])), max(A_MIN, min(A_MAX, a[1]))] for a in acts]
        new = self.pool.step([a[0] for a in acts], [a[1] for a in acts])
        rews, dones = [], []
        for i, (o, a) in enumerate(zip(new, acts)):
            path = self.paths[i]
            j = nearest(path, o, self.hint[i])
            lat = -math.sin(path[j, 2]) * (o["x"] - path[j, 0]) + math.cos(path[j, 2]) * (o["y"] - path[j, 1])
            head = math.atan2(math.sin(o["yaw"] - path[j, 2]), math.cos(o["yaw"] - path[j, 2]))
            xi = torch.tensor([[o[k] for k in STATE]], dtype=torch.float32)
            r, _, _ = reward(xi, torch.tensor([a]), torch.tensor([self.a_prev[i]]), torch.tensor([lat]), torch.tensor([head]), torch.tensor([self.vref[i]]))
            failed = (not o["ok"]) or o["tilt"] > 0.35 or abs(lat) > LAT_FAIL
            self.obs[i], self.a_prev[i], self.hint[i] = o, a, j
            done = failed or j >= self.end_idx[i]
            rews.append(float(r))
            dones.append(done)
            if done:
                self.reset(i)
        return rews, dones


def main(argv=None) -> int:
    args = parse(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    def log(m):
        print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    d = np.load(args.lift_data, allow_pickle=True)
    X = np.concatenate([ep["x"] for ep in d["train"]])
    norm = types.SimpleNamespace(mean=torch.as_tensor(X.mean(0), dtype=torch.float32), std=torch.as_tensor(X.std(0) + 1e-6, dtype=torch.float32))
    train_rng, eval_rng = np.random.default_rng(args.seed + 1), np.random.default_rng(args.seed + 2000)
    eval_cases = []
    for _ in range(args.eval_paths):
        p, _ = lane_change_path(eval_rng)
        eval_cases += [(p, v) for v in EVAL_SPEEDS]
    ev_args = types.SimpleNamespace(workers=args.workers)
    report = {"args": {k: v for k, v in vars(args).items() if k != "python"}, "curve": []}
    with ChronoPool(args.workers, args.python) as pool:
        # the same start as SuperMimic: limiter data on training paths, behaviour clone
        task_logs = []
        for _ in range(2):
            cases = [(lane_change_path(train_rng)[0], float(train_rng.uniform(*TRAIN_SPEEDS))) for _ in range(8)]
            for i in range(0, len(cases), args.workers):
                task_logs += drive(pool, cases[i:i + args.workers], "ssf_limiter", noise=0.05, rng=rng)
        bc_steps = sum(len(L["a"]) for L in task_logs)
        _st, paths = sample_starts(task_logs, 1, rng)
        obs_dim = features(torch.zeros(1, len(STATE)), torch.zeros(1, 3), torch.zeros(1), torch.zeros(1, dtype=torch.long), torch.ones(1),
                           torch.zeros(1, 2), torch.zeros(1, 10, 4), norm)[0].shape[-1]
        policy = Policy(obs_dim)
        popt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        eligible = [(e, t) for e, L in enumerate(task_logs) for t in range(1, len(L["a"]))]
        for _ in range(600):
            sel = [eligible[i] for i in rng.integers(0, len(eligible), 512)]
            x = torch.tensor(np.array([task_logs[e]["x"][t] for e, t in sel]), dtype=torch.float32)
            pose = torch.tensor(np.array([task_logs[e]["pose"][t] for e, t in sel]), dtype=torch.float32)
            s = torch.tensor([task_logs[e]["s"][t] for e, t in sel], dtype=torch.float32)
            a_prev = torch.tensor(np.array([task_logs[e]["a"][t - 1] for e, t in sel]), dtype=torch.float32)
            tgt = torch.tensor(np.array([task_logs[e]["a"][t] for e, t in sel]), dtype=torch.float32)
            ob, _, _ = features(x, pose, s, torch.tensor([e for e, _ in sel]), torch.tensor([task_logs[e]["v_ref"] for e, _ in sel], dtype=torch.float32), a_prev, paths, norm)
            loss = (((policy(ob) - tgt) / torch.tensor([1.0, 1.5])) ** 2).mean()
            popt.zero_grad()
            loss.backward()
            popt.step()
        ev, _ = evaluate(pool, eval_cases, "supermimic", ev_args, policy=policy, wm=norm)
        log(f"behaviour clone: rolled {ev['rolled']}, off path {ev['off_path']}, return {ev['mean_return']:.1f}, |lat| {ev['mean_abs_lat']:.3f}, speed {ev['mean_speed']:.1f}")
        report["curve"].append({"ppo_steps": 0, "bc_steps": bc_steps, **{k: v for k, v in ev.items() if k != "per_episode"}})

        log_std = nn.Parameter(torch.log(torch.tensor([0.1, 0.3])))
        value = nn.Sequential(nn.Linear(obs_dim, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 1))
        opt = torch.optim.Adam([*policy.parameters(), log_std, *value.parameters()], lr=3e-4)
        env = Env(pool, args.workers, norm, train_rng)
        ob = env.features()
        steps, next_eval = 0, args.eval_every
        while steps < args.steps:
            buf = {"ob": [], "a": [], "lp": [], "r": [], "d": [], "v": []}
            for _ in range(args.rollout):
                with torch.no_grad():
                    mu = policy(ob)
                    dist = torch.distributions.Normal(mu, log_std.exp())
                    a = dist.sample()
                    buf["lp"].append(dist.log_prob(a).sum(-1))
                    buf["v"].append(value(ob).squeeze(-1))
                r, dn = env.step(a.tolist())
                buf["ob"].append(ob)
                buf["a"].append(a)
                buf["r"].append(torch.tensor(r))
                buf["d"].append(torch.tensor(dn, dtype=torch.float32))
                ob = env.features()
                steps += args.workers
            with torch.no_grad():
                last_v = value(ob).squeeze(-1)
            T = len(buf["r"])
            adv, gae = torch.zeros(T, args.workers), torch.zeros(args.workers)
            for t in reversed(range(T)):
                nv = last_v if t == T - 1 else buf["v"][t + 1]
                delta = buf["r"][t] + 0.99 * nv * (1 - buf["d"][t]) - buf["v"][t]
                gae = delta + 0.99 * 0.95 * (1 - buf["d"][t]) * gae
                adv[t] = gae
            ret = adv + torch.stack(buf["v"])
            O, A, LP = torch.cat(buf["ob"]), torch.cat(buf["a"]), torch.cat(buf["lp"])
            ADV, RET = adv.reshape(-1), ret.reshape(-1)
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
            for _ in range(5):
                perm = torch.randperm(len(O))
                for mb in perm.split(512):
                    dist = torch.distributions.Normal(policy(O[mb]), log_std.exp())
                    ratio = (dist.log_prob(A[mb]).sum(-1) - LP[mb]).exp()
                    pl = -torch.min(ratio * ADV[mb], ratio.clamp(0.8, 1.2) * ADV[mb]).mean()
                    vl = ((value(O[mb]).squeeze(-1) - RET[mb]) ** 2).mean()
                    loss = pl + 0.5 * vl
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_([*policy.parameters(), log_std, *value.parameters()], 1.0)
                    opt.step()
            if steps >= next_eval or steps >= args.steps:
                next_eval += args.eval_every
                ev, _ = evaluate(pool, eval_cases, "supermimic", ev_args, policy=policy, wm=norm)
                env = Env(pool, args.workers, norm, train_rng)  # evaluation reset the workers
                ob = env.features()
                row = {"ppo_steps": steps, "bc_steps": bc_steps, **{k: v for k, v in ev.items() if k != "per_episode"}}
                report["curve"].append(row)
                report["final"] = ev
                log(f"PPO {steps} steps: rolled {ev['rolled']}/{ev['episodes']}, off path {ev['off_path']}, return {ev['mean_return']:.1f}, "
                    f"|lat| {ev['mean_abs_lat']:.3f} m, lift {ev['two_wheel_lift_s']:.1f} s, speed {ev['mean_speed']:.1f} m/s")
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(report, indent=1))
    log(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
