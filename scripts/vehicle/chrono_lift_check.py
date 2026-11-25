"""Can a learned world model capture the dynamics around two-wheel lift, and are its gradients right there?
(VEHICLE_PROPOSAL.md section 11.12.)

Wheel lift is a contact switch, the kind of non-smooth dynamics a single-track model cannot describe and SuperMimic
was built for (feet on the ground). Data: Chrono's HMMWV with an 800 kg roof load (static stability factor about 0.8,
below the tyres' friction, so it lifts two wheels and rolls), rigid ground, TMeasy tyres, the chassis colliding with
the ground, driven by random fishhook-like steering (fast reversals between random amplitudes) or sinusoids at
8-17 m/s; an episode ends at 6 s or once the roll exceeds 1.2 rad.

World model: an MLP that maps the last two states and actions to the change of state, trained with an 8-step
autoregressive loss in units of each component's standard deviation (as SuperMimic's world models). State: body
velocities (vx, vy, vz), body rates (yaw r, roll p, pitch q), roll and pitch angles, front-wheel angle, steering-servo
state, speed-loop target and integrator, and the four tyre normal loads (which carry the suspension's compression).
Actions: steering command and acceleration command. Two states of history stand in for what the state leaves out
(suspension deflections, tyre and powertrain states).

Checks, with every sample labelled by its contact regime at the start (all wheels down / some wheel off / both wheels of
one side off and tilt < 20 deg / rolling, tilt >= 20 deg) and separately the windows in which a two-wheel lift starts
or ends:
1. 8-step prediction error on held-out episodes, against a "no change" predictor in the same units;
2. gradient of the state after k steps (roll angle, roll rate, lateral velocity, yaw rate) with respect to the steering
   command held over those k steps: the world model's by autograd, Chrono's by central differences, taken by replaying
   the episode (rollouts are deterministic) with the commands shifted by +-h (and +-h/2 to flag non-smooth points).

    SUPERMIMIC_CHRONO_PYTHON=chrono_env/bin/python PYTHONPATH=src python scripts/vehicle/chrono_lift_check.py \\
        --out docs/results/vehicle_chrono_lift_check.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from supermimic.vehicles.chrono_env import ChronoPool

STATE = ["vx", "vy", "vz", "r", "p", "q", "roll_b", "pitch_b", "delta", "s_act", "v_target", "integ", "fz0", "fz1", "fz2", "fz3"]
OUT = ["roll_b", "p", "vy", "r"]  # gradient-check outputs
CFG = {"terrain": "rigid", "roof_kg": 800.0, "powertrain": "map", "chassis_collision": True, "max_roll": 1.2, "settle": 2.0}
REGIMES = ["grounded", "wheel_off", "two_wheel_lift", "rolling"]
LIFT_N = 100.0  # [N] a tyre carrying less is off the ground


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--heldout", type=int, default=80)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--updates", type=int, default=6000)
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--history", type=int, default=2, help="states (and actions) of history the world model sees")
    ap.add_argument("--grad-k", type=int, nargs="+", default=[5, 15])
    ap.add_argument("--grad-per-regime", type=int, default=40)
    ap.add_argument("--h", type=float, default=0.05, help="steering perturbation (normalised command units)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="logs/chrono_lift_data.npz")
    ap.add_argument("--python", default=None)
    ap.add_argument("--out", default="logs/chrono_lift_check.json")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------- data


def episode_plan(rng, steps):
    v0 = float(rng.uniform(8.0, 17.0))
    if rng.random() < 0.6:  # fishhook-like: holds at random amplitudes, fast reversals (servo rate limited)
        acts = []
        while len(acts) < steps:
            s = float(rng.choice([-1, 1]) * rng.uniform(0.2, 0.95))
            a = float(rng.uniform(-1.0, 0.5))
            acts += [[s, a]] * int(rng.uniform(0.3, 1.2) / 0.02)
    else:  # sinusoidal steering
        amp, f, ph = rng.uniform(0.2, 0.95), rng.uniform(0.3, 1.0), rng.uniform(0, 2 * math.pi)
        acts = [[float(amp * math.sin(2 * math.pi * f * 0.02 * i + ph)), float(rng.uniform(-0.5, 0.3))] for i in range(steps)]
    return {**CFG, "v": v0}, acts[:steps]


def collect(args, n, seed, log):
    rng = np.random.default_rng(seed)
    plans = [episode_plan(rng, args.steps) for _ in range(n)]
    eps = []
    with ChronoPool(args.workers, args.python) as pool:
        for i in range(0, n, args.workers):
            batch = plans[i:i + args.workers]
            for (cfg, acts), res in zip(batch, pool.rollout(batch, ids=range(len(batch)))):
                obs = [res["obs0"], *res["obs"]]
                eps.append({"cfg": cfg, "actions": np.array(acts[: len(obs) - 1], dtype=np.float64),
                            "x": np.array([[o[k] for k in STATE] for o in obs], dtype=np.float64),
                            "tilt": np.array([o["tilt"] for o in obs])})
            if (i // args.workers) % 20 == 0:
                log(f"  {len(eps)}/{n} episodes")
    return eps


def regime(x_row, tilt):
    fz = x_row[12:16]
    off = fz < LIFT_N
    two = (off[0] and off[2]) or (off[1] and off[3])
    if tilt >= math.radians(20):
        return 3
    if two:
        return 2
    return 1 if off.any() else 0


def two_lift(x_row):
    off = x_row[12:16] < LIFT_N
    return bool((off[0] and off[2]) or (off[1] and off[3]))


# ---------------------------------------------------------------------- model


class LiftWM(nn.Module):
    """Next state from the last ``hist`` states and actions (newest first in the input; ``hist`` = 2 is the original
    two-state model). Longer histories let it infer states it is not given, such as a sloshing payload's swing."""

    def __init__(self, n_x, n_a, hidden, mean, std, dstd, hist: int = 2):
        super().__init__()
        self.hist = hist
        layers, d = [], hist * (n_x + n_a)
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ELU()]
            d = h
        layers.append(nn.Linear(d, n_x))
        self.net = nn.Sequential(*layers)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.register_buffer("dstd", torch.as_tensor(dstd, dtype=torch.float32))

    def step_hist(self, Xh, Ah):
        """Xh (hist, n, X) states and Ah (hist, n, 2) actions, oldest first (Ah[-1] is the action applied now)."""
        z = torch.cat([*((Xh[i] - self.mean) / self.std for i in reversed(range(self.hist))), *(Ah[i] for i in reversed(range(self.hist)))], -1)
        return Xh[-1] + self.dstd * self.net(z)

    def step(self, x, x_prev, a, a_prev):
        assert self.hist == 2
        return self.step_hist(torch.stack([x_prev, x]), torch.stack([a_prev, a]))

    def rollout_hist(self, Xh, Ah_past, actions):
        """Xh (hist, n, X) states up to now, Ah_past (hist - 1, n, 2) the actions before now, actions (k, n, 2) ->
        states (k, n, X) after each step."""
        out = []
        for t in range(actions.shape[0]):
            x1 = self.step_hist(Xh, torch.cat([Ah_past, actions[t][None]]))
            out.append(x1)
            Xh = torch.cat([Xh[1:], x1[None]])
            Ah_past = torch.cat([Ah_past[1:], actions[t][None]]) if self.hist > 1 else Ah_past
        return torch.stack(out)

    def rollout(self, x0, x_prev, actions, a_prev):
        """hist = 2: actions (k, n, 2) -> states (k, n, X) after each step."""
        return self.rollout_hist(torch.stack([x_prev, x0]), a_prev[None], actions)


def windows(eps, k, hist: int = 2):
    """(episode, t) with t >= hist - 1 and t + k inside the episode."""
    return [(e, t) for e, ep in enumerate(eps) for t in range(hist - 1, len(ep["x"]) - k)]


def batch(eps, idx, k, hist: int = 2):
    """X (hist + k, n, X): x_{t-hist+1} .. x_{t+k}; A (hist - 1 + k, n, 2): a_{t-hist+1} .. a_{t+k-1}."""
    X = np.stack([eps[e]["x"][t - hist + 1:t + k + 1] for e, t in idx], 1)
    A = np.stack([eps[e]["actions"][t - hist + 1:t + k] for e, t in idx], 1)
    return torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(A, dtype=torch.float32)


def predict(wm, Xb, Ab):
    """The world model's k-step prediction for a batch from ``batch``: states x_{t+1} .. x_{t+k}."""
    H = wm.hist
    return wm.rollout_hist(Xb[:H], Ab[:H - 1], Ab[H - 1:])


def train(args, eps, log):
    X = np.concatenate([ep["x"] for ep in eps])
    D = np.concatenate([np.diff(ep["x"], axis=0) for ep in eps])
    mean, std, dstd = X.mean(0), X.std(0) + 1e-6, D.std(0) + 1e-6
    wm = LiftWM(len(STATE), 2, args.hidden, mean, std, dstd, hist=args.history)
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    idx = windows(eps, args.k, wm.hist)
    rng = np.random.default_rng(args.seed)
    stdt = torch.as_tensor(std, dtype=torch.float32)
    t0 = time.time()
    for u in range(args.updates):
        for g in opt.param_groups:
            g["lr"] = 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + math.cos(math.pi * u / args.updates))
        sel = [idx[i] for i in rng.integers(0, len(idx), 512)]
        Xb, Ab = batch(eps, sel, args.k, wm.hist)
        pred = predict(wm, Xb, Ab)
        loss = (((pred - Xb[wm.hist:]) / stdt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt.step()
        if (u + 1) % max(1, args.updates // 6) == 0:
            log(f"  update {u + 1}: {args.k}-step loss {float(loss):.4f} ({time.time() - t0:.0f} s)")
    return wm


# ---------------------------------------------------------------------- checks


@torch.no_grad()
def prediction_errors(wm, eps, k):
    idx = windows(eps, k, wm.hist)
    stdt = wm.std
    rows = {name: [] for name in [*REGIMES, "lift_transition"]}
    base = {name: [] for name in rows}
    per_var = {name: [] for name in rows}
    for i in range(0, len(idx), 4096):
        sel = idx[i:i + 4096]
        Xb, Ab = batch(eps, sel, k, wm.hist)
        pred = predict(wm, Xb, Ab)
        err = (((pred[-1] - Xb[-1]) / stdt) ** 2).mean(-1).numpy()
        zero = (((Xb[wm.hist - 1] - Xb[-1]) / stdt) ** 2).mean(-1).numpy()
        var = (((pred[-1] - Xb[-1]) / stdt) ** 2).numpy()
        for j, (e, t) in enumerate(sel):
            ep = eps[e]
            reg = REGIMES[regime(ep["x"][t], ep["tilt"][t])]
            for name in (reg, "lift_transition") if len({two_lift(r) for r in ep["x"][t:t + k + 1]}) > 1 else (reg,):
                rows[name].append(err[j])
                base[name].append(zero[j])
                per_var[name].append(var[j])
    out = {}
    for name, vals in rows.items():
        if vals:
            v = np.array(per_var[name]).mean(0)
            out[name] = {"windows": len(vals), "nse": float(np.mean(vals)), "nse_median": float(np.median(vals)),
                         "no_change_nse": float(np.mean(base[name])), "roll_nse": float(v[STATE.index("roll_b")]),
                         "roll_rate_nse": float(v[STATE.index("p")]), "tyre_load_nse": float(v[12:16].mean())}
    return out


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def gradient_check(args, wm, eps, log):
    rng = np.random.default_rng(args.seed + 5)
    oi = [STATE.index(o) for o in OUT]
    stdt = wm.std.numpy()[oi]
    res = {}
    with ChronoPool(args.workers, args.python) as pool:
        for k in args.grad_k:
            cands = {r: [] for r in range(len(REGIMES))}
            for e, ep in enumerate(eps):
                for t in range(wm.hist - 1, len(ep["x"]) - k):
                    a = ep["actions"][t:t + k, 0]
                    if np.abs(a).max() < 1 - args.h:  # no clipping of the perturbed commands
                        cands[regime(ep["x"][t], ep["tilt"][t])].append((e, t))
            picks = []
            for r, c in cands.items():
                for j in rng.permutation(len(c))[: args.grad_per_regime]:
                    picks.append((r, *c[j]))
            # Chrono: replay each episode up to t + k with the commands of steps t .. t+k-1 shifted
            jobs = []
            for _r, e, t in picks:
                ep = eps[e]
                for dh in (args.h, -args.h, args.h / 2, -args.h / 2):
                    acts = ep["actions"][: t + k].copy()
                    acts[t:t + k, 0] += dh
                    jobs.append((ep["cfg"], acts.tolist()))
            outs = []
            for i in range(0, len(jobs), args.workers):
                part = jobs[i:i + args.workers]
                outs += pool.rollout(part, ids=range(len(part)))

            def final(o, t, k=k):
                rows = o["obs"]
                return np.array([rows[t + k - 1][n] for n in OUT]) if len(rows) >= t + k else None

            rows = []
            for j, (r, e, t) in enumerate(picks):
                f = [final(outs[4 * j + m], t) for m in range(4)]
                if any(v is None for v in f):
                    continue
                g1 = (f[0] - f[1]) / (2 * args.h) / stdt
                g2 = (f[2] - f[3]) / args.h / stdt
                ep = eps[e]
                X = torch.as_tensor(ep["x"], dtype=torch.float32)
                A = torch.as_tensor(ep["actions"], dtype=torch.float32)
                shift = torch.zeros(1, requires_grad=True)
                acts = A[t:t + k].clone()
                acts[:, 0] = acts[:, 0] + shift
                H = wm.hist
                pred = wm.rollout_hist(X[t - H + 1:t + 1][:, None], A[t - H + 1:t][:, None], acts[:, None])
                y = (pred[-1, 0, oi] / wm.std[oi])
                gw = np.array([float(torch.autograd.grad(y[i], shift, retain_graph=True)[0]) for i in range(len(OUT))])
                rows.append({"regime": r, "smooth": cos(g1, g2) > 0.9, "tiny": float(np.linalg.norm(g1)) < 1e-2, "cos": cos(gw, g1),
                             "roll_sign": bool(np.sign(gw[0]) == np.sign(g1[0])), "ratio": float(np.linalg.norm(gw) / (np.linalg.norm(g1) + 1e-12))})
            summ = {}
            for r, name in enumerate(REGIMES):
                rr = [q for q in rows if q["regime"] == r]
                ok = [q for q in rr if q["smooth"] and not q["tiny"]]
                if not rr:
                    continue
                summ[name] = {"samples": len(rr), "non_smooth": sum(not q["smooth"] for q in rr), "tiny": sum(q["tiny"] and q["smooth"] for q in rr),
                              "compared": len(ok)}
                if ok:
                    summ[name].update({"cos_median": float(np.median([q["cos"] for q in ok])), "cos_above_0.8": float(np.mean([q["cos"] > 0.8 for q in ok])),
                                       "roll_sign_agree": float(np.mean([q["roll_sign"] for q in ok])), "mag_ratio_median": float(np.median([q["ratio"] for q in ok]))})
            res[str(k)] = summ
            log(f"  gradient k={k}: " + "; ".join(f"{n} cos {s.get('cos_median', float('nan')):.2f} (n {s['compared']}, non-smooth {s['non_smooth']})" for n, s in summ.items()))
    return res


def main(argv=None) -> int:
    args = parse(argv)
    torch.manual_seed(args.seed)
    t0 = time.time()
    def log(m):
        print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    cache = Path(args.cache)
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        train_eps, test_eps = list(d["train"]), list(d["test"])
        log(f"data from {cache}")
    else:
        log("collecting training episodes")
        train_eps = collect(args, args.episodes, args.seed, log)
        log("collecting held-out episodes")
        test_eps = collect(args, args.heldout, args.seed + 1000, log)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, train=np.array(train_eps, dtype=object), test=np.array(test_eps, dtype=object))
    counts = np.bincount([regime(x, tl) for ep in train_eps for x, tl in zip(ep["x"], ep["tilt"])], minlength=4)
    log(f"{len(train_eps)} training episodes, {sum(len(ep['x']) for ep in train_eps)} steps; regimes " +
        ", ".join(f"{n} {c}" for n, c in zip(REGIMES, counts)))
    wm = train(args, train_eps, log)
    wm.eval()
    err = prediction_errors(wm, test_eps, args.k)
    log(f"{args.k}-step error on held-out episodes (state-std units; 'no change' predictor for scale):")
    for n, e in err.items():
        log(f"  {n:<16} windows {e['windows']:6d}  error {e['nse']:.4f} (median {e['nse_median']:.4f})  no-change {e['no_change_nse']:.3f}  "
            f"roll {e['roll_nse']:.4f}  roll rate {e['roll_rate_nse']:.4f}  tyre loads {e['tyre_load_nse']:.4f}")
    grads = gradient_check(args, wm, test_eps, log)
    report = {"args": {k: v for k, v in vars(args).items() if k != "python"}, "state": STATE, "outputs": OUT, "regime_steps_train": dict(zip(REGIMES, counts.tolist())),
              "prediction_error": err, "gradient": grads}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    log(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
