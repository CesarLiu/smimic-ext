"""Are the world model's gradients with respect to the context xi (friction, tyre stiffness, mass, CoG position)
right? (VEHICLE_PROPOSAL.md section 11.7: the gradient check for chassis co-design and for identification /
stress testing.)

SuperMimic only ever used the world model's gradients with respect to actions (section 3.9). Co-design needs the
gradient of the closed-loop return with respect to design parameters; identification and worst-case search need the
gradient of predicted states or returns with respect to the friction and the vehicle parameters. Here both are
compared with the simulator's, on a trained run's world model and policy (``compare_ppo.py`` writes
``supermimic_wm.pt`` next to ``policies.pt``):

* closed loop: J = sum of rewards over H steps of the policy driving from logged start states, with every reward,
  observation, goal and policy call identical on both sides (``imagine``); the world model's dJ/dxi by autograd,
  the simulator's by central differences in float64 through the same rollout code with the simulator in place of
  the world model (``SimAsWM``). Starts where two difference steps disagree are excluded (non-smooth);
* open loop: d s_k / d xi along the simulator rollout's actions (identification from logged commands);
* descent check: a step of xi along the world model's gradient, then the simulator's change of J (does following
  the model's gradient improve the true return?), against random directions.

Per xi component (in units of its training range): sign agreement, correlation over the starts and median
magnitude ratio; per start, the cosine of the 4-component gradient. Grouped by the start's friction use.

    PYTHONPATH=src python scripts/vehicle/xi_gradient_check.py --run logs/compare/seed1 --out logs/xi_gradient_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_ppo
from demo_e2e import ObsHistory
from limit_study import env_cfg, libraries

from supermimic.imagination.rollout import imagine
from supermimic.vehicles import jacobian as J
from supermimic.vehicles.dynamics import VehicleDynamics
from supermimic.vehicles.env import ReferenceLibrary, VehicleTrackingEnv
from supermimic.vehicles.learner import VehicleStateSpecCfg
from supermimic.vehicles.threads import add_threads_arg, limit_threads
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

XI = ("mu", "tire_stiffness", "mass", "cog_shift")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="compare_ppo.py output directory with policies.pt and supermimic_wm.pt")
    ap.add_argument("--starts", type=int, default=128)
    ap.add_argument("--warm-steps", type=int, default=300, help="policy steps on held-out references before sampling starts")
    ap.add_argument("--horizons", type=int, nargs="+", default=[8, 50])
    ap.add_argument("--open-loop-k", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--h", type=float, default=0.04, help="difference step, as a share of each xi component's training range")
    ap.add_argument("--descent-step", type=float, default=0.1, help="xi step of the descent check (share of the ranges)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/xi_gradient_check.json")
    add_threads_arg(ap)
    return ap.parse_args(argv)


class SimAsWM:
    """The simulator behind the world model's interface, so ``imagine`` runs the true dynamics with the same
    rewards, observations and policy calls."""

    def __init__(self, wm: WorldModel, dyn: VehicleDynamics, cfg):
        self.heads, self.ensemble, self._wm, self.dyn, self.cfg = wm.heads, 1, wm, dyn, cfg

    def step(self, s, a, xi):
        return J.logged_step(self.dyn, self.cfg, s, a, xi)

    def termination_logits(self, s, ref):
        return self._wm.termination_logits(s, ref)


def load(args):
    blob = torch.load(Path(args.run) / "policies.pt", map_location="cpu")
    run = types.SimpleNamespace(**{**vars(compare_ppo.parse([])), **blob["args"], "device": "cpu"})
    wmblob = torch.load(Path(args.run) / "supermimic_wm.pt", map_location="cpu")
    cfg = env_cfg(run, args.starts, args.warm_steps + 60)
    lib = ReferenceLibrary.concat(libraries(run, cfg, run.eval_refs_per_level, run.seed + 1000))  # held-out references
    env = VehicleTrackingEnv(cfg, lib, seed=args.seed)
    heads, task, _spec, _io, _pcfg, policy, _critic = compare_ppo.networks(run, env, lib)
    policy.load_state_dict(blob["policies"]["supermimic"])
    policy.eval()
    wm = WorldModel(WorldModelCfg(hidden=tuple(wmblob["hidden"]), state_spec=VehicleStateSpecCfg.from_env(cfg, env.dyn.p),
                                  ensemble=wmblob["ensemble"]), heads, "cpu")
    wm.load_state_dict(wmblob["wm"])
    wm.eval()
    wm.requires_grad_(False)
    policy.requires_grad_(False)
    return run, cfg, lib, env, heads, task, policy, wm


@torch.no_grad()
def collect_starts(env, heads, policy, run, n, steps):
    obs = env.reset()
    hist = ObsHistory(heads, env.num_envs, run.history, "cpu")
    hist.reset(torch.arange(env.num_envs), obs["policy"])
    for _ in range(steps):
        obs, _r, _t, _tr, info = env.step(policy(hist.flat(), obs["goal"]))
        hist.push(obs["policy"])
        if info["done"].any():
            hist.reset(info["done"].nonzero().squeeze(-1), obs["policy"])
    return env.buffer.sample_starts(n, run.history)


def to64(start):
    return replace(start, **{k: getattr(start, k).double() for k in ("state", "xi", "prev_action", "hist_states", "hist_actions")})


def closed_loop_return(model, policy, task, start, H):
    ro = imagine(model, policy, task, start, H, obs_noise=False)
    return ro.rewards.sum(0), ro


def main(argv=None) -> int:
    args = parse(argv)
    limit_threads(args.threads)
    torch.manual_seed(args.seed)
    t0 = time.time()
    run, cfg, _lib, env, heads, task, policy, wm = load(args)
    start = collect_starts(env, heads, policy, run, args.starts, args.warm_steps)
    torch.set_default_dtype(torch.float64)
    wm.double()
    policy.double()
    start = to64(start)
    dyn = VehicleDynamics(env.dyn.p, cfg.dt, cfg.substeps, cfg.assist)
    sim = SimAsWM(wm, dyn, cfg)
    width = torch.tensor([cfg.mu_range[1] - cfg.mu_range[0], cfg.tire_stiffness_range[1] - cfg.tire_stiffness_range[0],
                          cfg.mass_range[1] - cfg.mass_range[0], cfg.cog_shift_range[1] - cfg.cog_shift_range[0]])
    u = J.utilisation(start.state, start.xi)
    groups = {"u<0.6": u < 0.6, "0.6-0.9": (u >= 0.6) & (u < 0.9), "u>=0.9": u >= 0.9}
    print(f"[{time.time() - t0:5.0f}s] {args.starts} starts; friction use: " + ", ".join(f"{k} {int(v.sum())}" for k, v in groups.items()), flush=True)
    report = {"run": str(args.run), "args": vars(args), "xi": list(XI), "xi_range_width": width.tolist(), "groups": {k: int(v.sum()) for k, v in groups.items()},
              "closed_loop": {}, "open_loop": {}, "descent": {}}

    def stats(gw, gs, ok):
        """gw, gs (n, 4) gradients in units of the xi ranges; ok (n,) smooth starts."""
        out = {}
        for name, m in {"all": ok, **{k: v & ok for k, v in groups.items()}}.items():
            if m.sum() < 3:
                continue
            a, b = gw[m], gs[m]
            row = {"n": int(m.sum()), "cos_median": float(J._cos(a, b).median())}
            for j, x in enumerate(XI):
                big = b[:, j].abs() > 1e-3 * b.abs().max()
                row[x] = {"sign_agree": float((torch.sign(a[big, j]) == torch.sign(b[big, j])).double().mean()) if big.any() else None,
                          "corr": float(torch.corrcoef(torch.stack([a[:, j], b[:, j]]))[0, 1]),
                          "mag_ratio_median": float((a[big, j].abs() / b[big, j].abs()).median()) if big.any() else None}
            out[name] = row
        return out

    for H in args.horizons:
        xi = start.xi.clone().requires_grad_(True)
        Jw, _ = closed_loop_return(wm, policy, task, replace(start, xi=xi), H)
        (gw,) = torch.autograd.grad(Jw.sum(), xi)
        gw = gw[:, :4] * width
        gs, gs2 = torch.zeros_like(gw), torch.zeros_like(gw)
        with torch.no_grad():
            for j in range(4):
                for h, g in ((args.h, gs), (args.h / 2, gs2)):
                    d = torch.zeros_like(start.xi)
                    d[:, j] = h * width[j]
                    Jp, _ = closed_loop_return(sim, policy, task, replace(start, xi=start.xi + d), H)
                    Jm, _ = closed_loop_return(sim, policy, task, replace(start, xi=start.xi - d), H)
                    g[:, j] = (Jp - Jm) / (2 * h)
        ok = J._cos(gs, gs2) > 0.95
        report["closed_loop"][str(H)] = {"smooth": int(ok.sum()), **stats(gw, gs, ok)}
        r = report["closed_loop"][str(H)].get("all", {})
        print(f"[{time.time() - t0:5.0f}s] closed loop H={H}: smooth {int(ok.sum())}/{args.starts}, cos median {r.get('cos_median', float('nan')):.3f}, "
              + ", ".join(f"{x} sign {r[x]['sign_agree']:.2f} corr {r[x]['corr']:.2f} ratio {r[x]['mag_ratio_median']:.2f}" for x in XI if x in r), flush=True)

        # descent check: step xi along the model's gradient (and along random directions), true change of J
        with torch.no_grad():
            J0, _ = closed_loop_return(sim, policy, task, start, H)
            res = {}
            for name, dirs in (("model_gradient", gw), ("simulator_gradient", gs), ("random", torch.randn_like(gw))):
                step = args.descent_step * dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12) * width
                xi_new = start.xi.clone()
                xi_new[:, :4] = (xi_new[:, :4] + step)
                J1, _ = closed_loop_return(sim, policy, task, replace(start, xi=xi_new), H)
                res[name] = {"improved_share": float((J1 > J0).double().mean()), "mean_gain": float((J1 - J0).mean())}
            report["descent"][str(H)] = res
            print(f"[{time.time() - t0:5.0f}s]   descent: " + ", ".join(f"{k} improves {v['improved_share']:.2f} (mean dJ {v['mean_gain']:+.4f})" for k, v in res.items()), flush=True)

    # open loop: d s_k / d xi along the simulator's closed-loop actions
    with torch.no_grad():
        _, ro = closed_loop_return(sim, policy, task, start, max(args.open_loop_k))
    acts = ro.actions
    err = wm.spec.err_scale.double()
    rows = J.LEARNED
    for k in args.open_loop_k:
        xi = start.xi.clone().requires_grad_(True)
        x = start.state
        for t in range(k):
            x = wm.step(x, acts[t], xi)
        out = x / err
        Jw_ = torch.zeros(args.starts, len(rows), 4)
        for i, row in enumerate(rows):
            (g,) = torch.autograd.grad(out[:, row].sum(), xi, retain_graph=True)
            Jw_[:, i] = g[:, :4] * width
        Js_, Js2_ = torch.zeros_like(Jw_), torch.zeros_like(Jw_)
        with torch.no_grad():
            for j in range(4):
                for h, Jx in ((args.h, Js_), (args.h / 2, Js2_)):
                    d = torch.zeros_like(start.xi)
                    d[:, j] = h * width[j]
                    xp, xm = start.state, start.state
                    for t in range(k):
                        xp, xm = sim.step(xp, acts[t], start.xi + d), sim.step(xm, acts[t], start.xi - d)
                    Jx[:, :, j] = ((xp - xm) / err)[:, rows] / (2 * h)
        ok = J._cos(Js_.flatten(1), Js2_.flatten(1)) > 0.95
        per = {}
        for j, x in enumerate(XI):
            c = J._cos(Jw_[:, :, j], Js_[:, :, j])
            per[x] = {"cos_median": float(c[ok].median()), "mag_ratio_median": float((Jw_[ok, :, j].norm(dim=-1) / Js_[ok, :, j].norm(dim=-1).clamp_min(1e-12)).median())}
            for g, m in groups.items():
                if (m & ok).sum() >= 3:
                    per[x][f"cos_median_{g}"] = float(c[m & ok].median())
        report["open_loop"][str(k)] = {"smooth": int(ok.sum()), **per}
        print(f"[{time.time() - t0:5.0f}s] open loop k={k}: smooth {int(ok.sum())}, " + ", ".join(f"{x} cos {v['cos_median']:.3f} ratio {v['mag_ratio_median']:.2f}" for x, v in per.items()), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"report: {args.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
