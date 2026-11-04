"""Step 2 after E1: can a policy trained through the world model track references at the grip limit, and
does a world model with accurate multi-step gradients (the large one of E1) matter for that?

    PYTHONPATH=src python scripts/vehicle/limit_study.py [--models small large] [--iters 2500]

* References: for each road friction level (default 0.5 / 0.8 / 1.0) trajectories generated under that
  friction, lateral-acceleration peaks 40-95 % of mu g, so they are feasible for it (VEHICLE_PROPOSAL.md
  §4, option 1); about an eighth of their steps use more than 80 % of the available friction. Training
  episodes run at their reference's friction (``match_friction``); tyre stiffness, mass, CoG and delay
  stay randomised. ABS / ESC off, as in E1, so the policy alone has to stay within the grip.
* For each world model size (small 256x256, large 512x512), the same SuperMimic loop as
  ``demo_e2e.py`` trains a policy for the same number of iterations.
* Evaluation on held-out references of every friction level (nominal context otherwise, no noise, whole
  reference from its start): normalised return, termination rate, lateral error; and on the reference
  steps that use more than 80 % of the friction ("near the limit"): the share of them the policy is still
  on the reference for, and its lateral / heading / sideslip errors there.

Writes a JSON report; there is no pass / fail gate (this is the measurement that decides between E1
mitigations and moving on to E3 / E4).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_e2e import ObsHistory

from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.policy.nets import Critic, CriticCfg, NetIO, Policy, PolicyCfg
from supermimic.vehicles import tracking
from supermimic.vehicles.dynamics import AssistCfg
from supermimic.vehicles.env import ReferenceLibrary, VehicleEnvCfg, VehicleTrackingEnv
from supermimic.vehicles.learner import VehicleHeads, VehicleStateSpecCfg, VehicleTask
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

MODELS = {"small": (256, 256), "large": (512, 512)}
NEAR_LIMIT = 0.8


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=["small", "large"], choices=list(MODELS))
    ap.add_argument("--mu-levels", type=float, nargs="+", default=[0.5, 0.8, 1.0])
    ap.add_argument("--ay-frac", type=float, nargs=2, default=[0.4, 0.95])
    ap.add_argument("--refs-per-level", type=int, default=12)
    ap.add_argument("--eval-refs-per-level", type=int, default=6)
    ap.add_argument("--ref-steps", type=int, default=500)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--policy-warmup", type=int, default=300)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--episode-steps", type=int, default=250)
    ap.add_argument("--history", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--action-noise", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--out", default="docs/results/vehicle_limit_study.json")
    return ap.parse_args(argv)


def env_cfg(args, n, capacity, train=True) -> VehicleEnvCfg:
    return VehicleEnvCfg(num_envs=n, episode_steps=args.episode_steps if train else 10_000, replay_capacity=capacity,
                         domain_randomization=train, reset_noise=train, match_friction=True, assist=AssistCfg(abs=False, esc=False))


def libraries(args, cfg, n_per_level, seed):
    return [ReferenceLibrary.generate(cfg, num=n_per_level, steps=args.ref_steps, seed=seed + 17 * k, mu=mu, ay_frac=tuple(args.ay_frac), oversample=4.0)
            for k, mu in enumerate(args.mu_levels)]


@torch.no_grad()
def evaluate(policy, heads, lib: ReferenceLibrary, args) -> dict:
    """Deterministic play of every reference from its start at its own friction, nominal context otherwise."""
    n, length = lib.layout.num_sequences, int(lib.lengths[0])
    cfg = env_cfg(args, n, 2, train=False)
    env = VehicleTrackingEnv(cfg, lib, record=False)
    env.forced_xi = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])  # friction then set from the reference
    env.forced_frame[:] = lib.layout.seq_start
    obs = env.reset()
    hist = ObsHistory(heads, n, args.history, "cpu")
    hist.reset(torch.arange(n), obs["policy"])
    alive = torch.ones(n, dtype=torch.bool)
    ret, steps, lat = torch.zeros(n), torch.zeros(n), torch.zeros(n)
    nl_total, nl_alive = torch.zeros(n), torch.zeros(n)
    nl_err = {"lat": torch.zeros(n), "yaw": torch.zeros(n), "beta": torch.zeros(n)}
    for _ in range(length - 1):
        frame = env.frame.clone()
        ref = lib.ref(frame)
        near = lib.util[frame] > NEAR_LIMIT
        a = policy(hist.flat(), obs["goal"])
        obs, r, _term, _trunc, info = env.step(a)
        e = tracking.errors(env.s, ref)  # rows that just ended are masked by ``alive`` below
        ret += r * alive
        lat += e["lat"].abs() * alive
        steps += alive.float()
        nl_total += near.float()
        live_near = near & alive & ~info["done"]
        nl_alive += live_near.float()
        for k in nl_err:
            nl_err[k] += e[k].abs() * live_near
        alive &= ~info["done"]
        hist.push(obs["policy"])
    out = {
        "normalized_return": float((ret / (heads.max_step_reward * (length - 1))).mean()),
        "termination_rate": float((steps < length - 1).float().mean()),
        "mean_abs_lateral_error_m": float((lat / steps.clamp_min(1)).mean()),
        "mean_steps": float(steps.mean()),
        "near_limit_steps": int(nl_total.sum()),
        "near_limit_survived_share": float(nl_alive.sum() / nl_total.sum().clamp_min(1)),
    }
    m = nl_alive.sum().clamp_min(1)
    out.update({f"near_limit_mean_abs_{k}": float(v.sum() / m) for k, v in nl_err.items()})
    return out


def evaluate_levels(policy, heads, eval_libs, args) -> dict:
    return {f"mu_{mu}": evaluate(policy, heads, lib, args) for mu, lib in zip(args.mu_levels, eval_libs)}


def train(model: str, args, lib, eval_libs, t0) -> dict:
    torch.manual_seed(args.seed)
    cfg = env_cfg(args, args.num_envs, args.episode_steps + 50)
    env = VehicleTrackingEnv(cfg, lib, seed=args.seed)
    heads = VehicleHeads(cfg, env.dyn.p.R_w)
    task = VehicleTask(lib, cfg.preview)
    wm = WorldModel(WorldModelCfg(hidden=MODELS[model], state_spec=VehicleStateSpecCfg.from_env(cfg, env.dyn.p)), heads, "cpu")
    io = NetIO.from_interfaces(wm.spec, heads, task, args.history)
    pcfg = PolicyCfg(encoder_hidden=(128,), decoder_hidden=(256, 128), latent_dim=32, history=args.history, decoder_last_gain=0.1)
    policy, critic = Policy(pcfg, io), Critic(CriticCfg(hidden=(256, 128)), io=io)
    wm_tr = WorldModelTrainer(wm, task, WMTrainerCfg(window=8, batches_per_update=2, batch_size=args.batch, lr=args.lr, compile=None))
    pol_tr = PolicyTrainer(wm, policy, critic, task, PolicyTrainerCfg(horizon=args.horizon, batch_size=args.batch, history=args.history, lr=args.lr,
                                                                       critic_lr=args.lr, critic_epochs=2, critic_minibatches=2, compile=None,
                                                                       value_clip=cfg.tracking.max_step_reward / (1 - 0.99)))
    before = evaluate_levels(policy, heads, eval_libs, args)
    n = args.num_envs
    obs = env.reset()
    hist = ObsHistory(heads, n, args.history, "cpu")
    hist.reset(torch.arange(n), obs["policy"])
    noise_half = heads.obs_noise_half()
    log, wm_m, pol_m = [], {}, {}
    for it in range(args.iters):
        with torch.no_grad():
            a = policy(hist.flat(), obs["goal"])
            obs, _r, _term, _trunc, info = env.step(a, args.action_noise * torch.randn_like(a))
        noisy = obs["policy"] + (torch.rand_like(obs["policy"]) * 2 - 1) * noise_half
        hist.push(noisy)
        if info["done"].any():
            hist.reset(info["done"].nonzero().squeeze(-1), noisy)
        if it >= args.warmup:
            wm_m = wm_tr.update(env.buffer)
        if it >= args.policy_warmup:
            pol_m = pol_tr.update(env.buffer)
        if (it + 1) % args.log_every == 0:
            row = {"iter": it + 1, "time_s": round(time.time() - t0, 1), "wm_nse_8step": round(float(wm_m.get("nse_kstep", float("nan"))), 4),
                   "imag_return": round(float(pol_m.get("imag_return_undisc", float("nan"))), 4), "critic_loss": round(float(pol_m.get("critic_loss", float("nan"))), 4)}
            log.append(row)
            print(f"[{row['time_s']:7.1f}s] {model:5s} it {row['iter']:5d}  wm 8-step {row['wm_nse_8step']:.4f}  imag return {row['imag_return']:.3f}  critic {row['critic_loss']:.4f}", flush=True)
    after = evaluate_levels(policy, heads, eval_libs, args)
    return {"world_model_hidden": MODELS[model], "before": before, "after": after, "log": log,
            "wm_8step_error_final": float(wm_tr.online_error(env.buffer, n=512, k=8)[-1])}


def table(results: dict, levels) -> str:
    lines = ["model  phase   " + "  ".join(f"mu {mu}: return / survived-near-limit / lat err near limit [m]" for mu in levels)]
    for model, res in results.items():
        for phase in ("before", "after"):
            cells = []
            for mu in levels:
                e = res[phase][f"mu_{mu}"]
                cells.append(f"{e['normalized_return']:.3f} / {e['near_limit_survived_share']:.2f} / {e['near_limit_mean_abs_lat']:.2f}")
            lines.append(f"{model:6s} {phase:7s} " + "   ".join(cells))
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse(argv)
    t0 = time.time()
    cfg = env_cfg(args, args.num_envs, 2)
    lib = ReferenceLibrary.concat(libraries(args, cfg, args.refs_per_level, args.seed))
    eval_libs = libraries(args, cfg, args.eval_refs_per_level, args.seed + 1000)
    util = lib.util
    print(f"[{time.time() - t0:7.1f}s] references: {lib.layout.num_sequences} train ({float((util > NEAR_LIMIT).float().mean()):.1%} of steps near the limit), "
          f"{sum(x.layout.num_sequences for x in eval_libs)} held-out, friction levels {args.mu_levels}", flush=True)
    results = {}
    for model in args.models:
        results[model] = train(model, args, lib, eval_libs, t0)
    report = {"args": vars(args), "near_limit_threshold": NEAR_LIMIT, "results": results, "wall_clock_s": round(time.time() - t0, 1)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(table(results, args.mu_levels))
    print(f"report: {out}  ({time.time() - t0:.0f}s)")
    finite = all(math.isfinite(v) for r in results.values() for p in ("after",) for lv in r[p].values() for v in lv.values() if isinstance(v, float))
    return 0 if finite else 1


if __name__ == "__main__":
    sys.exit(main())
