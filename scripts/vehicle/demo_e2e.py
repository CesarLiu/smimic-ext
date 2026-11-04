"""End-to-end demo: SuperMimic on the vehicle, from simulator data to a tracking policy.

    PYTHONPATH=src python scripts/vehicle/demo_e2e.py [--iters 1500] [--device cpu]

1. Reference trajectories: a training library and a separate held-out library (different seed).
2. ``VehicleTrackingEnv`` collects transitions with the current policy plus exploration noise into the
   replay buffer; episode starts come from the return-based reset sampler; the sim context is randomised.
3. After a warm-up, every iteration fits the world model (``VehicleStateSpec``, ``VehicleHeads``) on k-step
   replay windows; once it has had a head start, the policy and critic are improved by BPTT through
   16-step imagined rollouts (``PolicyTrainer`` with ``VehicleTask``), exactly the G1 code path.
4. The policy is evaluated deterministically in the simulator (nominal context, no noise) on the held-out
   references, before and after training: normalised return (episode reward over the best attainable for
   the whole reference; early termination scores low), termination rate and lateral error.

The demo succeeds (exit code 0) when the trained policy beats the untrained one by the margins in
``check`` and the world model's 8-step prediction error is small; it writes a JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.policy.nets import Critic, CriticCfg, NetIO, Policy, PolicyCfg
from supermimic.vehicles import tracking
from supermimic.vehicles.env import ReferenceLibrary, VehicleEnvCfg, VehicleTrackingEnv
from supermimic.vehicles.learner import VehicleHeads, VehicleStateSpecCfg, VehicleTask
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100, help="collection-only iterations before the world model learns")
    ap.add_argument("--policy-warmup", type=int, default=300, help="iterations before the policy learns (world model first)")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--refs", type=int, default=32)
    ap.add_argument("--eval-refs", type=int, default=16)
    ap.add_argument("--ref-steps", type=int, default=500)
    ap.add_argument("--episode-steps", type=int, default=250)
    ap.add_argument("--history", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--action-noise", type=float, default=0.1)
    ap.add_argument("--no-domain-randomization", action="store_true")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--out", default="logs/vehicle_demo/report.json")
    return ap.parse_args(argv)


class ObsHistory:
    """Per-env observation history (oldest first), backfilled at episode starts."""

    def __init__(self, heads, n, h, device):
        self.heads, self.h = heads, h
        self.buf = torch.zeros(h, n, heads.obs_dim, device=device)

    def reset(self, ids, obs):
        self.buf[:, ids] = obs[ids][None]

    def push(self, obs):
        self.buf = torch.cat([self.buf[1:], obs[None]], 0)

    def flat(self):
        return self.heads.flatten_obs_history(self.buf)


@torch.no_grad()
def evaluate(policy, heads, lib, env_cfg, history, device) -> dict:
    """Deterministic play of every held-out reference from its first step, nominal context, no noise."""
    n = lib.layout.num_sequences
    length = int(lib.lengths[0])
    cfg = VehicleEnvCfg(num_envs=n, domain_randomization=False, reset_noise=False, episode_steps=length, replay_capacity=2,
                        preview=env_cfg.preview, tracking=env_cfg.tracking, substeps=env_cfg.substeps)
    env = VehicleTrackingEnv(cfg, lib, device=device, record=False)
    env.forced_frame[:] = lib.layout.seq_start
    obs = env.reset()
    hist = ObsHistory(heads, n, history, device)
    hist.reset(torch.arange(n, device=device), obs["policy"])
    ret = torch.zeros(n, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    lat = torch.zeros(n, device=device)
    steps = torch.zeros(n, device=device)
    for _ in range(length - 1):
        a = policy(hist.flat(), obs["goal"])
        ref = lib.ref(env.frame)
        obs, r, _term, _trunc, info = env.step(a)
        done = info["done"]
        e = tracking.errors(env.s, ref) if not done.any() else None
        ret += r * alive
        if e is not None:
            lat += e["lat"].abs() * alive
        steps += alive.float()
        alive &= ~done
        if not alive.any():
            break
        hist.push(obs["policy"])
    terminated = (steps < length - 1)
    norm = ret / (env_cfg.tracking.max_step_reward * (length - 1))
    return {
        "normalized_return": float(norm.mean()),
        "termination_rate": float(terminated.float().mean()),
        "mean_abs_lateral_error_m": float((lat / steps.clamp_min(1)).mean()),
        "mean_steps": float(steps.mean()),
    }


def check(before: dict, after: dict, wm_err: float) -> list[str]:
    """Success: on held-out references the trained policy earns at least twice the untrained policy's return
    (and 0.15 more), stays on the reference at least twice as long, and the world model stays accurate."""
    fails = []
    if not after["normalized_return"] >= max(2.0 * before["normalized_return"], before["normalized_return"] + 0.15):
        fails.append(f"normalized return {after['normalized_return']:.3f} did not improve enough over {before['normalized_return']:.3f}")
    if not after["mean_steps"] >= 2.0 * before["mean_steps"]:
        fails.append(f"mean episode length {after['mean_steps']:.0f} not twice {before['mean_steps']:.0f}")
    if not (math.isfinite(wm_err) and wm_err < 3.0):
        fails.append(f"world-model 8-step error {wm_err:.3f} is not small")
    return fails


def main(argv=None) -> int:
    args = parse(argv)
    dev = args.device
    torch.manual_seed(args.seed)
    t0 = time.time()
    env_cfg = VehicleEnvCfg(num_envs=args.num_envs, episode_steps=args.episode_steps, replay_capacity=args.episode_steps + 50,
                            domain_randomization=not args.no_domain_randomization)
    lib = ReferenceLibrary.generate(env_cfg, num=args.refs, steps=args.ref_steps, seed=args.seed, device=dev)
    lib_eval = ReferenceLibrary.generate(env_cfg, num=args.eval_refs, steps=args.ref_steps, seed=args.seed + 1000, device=dev)
    print(f"[{time.time() - t0:6.1f}s] references: {args.refs} train / {args.eval_refs} held-out x {args.ref_steps} steps")

    env = VehicleTrackingEnv(env_cfg, lib, device=dev, seed=args.seed)
    heads = VehicleHeads(env_cfg, env.dyn.p.R_w)
    task = VehicleTask(lib, env_cfg.preview)
    wm = WorldModel(WorldModelCfg(hidden=(256, 256), state_spec=VehicleStateSpecCfg(dt=env_cfg.dt, max_delay=env_cfg.max_delay)), heads, dev)
    io = NetIO.from_interfaces(wm.spec, heads, task, args.history)
    pcfg = PolicyCfg(encoder_hidden=(128,), decoder_hidden=(256, 128), latent_dim=32, history=args.history, decoder_last_gain=0.1)
    policy, critic = Policy(pcfg, io).to(dev), Critic(CriticCfg(hidden=(256, 128)), io=io).to(dev)
    wm_tr = WorldModelTrainer(wm, task, WMTrainerCfg(window=8, batches_per_update=2, batch_size=args.batch, lr=args.lr, compile=None))
    pol_tr = PolicyTrainer(wm, policy, critic, task, PolicyTrainerCfg(horizon=args.horizon, batch_size=args.batch, history=args.history, lr=args.lr,
                                                                       critic_lr=args.lr, critic_epochs=2, critic_minibatches=2, compile=None,
                                                                       value_clip=env_cfg.tracking.max_step_reward / (1 - 0.99)))
    task_eval_heads = heads  # same robot, same heads; the held-out task only changes the references

    before = evaluate(policy, task_eval_heads, lib_eval, env_cfg, args.history, dev)
    print(f"[{time.time() - t0:6.1f}s] untrained policy on held-out references: {before}")

    n = args.num_envs
    obs = env.reset()
    hist = ObsHistory(heads, n, args.history, dev)
    hist.reset(torch.arange(n, device=dev), obs["policy"])
    noise_half = heads.obs_noise_half().to(dev)
    ep_ret = torch.zeros(n, device=dev)
    ep_len = torch.zeros(n, device=dev)
    finished = []
    log = []
    wm_m, pol_m = {}, {}
    for it in range(args.iters):
        with torch.no_grad():
            a = policy(hist.flat(), obs["goal"])
            obs, r, _term, _trunc, info = env.step(a, args.action_noise * torch.randn_like(a))
        ep_ret += r
        ep_len += 1
        done = info["done"]
        if done.any():
            ids = done.nonzero().squeeze(-1)
            finished += (ep_ret[ids] / (env_cfg.tracking.max_step_reward * ep_len[ids])).tolist()
            ep_ret[ids], ep_len[ids] = 0.0, 0.0
        noisy = obs["policy"] + (torch.rand_like(obs["policy"]) * 2 - 1) * noise_half
        hist.push(noisy)
        if done.any():
            hist.reset(ids, noisy)
        if it >= args.warmup:
            wm_m = wm_tr.update(env.buffer)
        if it >= args.policy_warmup:
            pol_m = pol_tr.update(env.buffer)
        if (it + 1) % args.log_every == 0:
            recent = finished[-200:]
            row = {"iter": it + 1, "time_s": round(time.time() - t0, 1), "train_episode_return": round(sum(recent) / max(len(recent), 1), 4),
                   "wm_nse_8step": round(float(wm_m.get("nse_kstep", float("nan"))), 5), "imag_return": round(float(pol_m.get("imag_return_undisc", float("nan"))), 4),
                   "critic_loss": round(float(pol_m.get("critic_loss", float("nan"))), 5)}
            log.append(row)
            print(f"[{row['time_s']:6.1f}s] it {row['iter']:5d}  train return {row['train_episode_return']:.3f}  wm 8-step {row['wm_nse_8step']:.4f}  "
                  f"imag return {row['imag_return']:.3f}  critic {row['critic_loss']:.4f}")

    wm_err = float(wm_tr.online_error(env.buffer, n=512, k=8)[-1])
    after = evaluate(policy, task_eval_heads, lib_eval, env_cfg, args.history, dev)
    print(f"[{time.time() - t0:6.1f}s] trained policy on held-out references:   {after}")
    print(f"world-model 8-step normalised error on recent replay: {wm_err:.4f}")
    fails = check(before, after, wm_err)
    report = {"args": vars(args), "before": before, "after": after, "wm_8step_error": wm_err, "log": log, "passed": not fails, "failures": fails,
              "wall_clock_s": round(time.time() - t0, 1)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("PASSED" if not fails else "FAILED: " + "; ".join(fails))
    print(f"report: {out}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
