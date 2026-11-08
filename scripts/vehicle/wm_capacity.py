"""Is the world model too small, or just not trained long enough? (VEHICLE_PROPOSAL.md section 10.8)

Takes a replay buffer saved by ``compare_ppo.py --save-buffer-at N`` (the data a SuperMimic run trained its world model
on at iteration N), holds out some of its environments, and trains world models of several sizes from scratch on the
rest with the run's trainer settings. Every ``--eval-every`` updates it reports the 8-step error (the ``nse_kstep`` of
the training log) on fixed held-out windows and on fixed training windows.

    PYTHONPATH=src python scripts/vehicle/wm_capacity.py --buffer logs/sm/replay_it2700.pt --out logs/wm_capacity.json \\
        -- --chassis

Arguments after ``--`` are ``compare_ppo.py`` flags; they must match the run that saved the buffer, since they define the
vehicle, the action dimension and the reference library the windows refer to.

Reading it: if the larger models reach a clearly lower held-out error with the same data and updates, capacity limits
the world model. If all sizes end close together and the error keeps falling with more updates, the number of updates
(training length) limits it. If the held-out error stays well above the training error, the data (coverage) limits it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_ppo
from limit_study import env_cfg, libraries

from supermimic.data.replay_buffer import ReplayBuffer, load_buffer
from supermimic.vehicles.env import ReferenceLibrary, VehicleTrackingEnv
from supermimic.vehicles.learner import VehicleStateSpecCfg
from supermimic.vehicles.threads import limit_threads
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buffer", required=True, help="replay_it<N>.pt from compare_ppo.py --save-buffer-at")
    ap.add_argument("--sizes", nargs="+", default=["256x256", "512x512", "1024x1024", "512x512x512"], help="hidden layer widths")
    ap.add_argument("--updates", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--heldout-envs", type=int, default=32, help="environments of the buffer kept out of training")
    ap.add_argument("--eval-windows", type=int, default=4096)
    ap.add_argument("--out", default="logs/wm_capacity.json")
    ap.add_argument("--threads", type=int, default=None)
    own, rest = list(argv if argv is not None else sys.argv[1:]), []
    if "--" in own:
        i = own.index("--")
        own, rest = own[:i], own[i + 1:]
    return ap.parse_args(own), compare_ppo.parse(rest)


def env_subset(buf: ReplayBuffer, ids: torch.Tensor) -> ReplayBuffer:
    """The rows of some environments of a buffer, as a buffer of its own."""
    sub = ReplayBuffer.__new__(ReplayBuffer)
    for k, v in vars(buf).items():
        setattr(sub, k, v[ids] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == buf.E else v)
    sub.E = len(ids)
    sub._env_ar = torch.arange(sub.E, device=buf.device)
    sub.total_rows = int(sub.valid.sum())
    return sub


def main(argv=None) -> int:
    own, args = parse(argv)
    print(f"cpu threads: {limit_threads(own.threads)}", flush=True)
    dev = args.device
    cfg = env_cfg(args, 2, 2)
    lib = ReferenceLibrary.concat(libraries(args, cfg, args.refs_per_level, args.seed)).to(dev)
    env = VehicleTrackingEnv(cfg, lib, device=dev, record=False)
    heads, task, *_ = compare_ppo.networks(args, env, lib)
    buf = load_buffer(own.buffer, dev)
    if buf.s.shape[-1] != cfg.state_dim:
        sys.exit(f"buffer state dim {buf.s.shape[-1]} does not match the run flags ({cfg.state_dim}); pass the run's flags after --")
    perm = torch.randperm(buf.E, generator=torch.Generator().manual_seed(0)).to(dev)
    test, train = env_subset(buf, perm[: own.heldout_envs]), env_subset(buf, perm[own.heldout_envs:])
    torch.manual_seed(123)
    fixed = {"heldout": test.sample_windows(own.eval_windows, 8), "train": train.sample_windows(own.eval_windows, 8)}
    print(f"buffer {own.buffer}: {buf.size} rows, {train.size} for training ({train.E} envs), {test.size} held out ({test.E} envs)", flush=True)

    out = {"buffer": str(own.buffer), "rows_train": train.size, "rows_heldout": test.size, "updates": own.updates, "models": {}}
    for size in own.sizes:
        hidden = tuple(int(x) for x in size.split("x"))
        torch.manual_seed(args.seed)
        wm = WorldModel(WorldModelCfg(hidden=hidden, state_spec=VehicleStateSpecCfg.from_env(cfg, env.dyn.p)), heads, dev)
        tr = WorldModelTrainer(wm, task, WMTrainerCfg(window=8, batches_per_update=2, batch_size=args.batch, lr=args.lr, compile=None))
        n_par = sum(p.numel() for p in wm.dynamics.parameters())

        def evaluate(tr=tr):
            with torch.no_grad():
                return {k: round(float(tr.loss_on(w)[1]["nse_kstep"]), 4) for k, w in fixed.items()}

        curve, t0 = [], time.time()
        for u in range(own.updates + 1):
            if u % own.eval_every == 0:
                e = evaluate()
                curve.append({"updates": u, **e})
                print(f"[{time.time() - t0:6.0f}s] {size:>12}  update {u:5d}  8-step error held-out {e['heldout']:.4f}  train {e['train']:.4f}", flush=True)
            if u < own.updates:
                tr.update(train)
        out["models"][size] = {"hidden": hidden, "parameters": n_par, "seconds": round(time.time() - t0, 1), "curve": curve}
        Path(own.out).parent.mkdir(parents=True, exist_ok=True)
        Path(own.out).write_text(json.dumps(out, indent=1))

    print("\nsize          parameters   held-out 8-step error at updates " + " ".join(f"{c['updates']:>7d}" for c in out["models"][own.sizes[0]]["curve"]))
    for size, m in out["models"].items():
        print(f"{size:>12}  {m['parameters']:>10d}   {'':>33}" + " ".join(f"{c['heldout']:7.3f}" for c in m["curve"]))
    print(f"\nresults: {own.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
