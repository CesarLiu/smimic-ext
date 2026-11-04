"""E1 Jacobian check: the vehicle world model's gradients against the simulator's, binned by friction use.

    PYTHONPATH=src python scripts/vehicle/jacobian_check.py [--updates 4000] [--out docs/results/vehicle_jacobian_check.json]

1. Data: ``VehicleTrackingEnv`` with ABS / ESC off (the question is tyre saturation; switching logic has no
   derivative to compare) and randomised friction; the behaviour policy replays the references' actions
   with noise, so the data reach and pass the grip limit.
2. A world model (``VehicleStateSpec`` / ``VehicleHeads``) is fitted offline on k = 8 windows.
3. On data from separate held-out references, states are stratified by friction utilisation
   u = sqrt(ax^2 + ay^2) / (mu g) and compared (``supermimic.vehicles.jacobian``): one-step Jacobians,
   8-step action gradients, key sensitivities, and the 8-step prediction error per bin.

Gate (VEHICLE_PROPOSAL.md §6, E1, revised 2026-11-04 in §3.9; ``jacobian.e1_gate``): near saturation
(u > 0.9, bins pooled) the median cosine of the 8-step action gradient exceeds 0.8 and the mean 8-step
prediction error is below 0.2. The one-step Jacobian is reported as a diagnostic. Exit code 0 if the gate
holds. A JSON report is written either way.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from supermimic.vehicles.threads import add_threads_arg, limit_threads, usable_cores
from supermimic.vehicles import jacobian as J
from supermimic.vehicles.dynamics import AssistCfg, VehicleDynamics
from supermimic.vehicles.env import ReferenceLibrary, VehicleEnvCfg, VehicleTrackingEnv
from supermimic.vehicles.learner import VehicleHeads, VehicleStateSpecCfg, VehicleTask
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--refs", type=int, default=32)
    ap.add_argument("--ref-steps", type=int, default=500)
    ap.add_argument("--collect", type=int, default=1000, help="training collection steps")
    ap.add_argument("--eval-collect", type=int, default=350)
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--noise", type=float, default=0.3, help="behaviour-policy action noise (raw action units)")
    ap.add_argument("--per-bin", type=int, default=60, help="states compared per utilisation bin")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="docs/results/vehicle_jacobian_check.json")
    ap.add_argument("--save", default=None, help="optional .pt with the world model and the compared Jacobians")
    add_threads_arg(ap)
    return ap.parse_args(argv)


def env_cfg(args, n, capacity) -> VehicleEnvCfg:
    return VehicleEnvCfg(num_envs=n, replay_capacity=capacity, episode_steps=300, assist=AssistCfg(abs=False, esc=False), mu_range=(0.5, 1.1))


@torch.no_grad()
def collect(env: VehicleTrackingEnv, steps: int, noise: float, gen: torch.Generator):
    lib = env.lib
    for _ in range(steps):
        a = lib.actions[(env.frame - 1).clamp_min(0)]
        env.step(a + noise * torch.randn(a.shape, generator=gen))


def stratify(u: torch.Tensor, per_bin: int, gen: torch.Generator) -> torch.Tensor:
    b = J.bin_index(u)
    keep = []
    for i in range(len(J.BINS) - 1):
        idx = (b == i).nonzero().squeeze(-1)
        keep.append(idx[torch.randperm(idx.numel(), generator=gen)[:per_bin]])
    return torch.cat(keep)


def summarise(metrics: dict[str, torch.Tensor], bins: torch.Tensor, ok: torch.Tensor) -> list[dict]:
    rows = []
    for i in range(len(J.BINS) - 1):
        m = (bins == i) & ok
        row = {"bin": J.bin_label(i), "n": int(m.sum()), "excluded_nonsmooth": int(((bins == i) & ~ok).sum())}
        if m.any():
            for k, v in metrics.items():
                if k.endswith(("_sim", "_wm")):
                    continue
                vals = v[m]
                vals = vals[~torch.isnan(vals)]
                if vals.numel() == 0:
                    row[k] = None
                    continue
                row[k] = round(float(vals.median() if not k.endswith("sign_ok") else vals.mean()), 4)
                if k.endswith("sign_ok"):
                    row[k.replace("sign_ok", "n")] = int(vals.numel())
        rows.append(row)
    return rows


def main(argv=None) -> int:
    args = parse(argv)
    threads = limit_threads(args.threads)
    print(f"cpu threads: {threads} (of {usable_cores()} usable cores)", flush=True)
    t0 = time.time()
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    cfg = env_cfg(args, args.num_envs, args.collect + 100)
    lib = ReferenceLibrary.generate(cfg, num=args.refs, steps=args.ref_steps, seed=args.seed)
    lib_eval = ReferenceLibrary.generate(cfg, num=max(args.refs // 2, 2), steps=args.ref_steps, seed=args.seed + 1000)
    env = VehicleTrackingEnv(cfg, lib, seed=args.seed)
    collect(env, args.collect, args.noise, gen)
    print(f"[{time.time() - t0:6.1f}s] collected {env.buffer.size} rows")

    heads = VehicleHeads(cfg, env.dyn.p.R_w)
    task = VehicleTask(lib, cfg.preview)
    wm = WorldModel(WorldModelCfg(hidden=tuple(args.hidden), state_spec=VehicleStateSpecCfg.from_env(cfg, env.dyn.p)), heads, "cpu")
    tr = WorldModelTrainer(wm, task, WMTrainerCfg(window=args.k, batches_per_update=1, batch_size=args.batch, lr=args.lr, compile=None))
    for u in range(args.updates):
        tr.set_lr(args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * u / args.updates))))
        m = tr.update(env.buffer)
        if (u + 1) % max(args.updates // 8, 1) == 0:
            print(f"[{time.time() - t0:6.1f}s] update {u + 1:5d}  1-step {m['nse_1step']:.4f}  {args.k}-step {m['nse_kstep']:.4f}")
    wm.eval()
    wm.requires_grad_(False)

    # held-out evaluation data
    cfg_e = env_cfg(args, args.num_envs, args.eval_collect + 50)
    env_e = VehicleTrackingEnv(cfg_e, lib_eval, seed=args.seed + 1)
    collect(env_e, args.eval_collect, args.noise, gen)
    w = env_e.buffer.sample_windows(20000, args.k)
    spec = wm.spec
    err_scale = spec.err_scale.double()
    u_all = J.utilisation(w.states[0].double(), w.xi.double())

    # 8-step open-loop error per bin (all sampled windows)
    with torch.no_grad():
        x = w.states[0]
        for t in range(args.k):
            x = wm.step(x, w.a_exec[t], w.xi)
        nse = spec.normalized_sq_error(x, w.states[args.k]).double()
    b_all = J.bin_index(u_all)
    err_rows = [{"bin": J.bin_label(i), "windows": int((b_all == i).sum()), "nse_8step": round(float(nse[b_all == i].mean()), 4) if (b_all == i).any() else None}
                for i in range(len(J.BINS) - 1)]

    # Jacobians on a stratified subset
    sel = stratify(u_all, args.per_bin, gen)
    s0, a0, xi = w.states[0][sel].double(), w.a_exec[0][sel].double(), w.xi[sel].double()
    acts = w.a_exec[:, sel].double()
    dyn = VehicleDynamics(env.dyn.p, cfg.dt, cfg.substeps, cfg.assist)
    step = lambda s, a, x: J.logged_step(dyn, cfg, s, a, x)
    Js1, Js2 = J.sim_jacobian(step, s0, a0, xi, err_scale, 1e-2), J.sim_jacobian(step, s0, a0, xi, err_scale, 1e-3)
    smooth = J.compare(Js1, Js2)["cos"] > 0.98
    Jw = J.wm_jacobian(wm, s0.float(), a0.float(), xi.float(), spec.err_scale).double()
    one = J.compare(Jw, Js1)
    Ks1 = J.sim_kstep_action_grad(step, s0, acts, xi, err_scale, 1e-2)
    Ks2 = J.sim_kstep_action_grad(step, s0, acts, xi, err_scale, 1e-3)
    rows = J.LEARNED
    smooth_k = J._cos(Ks1[:, rows], Ks2[:, rows]) > 0.98
    Kw = J.wm_kstep_action_grad(wm, s0.float(), acts.float(), xi.float(), spec.err_scale).double()
    kcos = J._cos(Kw[:, rows], Ks1[:, rows])
    kmag = Kw[:, rows].reshape(len(sel), -1).norm(dim=-1) / Ks1[:, rows].reshape(len(sel), -1).norm(dim=-1).clamp_min(1e-12)
    bins = J.bin_index(J.utilisation(s0, xi))
    one_rows = summarise(one, bins, smooth)
    k_rows = summarise({"cos": kcos, "mag_ratio": kmag}, bins, smooth_k)

    # gate
    sat = (bins >= J.SAT_BIN) & smooth
    sat_cos = float(one["cos"][sat].median()) if sat.any() else float("nan")
    sat_k = (bins >= J.SAT_BIN) & smooth_k
    sat_kcos = float(kcos[sat_k].median()) if sat_k.any() else float("nan")
    sat_err = J.saturation_error(err_rows)
    passed, fails = J.e1_gate(err_rows, sat_kcos)

    print(f"\n8-step normalised error per utilisation bin (held-out, {w.states.shape[1]} windows):")
    for r in err_rows:
        print(f"  u {r['bin']:8s} windows {r['windows']:6d}  nse {r['nse_8step']}")
    print("\none-step Jacobian (learned rows; medians; sign_ok = fraction):")
    for r in one_rows:
        print("  " + "  ".join(f"{k} {v}" for k, v in r.items()))
    print(f"\n{args.k}-step gradient w.r.t. the first action:")
    for r in k_rows:
        print("  " + "  ".join(f"{k} {v}" for k, v in r.items()))
    print(f"\nnear saturation (u > 0.9): {args.k}-step action-gradient cosine median {sat_kcos:.3f} (n {int(sat_k.sum())}), "
          f"{args.k}-step error {sat_err:.3f}; one-step cosine median {sat_cos:.3f} (diagnostic, n {int(sat.sum())})")
    if args.save:
        torch.save({"wm": wm.state_dict(), "spec_cfg": wm.cfg.state_spec, "s0": s0, "a0": a0, "xi": xi, "acts": acts, "J_sim": Js1, "J_wm": Jw,
                    "K_sim": Ks1, "K_wm": Kw, "bins": bins, "smooth": smooth, "smooth_k": smooth_k}, args.save)
    report = {
        "args": vars(args), "wall_clock_s": round(time.time() - t0, 1), "train_rows": env.buffer.size,
        "nse_8step_by_bin": err_rows, "one_step_by_bin": one_rows, "kstep_by_bin": k_rows,
        "saturation": {"one_step_cos_median": sat_cos, "n": int(sat.sum()), "kstep_cos_median": sat_kcos, "n_k": int(sat_k.sum()), "kstep_error": sat_err},
        "gate": {"definition": "revised 2026-11-04: u > 0.9 pooled, k-step action-gradient cosine median > 0.8 and mean k-step error < 0.2",
                 "passed": passed, "failures": fails},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("PASSED" if passed else "FAILED: " + "; ".join(fails))
    print(f"report: {out}  ({time.time() - t0:.0f}s)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
