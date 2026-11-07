"""Integrated-chassis demo (VEHICLE_PROPOSAL.md §10, C1): the same learners with the plain 2-D action (front-wheel
angle, acceleration) and with the 5-D chassis action (plus rear-wheel angle, torque-vectoring yaw moment and the
front / rear roll-stiffness split), trained on the same random-curve library with the same budget, then compared on
held-out curves, the S-bend and the elk test.

    PYTHONPATH=src python scripts/vehicle/chassis_demo.py --out-dir logs/chassis_demo
    # reuse finished compare_ppo.py runs instead of training:  --plain-from DIR --chassis-from DIR

Steps:
1. train (``compare_ppo.py``, SuperMimic and PPO, iteration budgets so that both variants get the same amount of
   training whatever the machine) unless ``--plain-from`` / ``--chassis-from`` point at finished runs;
2. collect from their reports: held-out and S-bend returns, S-bend lateral error, and sample efficiency (simulator
   steps until the held-out return first reaches ``--target-return`` at the evaluation milestones);
3. the elk test (``elk_test.py``) for every policy, plus the on-board NMPC (2-D, the conventional controller);
4. ``summary.json``, a table on stdout, and two videos: the elk test and the S-bend with all four learned policies.

One seed per variant; the comparison is a demonstration, not a statistical result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_ppo
import elk_test
import render_scurve
from compare_ppo import networks
from limit_study import env_cfg

from supermimic.vehicles.env import ReferenceLibrary, VehicleTrackingEnv
from supermimic.vehicles.learner import VehicleHeads
from supermimic.vehicles.threads import add_threads_arg, limit_threads

VARIANTS = ("plain", "chassis")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plain-from", default=None, help="a finished compare_ppo.py output directory (2-D action)")
    ap.add_argument("--chassis-from", default=None, help="a finished compare_ppo.py --chassis output directory (5-D action)")
    ap.add_argument("--sm-max-iters", type=int, default=2700)
    ap.add_argument("--ppo-max-updates", type=int, default=408)
    ap.add_argument("--ppo-envs", type=int, default=1024)
    ap.add_argument("--train-extra", nargs=argparse.REMAINDER, default=[], help="further compare_ppo.py arguments for training (last)")
    ap.add_argument("--target-return", type=float, default=0.7, help="held-out return for the sample-efficiency count")
    ap.add_argument("--elk-dry-kmh", type=float, nargs="*", default=[40, 50, 60])
    ap.add_argument("--elk-wet-kmh", type=float, nargs="*", default=[30, 40])
    ap.add_argument("--refs-cache", default="docs/results/vehicle_elk_refs.pt")
    ap.add_argument("--video-elk", default="1.0:50,0.5:40", help="elk cases in the video (mu:km/h, comma separated)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--out-dir", default="logs/chassis_demo")
    ap.add_argument("--device", default="cpu")
    add_threads_arg(ap)
    return ap.parse_args(argv)


def train(args, variant: str, out: Path) -> Path:
    argv = ["--methods", "supermimic", "ppo", "--budget-min", "1000", "--eval-every-s", "600", "--sm-max-iters", str(args.sm_max_iters),
            "--ppo-max-updates", str(args.ppo_max_updates), "--ppo-envs", str(args.ppo_envs), "--device", args.device, "--out-dir", str(out)]
    if variant == "chassis":
        argv.append("--chassis")
    print(f"=== training the {variant} variant into {out}", flush=True)
    compare_ppo.main(argv + list(args.train_extra))
    return out


def steps_to(curve: list, key: str, target: float):
    for row in curve:
        if row[key] >= target:
            return row["env_steps"]
    return None


def summarise(report: dict, target: float) -> dict:
    out = {}
    for m, r in report["methods"].items():
        mean = lambda d: sum(v["normalized_return"] for v in d.values()) / len(d)  # noqa: E731
        lat = {k: sum(abs(x) for x in t["lat"]) / max(len(t["lat"]), 1) for k, t in r["scurve_traj"].items()}
        out[m] = {"heldout_return": {k: v["normalized_return"] for k, v in r["heldout_final"].items()}, "heldout_mean": mean(r["heldout_final"]),
                  "scurve_return": {k: v["normalized_return"] for k, v in r["scurve_final"].items()}, "scurve_mean": mean(r["scurve_final"]),
                  "scurve_completed": {k: v["termination_rate"] == 0 for k, v in r["scurve_final"].items()}, "scurve_mean_abs_lat_m": lat,
                  "env_steps": r.get("env_steps"), f"steps_to_heldout_{target}": steps_to(r["curve"], "heldout_return", target) if r["curve"] else None}
    return out


def load_policy(run_dir: Path, method: str, device):
    blob = torch.load(run_dir / "policies.pt", map_location="cpu")
    run = SimpleNamespace(**{**vars(compare_ppo.parse([])), **blob["args"], "device": device, "eval_noise": False, "eval_delay": 0, "eval_mismatch": False})
    cfg = env_cfg(run, 2, 2)
    return run, cfg, blob["policies"][method]


def drive_all(policies: dict, lib, device):
    """policies: name -> (run, cfg, state_dict) -> per-name drive records on ``lib``."""
    runs = {}
    for name, (run, cfg, sd) in policies.items():
        env = VehicleTrackingEnv(cfg, lib, device=device, record=False)
        heads, _t, _s, _io, _pc, policy, _c = networks(run, env, lib)
        policy.load_state_dict(sd)
        policy.eval()
        runs[name] = render_scurve.drive(policy, heads, lib, run)
    return runs


def videos(args, dirs: dict, out: Path):
    pol = {}
    for variant, d in dirs.items():
        for m in ("supermimic", "ppo"):
            name = m if variant == "plain" else f"{m}_chassis"
            pol[name] = load_policy(d, m, args.device)
    order = ["supermimic", "supermimic_chassis", "ppo", "ppo_chassis"]
    pol = {k: pol[k] for k in order if k in pol}
    run0, cfg0, _ = pol[order[0]]
    params = VehicleTrackingEnv(cfg0, ReferenceLibrary.s_curve(cfg0, speed=run0.scurve_speed, ay_frac=run0.scurve_ay, mu=1.0), record=False).dyn.p
    vargs = SimpleNamespace(fps=args.fps, dpi=args.dpi, view=36.0)
    # S-bend
    segs = []
    for mu in run0.mu_levels:
        lib = ReferenceLibrary.s_curve(cfg0, speed=run0.scurve_speed, ay_frac=run0.scurve_ay, mu=mu).to(args.device)
        runs = drive_all(pol, lib, args.device)
        lat_max = max(max(abs(v) for v in r["lat"]) for r in runs.values())
        segs.append({"mu": mu, "speed": run0.scurve_speed, "ay": run0.scurve_ay, "ref": {"x": lib.x.cpu().tolist(), "y": lib.y.cpu().tolist(), "util": lib.util.cpu().tolist()},
                     "runs": runs, "length": int(lib.lengths[0]), "lat_lim": (-max(0.5, 1.15 * lat_max), max(0.5, 1.15 * lat_max)),
                     "title": f"S-bend at μ = {mu}: 2-D action vs integrated chassis (5-D)"})
    print("video:", render_scurve.render(segs, params, vargs, out / "chassis_scurve.mp4"), flush=True)
    # elk test
    blob = torch.load(args.refs_cache, map_location="cpu")
    segs = []
    for case in args.video_elk.split(","):
        mu, kmh = (float(v) for v in case.split(":"))
        lib, tr = elk_test.load_ref(blob, mu, kmh, args.device)
        n_end = int((lib.x.cpu() < tr.lanes()[-1][1] + elk_test.END_MARGIN).sum())
        runs = {}
        for name, r in drive_all(pol, lib, args.device).items():
            r = elk_test.cut(r, n_end)
            seen, cum = set(), []
            for tset in tr.cone_hits(r["x"], r["y"], r["yaw"]):
                seen |= tset
                cum.append(len(seen))
            r["hits"] = cum
            runs[name] = r
        lat_max = max(max(abs(v) for v in r["lat"]) for r in runs.values())
        segs.append({"mu": mu, "kmh": kmh, "layout": "elk", "ref": {"x": lib.x.cpu().tolist()[:n_end], "y": lib.y.cpu().tolist()[:n_end], "util": lib.util.cpu().tolist()[:n_end]},
                     "runs": runs, "length": n_end, "cones": tr.cones(), "y_mid": tr.lanes()[1][2] / 2, "lat_lim": (-max(0.6, 1.15 * lat_max), max(0.6, 1.15 * lat_max)),
                     "title": f"Elk test, μ = {mu}, {kmh:.0f} km/h: 2-D action vs integrated chassis (5-D)"})
    vargs.view = 44.0
    print("video:", render_scurve.render(segs, params, vargs, out / "chassis_elk.mp4"), flush=True)


def main(argv=None) -> int:
    args = parse(argv)
    limit_threads(args.threads)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dirs = {"plain": Path(args.plain_from) if args.plain_from else None, "chassis": Path(args.chassis_from) if args.chassis_from else None}
    for v in VARIANTS:
        if dirs[v] is None:
            dirs[v] = train(args, v, out / v)
    summary = {"args": vars(args), "training": {}, "elk": {}}
    for v in VARIANTS:
        summary["training"][v] = summarise(json.loads((dirs[v] / "report.json").read_text()), args.target_return)
    # elk test: every policy, and the on-board NMPC once (it ignores the chassis actuators)
    for v in VARIANTS:
        methods = ["supermimic", "ppo"] + (["nmpc_onboard"] if v == "plain" else [])
        try:
            import casadi  # noqa: F401
        except ImportError:
            methods = [m for m in methods if m != "nmpc_onboard"]
        elk_argv = ["--policies", str(dirs[v] / "policies.pt"), "--methods", *methods, "--dry-kmh", *[str(k) for k in args.elk_dry_kmh],
                    "--wet-kmh", *[str(k) for k in args.elk_wet_kmh], "--refs-cache", args.refs_cache, "--out-dir", str(out / f"elk_{v}"),
                    "--no-video", "--no-plots", "--device", args.device]
        elk_test.main(elk_argv)
        rep = json.loads((out / f"elk_{v}" / "report.json").read_text())
        for c in rep["cases"]:
            key = f"mu_{c['mu']}_{c['kmh']:g}kmh"
            for m, r in c["results"].items():
                name = m if v == "plain" or m.startswith("nmpc") else f"{m}_chassis"
                summary["elk"].setdefault(key, {})[name] = {k: r[k] for k in ("passed", "completed", "cones_hit", "overhang_steps", "max_lateral_error_m")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    # table
    print("\n=== training results (held-out / S-bend mean return; S-bend mean |lateral error| per mu; steps to the target return)")
    for v in VARIANTS:
        for m, r in summary["training"][v].items():
            name = m if v == "plain" or m.startswith("nmpc") else f"{m}_chassis"
            print(f"{name:20s} held-out {r['heldout_mean']:.3f}  S-bend {r['scurve_mean']:.3f}  lat " + " ".join(f"{x:.2f}" for x in r["scurve_mean_abs_lat_m"].values())
                  + f"  steps to {args.target_return}: {r[f'steps_to_heldout_{args.target_return}']}  (trained {r['env_steps']} steps)")
    print("\n=== elk test: cones hit (P = passed)")
    names = sorted({n for c in summary["elk"].values() for n in c}, key=lambda n: ("nmpc" in n, n))
    print("case".ljust(18) + "".join(n[:20].rjust(21) for n in names))
    for key, c in summary["elk"].items():
        print(key.ljust(18) + "".join(("P" if c[n]["passed"] else (str(c[n]["cones_hit"]) if c[n]["completed"] else "off")).rjust(21) if n in c else "-".rjust(21) for n in names))
    print(f"\nsummary: {out / 'summary.json'}", flush=True)
    if not args.no_video:
        videos(args, dirs, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
