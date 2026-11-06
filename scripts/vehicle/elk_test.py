"""Elk test (moose test) demo: the ISO 3888-2 obstacle-avoidance lane change, driven by the trained policies and
the NMPC baselines at several entry speeds, scored against the cones.

    PYTHONPATH=src python scripts/vehicle/elk_test.py --policies logs/compare/seed0/policies.pt --out-dir logs/elk

The track (``supermimic.vehicles.elk``): sections 12 / 13.5 / 11 / 12.5 / 12 m; lanes 1.1 b + 0.25, b + 1 (1 m gap to the
entry lane, so about 3.5 m to the left) and 1.3 b + 0.25 (at least 3 m) wide, b = 1.8 m. Taken from secondary
sources, not the standard itself; ``elk.py`` lists what is uncertain.
The policies were trained on random curves and have never seen this manoeuvre. Every method gets the same
reference (the best line through the cones, as driven by the privileged NMPC) at the case's speed. A run passes when
it is not terminated and touches no cone (the standard's "no cone displaced"; cone size and spacing assumed);
overhang steps (a body corner outside a lane line) are reported as a stricter measure. Cases: dry road (``--dry-kmh``)
and wet road (``--wet-kmh``, mu ``--wet-mu``), each at several entry speeds; the highest passed speed per method is
the usual moose-test score. A speed only becomes a case if the privileged NMPC passes it itself.

Generating the references needs casadi (best line and NMPC); they are cached in ``--refs-cache`` (committed in
``docs/results``) so that the policies alone can be evaluated without it. ``--eval-noise`` / ``--eval-delay`` /
``--eval-mismatch`` as in ``compare_ppo.py``.
Writes ``report.json``, ``elk_paths.png`` and ``elk.mp4`` to ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_ppo
import render_scurve
from compare_ppo import COLORS, networks
from limit_study import env_cfg

from supermimic.vehicles.elk import ElkTrack, best_line, elk_reference
from supermimic.vehicles.env import ReferenceLibrary, VehicleTrackingEnv
from supermimic.vehicles.learner import VehicleHeads
from supermimic.vehicles.threads import add_threads_arg, limit_threads

METHODS = ("supermimic", "ppo", "nmpc", "nmpc_onboard")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", default=None, help="policies.pt written by compare_ppo.py (SuperMimic / PPO)")
    ap.add_argument("--methods", nargs="+", default=None, choices=METHODS, help="default: the policies in --policies, plus nmpc_onboard if casadi is installed")
    ap.add_argument("--dry-mu", type=float, default=1.0)
    ap.add_argument("--dry-kmh", type=float, nargs="*", default=[40, 45, 50, 55, 60, 65, 70, 75])
    ap.add_argument("--wet-mu", type=float, default=0.5)
    ap.add_argument("--wet-kmh", type=float, nargs="*", default=[30, 35, 40, 45, 50])
    ap.add_argument("--refs-cache", default="docs/results/vehicle_elk_refs.pt", help="references: loaded when present (no casadi needed), else generated and saved here")
    ap.add_argument("--eval-noise", action="store_true")
    ap.add_argument("--eval-delay", type=int, default=0)
    ap.add_argument("--eval-mismatch", action="store_true")
    ap.add_argument("--nmpc-assumed-delay", type=int, default=1)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--view", type=float, default=44.0, help="width of the following camera's view [m]")
    ap.add_argument("--video-cases", default="all", help="'all', or a comma list like 1.0:80,0.5:60 (mu:km/h)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--out-dir", default="logs/elk")
    ap.add_argument("--device", default="cpu")
    add_threads_arg(ap)
    return ap.parse_args(argv)


def ref_key(mu: float, kmh: float) -> str:
    return f"{float(mu)}:{float(kmh):g}"


def pack_ref(lib, tr) -> dict:
    return {"lib": {k: getattr(lib, k).cpu() for k in lib._FIELDS}, "util": lib.util.cpu(), "lengths": lib.lengths, "seq_mu": lib.seq_mu, "track": asdict(tr)}


def load_ref(blob, mu, kmh, device):
    d = blob[ref_key(mu, kmh)]
    lib = ReferenceLibrary(**{k: v.to(device) for k, v in d["lib"].items()}, lengths=tuple(d["lengths"]), util=d["util"].to(device), seq_mu=tuple(d["seq_mu"]))
    return lib, ElkTrack(**d["track"])


def get_cases(args, cfg):
    wanted = [(args.dry_mu, k) for k in args.dry_kmh] + [(args.wet_mu, k) for k in args.wet_kmh]
    path = Path(args.refs_cache)
    blob = torch.load(path, map_location="cpu") if path.exists() else {}
    cases, new, line = [], False, None
    for mu, kmh in wanted:
        key = ref_key(mu, kmh)
        if key not in blob:
            print(f"generating the reference for mu {mu}, {kmh:.0f} km/h (privileged NMPC, needs casadi)", flush=True)
            try:
                if line is None:
                    from supermimic.vehicles.params import VehicleParams

                    p = VehicleParams()
                    tr0 = ElkTrack()
                    tr0._front, tr0._rear = p.a + tr0.front_overhang, p.b + tr0.rear_overhang
                    line = best_line(tr0)
                    print(f"least-curvature line: peak curvature {line[2]:.4f} 1/m (lateral acceleration 1 g at {3.6 * (9.81 / line[2]) ** 0.5:.0f} km/h)", flush=True)
                lib, tr = elk_reference(cfg, kmh / 3.6, mu=mu)
                print(f"  line clearance {tr.clearance:.2f} m", flush=True)
            except RuntimeError as e:  # the NMPC cannot drive the track this fast: no reference, no case
                print(f"  skipped: {e}", flush=True)
                continue
            blob[key], new = pack_ref(lib, tr), True
        lib, tr = load_ref(blob, mu, kmh, args.device)
        cases.append((mu, kmh, lib, tr))
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, path)
        print(f"references saved to {path}", flush=True)
    return cases


END_MARGIN = 3.0  # [m] the run ends this far past the last cone: the reference's own end (the preview runs out) is not part of the manoeuvre


def cut(out: dict, n: int) -> dict:
    """The first ``n`` samples of every per-step list; ``terminated`` only if that happened before ``n``."""
    res = {k: (v[:n] if isinstance(v, list) else v) for k, v in out.items()}
    res["terminated"] = bool(out["terminated"] and len(out["x"]) < n)
    return res


def main(argv=None) -> int:
    args = parse(argv)
    limit_threads(args.threads)
    blob = torch.load(args.policies, map_location="cpu") if args.policies else None
    run = SimpleNamespace(**{**vars(compare_ppo.parse([])), **(blob["args"] if blob else {}), "device": args.device, "eval_noise": args.eval_noise,
                             "eval_delay": args.eval_delay, "eval_mismatch": args.eval_mismatch, "nmpc_assumed_delay": args.nmpc_assumed_delay})
    cfg = env_cfg(run, 2, 2)
    methods = args.methods
    if methods is None:
        methods = list(blob["policies"]) if blob else []
        try:
            import casadi  # noqa: F401

            methods.append("nmpc_onboard")
        except ImportError:
            print("casadi not installed: no NMPC in this run")
    if not methods:
        sys.exit("nothing to run: give --policies and/or --methods nmpc_onboard")
    cases = get_cases(args, cfg)
    heads = VehicleHeads(cfg, VehicleTrackingEnv(cfg, cases[0][2], device=args.device, record=False).dyn.p.R_w)
    ctls = {}
    if any(m.startswith("nmpc") for m in methods):
        from supermimic.baselines.nmpc import NMPC

        for m in methods:
            if m.startswith("nmpc"):
                ctls[m] = NMPC(cfg, cfg=compare_ppo.nmpc_cfg(run, m))
    report = {"args": vars(args), "cases": []}
    segments = []
    for mu, kmh, lib, tr in cases:
        env = VehicleTrackingEnv(cfg, lib, device=args.device, record=False)
        x_end = tr.lanes()[-1][1] + END_MARGIN
        n_end = int((lib.x.cpu() < x_end).sum())  # samples until the car should be past the last cone
        runs, res = {}, {}
        for m in methods:
            if m.startswith("nmpc"):
                ctls[m].reset()
                out = render_scurve.drive(None, heads, lib, run, controller=ctls[m])
            else:
                _h, _t, _s, _io, _pc, policy, _c = networks(run, env, lib)
                policy.load_state_dict(blob["policies"][m])
                policy.eval()
                out = render_scurve.drive(policy, heads, lib, run)
            out = cut(out, n_end)
            touched = tr.cone_hits(out["x"], out["y"], out["yaw"])
            seen, cum = set(), []
            for tset in touched:
                seen |= tset
                cum.append(len(seen))
            out["hits"] = cum  # cones touched so far (for the video)
            overhang = sum(1 for h in tr.hits(out["x"], out["y"], out["yaw"]) if h)
            runs[m] = out
            lat = [abs(v) for v in out["lat"]]
            completed = not out["terminated"] and len(out["x"]) >= n_end
            res[m] = {"completed": completed, "passed": bool(completed and not seen), "cones_hit": len(seen), "overhang_steps": overhang, "steps": len(out["x"]),
                      "max_lateral_error_m": max(lat), "mean_lateral_error_m": sum(lat) / len(lat), "max_friction_use": max(out["util"])}
        print(f"mu {mu}, {kmh:.0f} km/h (reference friction use up to {float(lib.util.max()):.2f}): " +
              "  ".join(f"{m} {'PASS' if r['passed'] else ('HIT cones' if r['completed'] else 'LEFT TRACK')}" for m, r in res.items()), flush=True)
        report["cases"].append({"mu": mu, "kmh": kmh, "reference_util_max": float(lib.util.max()), "line_clearance_m": tr.clearance, "results": res})
        ref = {"x": lib.x.cpu().tolist()[:n_end], "y": lib.y.cpu().tolist()[:n_end], "util": lib.util.cpu().tolist()[:n_end]}
        lat_max = max(max(abs(v) for v in r["lat"]) for r in runs.values())
        segments.append({"mu": mu, "kmh": kmh, "layout": "elk", "ref": ref, "runs": runs, "length": n_end, "cones": tr.cones(), "y_mid": tr.lanes()[1][2] / 2,
                         "lat_lim": (-max(0.6, 1.15 * lat_max), max(0.6, 1.15 * lat_max)), "track": tr,
                         "title": f"Elk test, μ = {mu}, entry speed {kmh:.0f} km/h ({kmh / 3.6:.1f} m/s)   policies trained on random curves, never on this manoeuvre"})
    # highest passed speed per method and friction
    best = {}
    for mu in sorted({c["mu"] for c in report["cases"]}):
        for m in methods:
            ok = [c["kmh"] for c in report["cases"] if c["mu"] == mu and c["results"][m]["passed"]]
            best[f"mu_{mu}"] = {**best.get(f"mu_{mu}", {}), m: max(ok) if ok else None}
    report["highest_passed_kmh"] = best
    print("highest passed entry speed [km/h]:", json.dumps(best))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    if not args.no_plots:
        plots(segments, out_dir / "elk_paths.png")
    if not args.no_video:
        sel = segments
        if args.video_cases != "all":
            keys = {tuple(float(x) for x in c.split(":")) for c in args.video_cases.split(",")}
            sel = [s for s in segments if (s["mu"], s["kmh"]) in keys]
        vargs = SimpleNamespace(fps=args.fps, dpi=args.dpi, view=args.view)
        print("video:", render_scurve.render(sel, VehicleTrackingEnv(cfg, cases[0][2], device=args.device, record=False).dyn.p, vargs, out_dir / "elk.mp4"))
    return 0


def plots(segments, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(segments)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.1 * n), squeeze=False)
    for ax, seg in zip(axes[:, 0], segments):
        ref = seg["ref"]
        ax.plot(ref["x"], ref["y"], "--", color="0.6", lw=1, label="reference")
        ax.plot([c[0] for c in seg["cones"]], [c[1] for c in seg["cones"]], "^", color="#ff8c00", ms=5, mec="k", mew=0.5)
        for m, r in seg["runs"].items():
            ax.plot(r["x"], r["y"], color=COLORS[m], lw=1.6, label=f"{render_scurve.NAMES[m]}" + (" (terminated)" if r["terminated"] else (f" ({r['hits'][-1]} cones)" if r["hits"][-1] else "")))
            if r["terminated"]:
                ax.plot(r["x"][-1], r["y"][-1], "x", color=COLORS[m], ms=10, mew=2)
        ax.set_ylim(-1.8, 5.5)
        ax.set_ylabel("y [m]")
        ax.set_title(f"μ = {seg['mu']}, {seg['kmh']:.0f} km/h", fontsize=9, loc="left")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=5, loc="upper right")
    axes[-1, 0].set_xlabel("x [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
