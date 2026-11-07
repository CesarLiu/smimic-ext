"""Render an evaluation video of the trained policies on the S-bend (from ``compare_ppo.py``'s ``policies.pt``).

    PYTHONPATH=src python scripts/vehicle/render_scurve.py --policies logs/compare/seed0/policies.pt --out logs/compare/seed0/s_curve.mp4

The NMPC baselines drive too when the run included them, or with ``--nmpc`` (true state and friction) and
``--nmpc-onboard`` (the policies' information; needs ``casadi``) for runs made without them.

Each policy drives the S-bend deterministically (nominal context, no noise) at every friction level of the run,
one after the other in the video. Left: a camera following the cars (body outline, steered front wheels,
trail, reference line). Right: the whole bend, the lateral error and the friction use sqrt(ax^2 + ay^2) / (mu g)
against time, with the reference's friction use for comparison. Writes MP4 through ffmpeg when it is available
(system ``ffmpeg`` or the ``imageio-ffmpeg`` package), otherwise an animated GIF.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_ppo import COLORS, networks
from demo_e2e import ObsHistory
from limit_study import env_cfg, eval_xi, measurement

from supermimic.vehicles import tracking
from supermimic.vehicles.dynamics import AX, AY, DELTA, VX, VY, YAW, X, Y
from supermimic.vehicles.env import ReferenceLibrary, VehicleTrackingEnv
from supermimic.vehicles.learner import VehicleHeads
from supermimic.vehicles.threads import add_threads_arg, limit_threads

NAMES = {"supermimic": "SuperMimic", "ppo": "PPO", "nmpc": "NMPC (true state)", "nmpc_onboard": "NMPC (on-board)",
         "supermimic_chassis": "SuperMimic + chassis", "ppo_chassis": "PPO + chassis"}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", required=True, help="policies.pt written by compare_ppo.py")
    ap.add_argument("--out", default=None, help="video path (.mp4 or .gif); default: next to policies.pt")
    ap.add_argument("--mu", type=float, nargs="*", default=None, help="friction levels to show (default: those of the run)")
    ap.add_argument("--fps", type=int, default=25, help="video frame rate; the simulation runs at 50 Hz, shown in real time")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--view", type=float, default=36.0, help="width of the following camera's view [m]")
    ap.add_argument("--nmpc", action="store_true", help="add NMPC with the true state and friction even if the run did not include it")
    ap.add_argument("--nmpc-onboard", action="store_true", help="add NMPC with the policies' information (estimator, delay compensation)")
    ap.add_argument("--nmpc-assumed-delay", type=int, default=1, help="with --nmpc-onboard: the delay it compensates [control steps]")
    ap.add_argument("--nmpc-friction", default="known", choices=["known", "nominal"], help="with --nmpc: whether NMPC knows the road friction")
    ap.add_argument("--device", default="cpu")
    add_threads_arg(ap)
    return ap.parse_args(argv)


@torch.no_grad()
def drive(policy, heads, lib: ReferenceLibrary, args, controller=None) -> dict:
    """Deterministic play from the reference's start; per-step pose, steering, errors and friction use.
    ``controller(env, measurements) -> actions`` replaces the policy when given."""
    length = int(lib.lengths[0])
    env = VehicleTrackingEnv(env_cfg(args, 1, 2, train=False), lib, device=args.device, record=False)
    env.forced_xi = eval_xi(args)
    env.forced_frame[:] = lib.layout.seq_start[:1]
    obs = env.reset()
    meas = measurement(obs["policy"], heads, args)
    hist = ObsHistory(heads, 1, args.history, args.device)
    hist.reset(torch.arange(1, device=args.device), meas)
    mu = float(env.ctx.mu[0])
    keys = ("x", "y", "yaw", "delta", "delta_r", "speed", "lat", "util")
    out = {k: [] for k in keys}

    def record(ref):
        s = env.s[0]
        e = tracking.errors(env.s, ref)
        dr = env.act.chassis[0, 0] if env.act.chassis is not None else torch.zeros(())
        for k, v in (("x", s[X]), ("y", s[Y]), ("yaw", s[YAW]), ("delta", s[DELTA]), ("delta_r", dr), ("speed", torch.hypot(s[VX], s[VY])),
                     ("lat", e["lat"][0]), ("util", torch.hypot(s[AX], s[AY]) / (mu * 9.81))):
            out[k].append(float(v))

    record(lib.ref(env.frame.clone()))
    out["terminated"] = False
    for _ in range(length - 1):
        ref = lib.ref(env.frame.clone())
        obs, _r, term, _trunc, info = env.step(policy(hist.flat(), obs["goal"]) if controller is None else controller(env, meas))
        if info["done"][0]:
            out["terminated"] = bool(term[0])
            break
        record(ref)
        meas = measurement(obs["policy"], heads, args)
        hist.push(meas)
    return out


def car_outline(x, y, yaw, delta, p, delta_r=0.0):
    """Body and wheel polygons (world coordinates) of a car at pose (x, y, yaw) with front-wheel angle delta."""
    c, s = math.cos(yaw), math.sin(yaw)
    to_world = lambda pts: [(x + c * u - s * v, y + s * u + c * v) for u, v in pts]  # noqa: E731
    front, rear, half_w = p.a + 0.85, p.b + 0.9, 0.88
    body = to_world([(front, -half_w), (front, half_w), (-rear, half_w), (-rear, -half_w)])
    wl, ww = 0.33, 0.11

    def wheel(u0, v0, ang):
        ca, sa = math.cos(ang), math.sin(ang)
        return to_world([(u0 + ca * du - sa * dv, v0 + sa * du + ca * dv) for du, dv in ((wl, -ww), (wl, ww), (-wl, ww), (-wl, -ww))])

    wheels = [wheel(p.a, -p.T_f / 2, delta), wheel(p.a, p.T_f / 2, delta), wheel(-p.b, -p.T_r / 2, delta_r), wheel(-p.b, p.T_r / 2, delta_r)]
    return body, wheels


def writer_for(path: Path, fps: int):
    from matplotlib import animation, rcParams

    if path.suffix.lower() == ".mp4":
        exe = shutil.which("ffmpeg")
        if exe is None:
            try:
                import imageio_ffmpeg

                exe = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                exe = None
        if exe is not None:
            rcParams["animation.ffmpeg_path"] = exe
            return path, animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=3000, extra_args=["-pix_fmt", "yuv420p"])
        path = path.with_suffix(".gif")
        print(f"ffmpeg not found; writing {path} instead")
    return path, animation.PillowWriter(fps=fps)


def render(segments, params, args, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    stride = max(1, round(50 / args.fps))  # simulation steps per video frame (real time)
    frames = [(j, k) for j, seg in enumerate(segments) for k in range(0, seg["length"], stride)]
    fig = plt.figure(figsize=(12.8, 7.2), dpi=args.dpi)
    elk = segments[0].get("layout") == "elk"  # wide camera on top, no whole-track panel (the track is long and 1 m wide)
    if elk:
        gs = fig.add_gridspec(2, 2, height_ratios=(1.35, 1.0), left=0.06, right=0.98, top=0.9, bottom=0.08, hspace=0.32, wspace=0.18)
        ax_cam, ax_all, ax_lat, ax_util = fig.add_subplot(gs[0, :]), None, fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
        bb = ax_cam.get_position()
        y_half = args.view / 2 * (bb.height * 7.2) / (bb.width * 12.8)  # equal scaling without shrinking the axes
    else:
        gs = fig.add_gridspec(3, 2, width_ratios=(1.55, 1.0), height_ratios=(1.1, 1.0, 1.0), left=0.05, right=0.98, top=0.9, bottom=0.07, hspace=0.42, wspace=0.16)
        ax_cam, ax_all, ax_lat, ax_util = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 1])
    title = fig.suptitle("", fontsize=13)
    methods = list(segments[0]["runs"])

    def draw(frame):
        j, k = frames[frame]
        seg = segments[j]
        ref, runs, t = seg["ref"], seg["runs"], k * 0.02
        if "title" in seg:
            title.set_text(f"{seg['title']}   t = {t:4.1f} s")
        else:
            title.set_text(f"S-bend at μ = {seg['mu']}   ({seg['speed']:.0f} m/s, arcs at {seg['ay']:.1f} μg; policies trained on random curves, never on this bend)   t = {t:4.1f} s")
        for a in (ax_cam, ax_all, ax_lat, ax_util):
            if a is not None:
                a.clear()
        # following camera
        alive = {m: min(k, len(r["x"]) - 1) for m, r in runs.items()}
        cx = sum(runs[m]["x"][alive[m]] for m in methods) / len(methods)
        cy = sum(runs[m]["y"][alive[m]] for m in methods) / len(methods)
        h = args.view * 0.95
        ax_cam.plot(ref["x"], ref["y"], "--", color="0.55", lw=1.5, zorder=1, label="reference")
        if seg.get("cones"):
            ax_cam.plot([c[0] for c in seg["cones"]], [c[1] for c in seg["cones"]], "^", color="#ff8c00", ms=7, mec="k", mew=0.6, zorder=2, label="cones")
        for m in methods:
            r, i = runs[m], alive[m]
            ax_cam.plot(r["x"][: i + 1], r["y"][: i + 1], color=COLORS[m], lw=2, alpha=0.8, zorder=2)
            body, wheels = car_outline(r["x"][i], r["y"][i], r["yaw"][i], r["delta"][i], params, r.get("delta_r", [0.0] * (i + 1))[i])
            ax_cam.add_patch(Polygon(body, closed=True, fc=COLORS[m], ec="k", lw=1, alpha=0.45, zorder=3))
            for w in wheels:
                ax_cam.add_patch(Polygon(w, closed=True, fc="k", ec="k", zorder=4))
            ended = r["terminated"] and k >= len(r["x"]) - 1
            state = "TERMINATED" if ended else f"{r['speed'][i]:4.1f} m/s   lateral error {r['lat'][i]:+5.2f} m   friction use {r['util'][i]:4.2f}"
            if "hits" in r and not ended:
                state += "   cones: " + (f"{r['hits'][i]} hit" if r["hits"][i] > 0 else "clear")
            ax_cam.plot([], [], color=COLORS[m], lw=3, label=f"{NAMES[m]}: {state}")
            if ended:
                ax_cam.plot(r["x"][i], r["y"][i], "x", color=COLORS[m], ms=16, mew=3, zorder=5)
        ax_cam.set_xlim(cx - args.view / 2, cx + args.view / 2)
        if elk:
            ax_cam.set_ylim(seg["y_mid"] - y_half, seg["y_mid"] + y_half)
        else:
            ax_cam.set_ylim(cy - h / 2, cy + h / 2)
            ax_cam.set_aspect("equal")
        ax_cam.set_xlabel("x [m]")
        ax_cam.set_ylabel("y [m]")
        ax_cam.grid(alpha=0.25)
        ax_cam.legend(loc="lower left" if elk else "upper left", fontsize=9, framealpha=0.9)
        # whole bend
        if ax_all is not None:
            ax_all.plot(ref["x"], ref["y"], "--", color="0.55", lw=1.2)
            for m in methods:
                r, i = runs[m], alive[m]
                ax_all.plot(r["x"][: i + 1], r["y"][: i + 1], color=COLORS[m], lw=1.5)
                ax_all.plot(r["x"][i], r["y"][i], "o", color=COLORS[m], ms=5)
            ax_all.set_aspect("equal", adjustable="datalim")
            ax_all.set_title("whole bend", fontsize=10)
            ax_all.tick_params(labelsize=8)
        # time series
        T = seg["length"] * 0.02
        for a, key, lab in ((ax_lat, "lat", "lateral error [m]"), (ax_util, "util", "friction use")):
            if key == "util":
                tr = [n * 0.02 for n in range(len(ref["util"]))]
                a.plot(tr, ref["util"], "--", color="0.55", lw=1.2, label="reference")
                a.axhline(1.0, color="k", lw=0.8, alpha=0.5)
            else:
                a.axhline(0.0, color="k", lw=0.8, alpha=0.5)
            for m in methods:
                r, i = runs[m], alive[m]
                a.plot([n * 0.02 for n in range(i + 1)], r[key][: i + 1], color=COLORS[m], lw=1.6, label=NAMES[m])
            a.axvline(t, color="0.3", lw=0.8)
            a.set_xlim(0, T)
            a.set_ylabel(lab, fontsize=9)
            a.tick_params(labelsize=8)
            a.grid(alpha=0.25)
        ax_lat.set_ylim(*seg["lat_lim"])
        ax_util.set_ylim(0, 1.25)
        ax_util.set_xlabel("t [s]", fontsize=9)
        ax_util.legend(fontsize=8, loc="lower right", ncol=3)
        return []

    from matplotlib import animation

    out, writer = writer_for(out, args.fps)
    anim = animation.FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save(str(out), writer=writer, dpi=args.dpi)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    args = parse(argv)
    limit_threads(args.threads)
    blob = torch.load(args.policies, map_location="cpu")
    run = SimpleNamespace(**{**blob["args"], "device": args.device})
    levels = args.mu if args.mu else run.mu_levels
    cfg = env_cfg(run, 2, 2)
    controllers = dict(blob.get("controllers", {}))
    if args.nmpc and "nmpc" not in controllers:
        controllers["nmpc"] = {"friction": args.nmpc_friction}
    if args.nmpc_onboard and "nmpc_onboard" not in controllers:
        controllers["nmpc_onboard"] = {"information": "onboard", "friction": args.nmpc_friction, "assumed_delay": args.nmpc_assumed_delay}
    nmpcs = {}
    if controllers:
        from supermimic.baselines.nmpc import NMPC, NMPCCfg

        nmpcs = {name: NMPC(cfg, cfg=NMPCCfg(**c)) for name, c in controllers.items()}
    segments = []
    for mu in levels:
        lib = ReferenceLibrary.s_curve(cfg, speed=run.scurve_speed, ay_frac=run.scurve_ay, mu=mu).to(args.device)
        env = VehicleTrackingEnv(cfg, lib, device=args.device, record=False)
        runs = {}
        for m, sd in blob["policies"].items():
            heads, _task, _spec, _io, _pcfg, policy, _critic = networks(run, env, lib)
            policy.load_state_dict(sd)
            policy.eval()
            runs[m] = drive(policy, heads, lib, run)
        for name, ctl in nmpcs.items():
            ctl.reset()
            runs[name] = drive(None, VehicleHeads(cfg, env.dyn.p.R_w), lib, run, controller=ctl)
        for m in runs:
            print(f"mu {mu}: {m} {'terminated' if runs[m]['terminated'] else 'completed'} after {len(runs[m]['x'])} steps, "
                  f"mean |lateral error| {sum(abs(v) for v in runs[m]['lat']) / len(runs[m]['lat']):.2f} m", flush=True)
        ref = {"x": lib.x.cpu().tolist(), "y": lib.y.cpu().tolist(), "util": lib.util.cpu().tolist()}
        lat_max = max(max(abs(v) for v in r["lat"]) for r in runs.values())
        segments.append({"mu": mu, "speed": run.scurve_speed, "ay": run.scurve_ay, "ref": ref, "runs": runs, "length": int(lib.lengths[0]),
                         "lat_lim": (-max(0.5, 1.15 * lat_max), max(0.5, 1.15 * lat_max))})
    out = Path(args.out) if args.out else Path(args.policies).with_name("s_curve.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    out = render(segments, env.dyn.p, args, out)
    print(f"video: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
