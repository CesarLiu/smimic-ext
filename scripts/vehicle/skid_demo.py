"""Understeer, oversteer and spins in our double-track simulator, with scripted drivers (no learning involved).

The tracking environment never shows these: its references stay inside the grip limit and an episode ends as
soon as the car leaves the reference (|lateral| > 2 m, |heading| > 0.6 rad, |sideslip| > 0.35 rad). Here the
dynamics run on their own (``VehicleDynamics``, ABS and ESC off) under three classic manoeuvres:

1. steady-state circle with slowly increasing steering at constant speed (as ISO 4138): a car with the CoG moved
   forward and more roll stiffness at the front saturates its front axle first and runs wide (understeer, plough);
   the same car with the CoG moved back and more roll stiffness at the rear saturates the rear axle and spins
   (oversteer); the nominal car in between;
2. braking in a steady corner: light braking moves load to the front, the unloaded rear lets go and the car spins
   (trail-brake / lift-off oversteer); hard braking locks the front wheels (brakes 66 % front, no ABS), they lose
   their lateral force and the car slides straight on; no braking as the reference;
3. lane change on a wet road (mu 0.5): a small one stays stable; a slightly larger one makes the car swing back past
   the straight line after the second steer (pendulum effect) and slide off sideways when the driver lifts off, or
   spin when the driver keeps the throttle on (rear-wheel drive).

The video shows, per scenario, a top view (paths, cars, wheels coloured by how much of their grip they use, black
when locked, an arrow along the direction of travel: its angle to the body is the sideslip) and, against time, the
sideslip, the yaw rate next to the yaw rate the steering angle asks for (v delta / L), and each axle's used friction
F / (mu Fz). A run ends when the car is nearly stopped.

    PYTHONPATH=src python scripts/vehicle/skid_demo.py --out docs/results/vehicle_skid_demo.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_scurve import car_outline, writer_for

from supermimic.vehicles.dynamics import (
    DELTA,
    VX,
    VY,
    YAW,
    AssistCfg,
    Context,
    R,
    VehicleDynamics,
    X,
    Y,
)

DT = 0.02


def hold_speed(v0, lift=True):
    """Speed controller of the scripted driver; with ``lift`` it lifts off once the car slides (|sideslip| > 15 deg),
    without it keeps asking for speed (rear-wheel drive: the drive force eats into the rear tyres' grip)."""

    def f(t, vx, beta):
        return 0.0 if lift and abs(beta) > 15 else max(-3.0, min(3.0, 2.0 * (v0 - vx)))

    return f


def scenarios():
    ramp = lambda t: min(0.2, 0.03 * t)
    corner = lambda t: 0.045

    def brake(b, v0=20.0):
        hold = hold_speed(v0)
        return lambda t, vx, beta: hold(t, vx, beta) if t < 2.0 else b

    def lane(amp):
        return lambda t: amp * math.sin(2 * math.pi * 0.5 * (t - 0.5)) if 0.5 <= t <= 2.5 else 0.0

    return [
        {"title": "1  稳态回转，方向盘缓慢加大（μ = 0.8，20 m/s）",
         "note": "推头：前轴先饱和，横摆角速度上不去，车沿越来越大的半径冲出；甩尾：后轴饱和，侧偏角迅速增大，整车旋转",
         "mu": 0.8, "v0": 20.0, "T": 6.5, "cars": [
             ("基准车（本身略推头）", "tab:blue", 0.0, 0.0, ramp, hold_speed(20.0)),
             ("推头型：质心前移 0.3 m，前轴侧倾刚度 +20%", "tab:green", 0.3, 0.2, ramp, hold_speed(20.0)),
             ("甩尾型：质心后移 0.45 m，后轴侧倾刚度 +20%", "tab:red", -0.45, -0.2, ramp, hold_speed(20.0))]},
        {"title": "2  弯中制动，ABS 关闭（μ = 0.8，20 m/s，2 s 时开始制动）",
         "note": "轻刹：载荷前移，后轴卸载，车尾甩出；急刹：前轮先抱死（制动力 66% 在前轴），失去侧向力，车不转弯、直着滑出",
         "mu": 0.8, "v0": 20.0, "T": 6.0, "cars": [
             ("不制动", "tab:blue", 0.0, 0.0, corner, hold_speed(20.0)),
             ("轻刹 -2 m/s²", "tab:red", 0.0, 0.0, corner, brake(-2.0)),
             ("急刹 -8 m/s²", "tab:purple", 0.0, 0.0, corner, brake(-8.0))]},
        {"title": "3  湿滑路面换道（μ = 0.5，20 m/s）",
         "note": "幅度稍大时，第二次反打方向后车尾越过中线继续摆动（钟摆效应）；侧滑中继续给油（后驱）会把侧滑变成整车旋转",
         "mu": 0.5, "v0": 20.0, "T": 7.0, "cars": [
             ("幅值 0.04 rad", "tab:blue", 0.0, 0.0, lane(0.04), hold_speed(20.0)),
             ("幅值 0.07 rad，侧滑后松油门", "tab:orange", 0.0, 0.0, lane(0.07), hold_speed(20.0)),
             ("幅值 0.07 rad，侧滑中继续给油", "tab:red", 0.0, 0.0, lane(0.07), hold_speed(20.0, lift=False))]},
    ]


@torch.no_grad()
def simulate(dyn, mu, v0, T, cog, roll_off, steer_fn, accel_fn):
    p = dyn.p
    s = dyn.initial_state(torch.tensor([float(v0)]))
    flags = dyn.new_flags(1)
    ctx = Context(torch.tensor([float(mu)]), torch.ones(1), torch.ones(1), torch.tensor([float(cog)]))
    chassis = torch.tensor([[0.0, 0.0, float(roll_off)]])
    a, b = p.a - cog, p.b + cog
    rows = []
    for k in range(round(T / DT)):
        t = k * DT
        vx, vy = float(s[0, VX]), float(s[0, VY])
        beta = math.degrees(math.atan2(vy, vx))
        if k > 0 and math.hypot(vx, vy) < 1.5:
            break
        rate = max(-p.steer_rate_max, min(p.steer_rate_max, (steer_fn(t) - float(s[0, DELTA])) / 0.05))
        s, flags = dyn.step(s, flags, torch.tensor([rate]), torch.tensor([float(accel_fn(t, vx, beta))]), ctx, chassis)
        ts = dyn.tyre_state(s, ctx, chassis)
        F = torch.sqrt(ts["fx"] ** 2 + ts["fy"] ** 2)[0].numpy()
        fz = ts["fz"][0].numpy()
        vx, vy = float(s[0, VX]), float(s[0, VY])
        rows.append({"t": t + DT, "x": float(s[0, X]), "y": float(s[0, Y]), "yaw": float(s[0, YAW]), "delta": float(s[0, DELTA]),
                     "vx": vx, "vy": vy, "speed": math.hypot(vx, vy), "beta": math.degrees(math.atan2(vy, vx)), "r": float(s[0, R]),
                     "r_kin": vx * float(s[0, DELTA]) / (a + b), "uf": float((F[0] + F[1]) / (fz[0] + fz[1])) / mu,
                     "ur": float((F[2] + F[3]) / (fz[2] + fz[3])) / mu, "uw": (F / fz / mu).tolist(), "kappa": ts["kappa"][0].tolist()})
    return rows


def summarise(rows):
    beta = np.array([r["beta"] for r in rows])
    yaw = np.unwrap([r["yaw"] for r in rows])
    heading = np.unwrap([math.atan2(r["vy"], r["vx"]) + r["yaw"] for r in rows])
    return {"duration_s": rows[-1]["t"], "max_abs_sideslip_deg": float(np.abs(beta).max()), "yaw_change_deg": float(np.degrees(yaw[-1] - yaw[0])),
            "course_change_deg": float(np.degrees(heading[-1] - heading[0])), "spun": bool(np.abs(beta).max() > 90),
            "max_front_use": max(r["uf"] for r in rows), "max_rear_use": max(r["ur"] for r in rows), "final_speed": rows[-1]["speed"]}


def render(results, params, out: Path, fps: int = 25):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation, font_manager, rcParams
    from matplotlib.patches import Polygon

    for name in ("WenQuanYi Zen Hei", "Noto Sans CJK SC", "Noto Sans SC", "SimHei"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            rcParams["font.family"] = name
            break
    rcParams["axes.unicode_minus"] = False
    stride = round(1 / (fps * DT))
    frames = []
    for si, sc in enumerate(results):
        n = max(len(c["rows"]) for c in sc["cars"])
        frames += [(si, k) for k in range(0, n, stride)] + [(si, n - 1)] * fps  # hold the last frame for a second
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.45, 1], left=0.04, right=0.98, top=0.86, bottom=0.07, hspace=0.45, wspace=0.12)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_b, ax_r, ax_u = (fig.add_subplot(gs[i, 1]) for i in range(3))
    cmap = plt.get_cmap("RdYlGn_r")

    def draw(frame):
        si, k = frame
        sc = results[si]
        for ax in (ax_map, ax_b, ax_r, ax_u):
            ax.clear()
        fig.texts.clear()
        fig.text(0.04, 0.955, sc["title"], fontsize=15, ha="left", weight="bold")
        fig.text(0.04, 0.905, sc["note"], fontsize=10.5, ha="left", color="0.25")
        xs = np.concatenate([[r["x"] for r in c["rows"]] for c in sc["cars"]])
        ys = np.concatenate([[r["y"] for r in c["rows"]] for c in sc["cars"]])
        pad = 6
        ax_map.set_xlim(xs.min() - pad, xs.max() + pad)
        ax_map.set_ylim(ys.min() - pad, ys.max() + pad)
        ax_map.set_aspect("equal", adjustable="datalim")
        ax_map.grid(alpha=0.25)
        ax_map.set_xlabel("x [m]")
        ax_map.set_ylabel("y [m]")
        t_now = 0.0
        for c in sc["cars"]:
            rows = c["rows"]
            j = min(k, len(rows) - 1)
            r = rows[j]
            t_now = max(t_now, r["t"])
            ax_map.plot([q["x"] for q in rows], [q["y"] for q in rows], color=c["color"], alpha=0.15, lw=1)
            geo = replace(params, a=params.a - c["cog"], b=params.b + c["cog"])  # the state is the CoG
            for g in range(0, j, round(0.5 / DT)):  # a ghost every 0.5 s: the body's heading along the path
                q = rows[g]
                ax_map.add_patch(Polygon(car_outline(q["x"], q["y"], q["yaw"], q["delta"], geo)[0], closed=True, facecolor="none",
                                         edgecolor=c["color"], alpha=0.35, lw=0.8))
            ax_map.plot([q["x"] for q in rows[: j + 1]], [q["y"] for q in rows[: j + 1]], color=c["color"], lw=2,
                        label=f"{c['label']}   侧偏角 {r['beta']:+.0f}°  车速 {r['speed']:.0f} m/s")
            body, wheels = car_outline(r["x"], r["y"], r["yaw"], r["delta"], geo)
            ax_map.add_patch(Polygon(body, closed=True, facecolor=c["color"], alpha=0.35, edgecolor=c["color"], lw=1.5))
            for w, u, kap in zip(wheels, r["uw"], r["kappa"]):
                col = "black" if kap < -0.6 else cmap(min(1.0, u / 1.1))
                ax_map.add_patch(Polygon(w, closed=True, facecolor=col, edgecolor="0.2", lw=0.5))
            if r["speed"] > 1.0:  # direction of travel
                h = r["yaw"] + math.atan2(r["vy"], r["vx"])
                ax_map.annotate("", xy=(r["x"] + 4 * math.cos(h), r["y"] + 4 * math.sin(h)), xytext=(r["x"], r["y"]),
                                arrowprops={"arrowstyle": "->", "color": "k", "lw": 1.2})
        ax_map.legend(loc="upper left", fontsize=9, framealpha=0.85)
        ax_map.text(0.99, 0.01, "车轮颜色：绿 → 红 = 用掉的附着；黑 = 抱死。箭头：行驶方向。细框：每 0.5 s 的车身位置", transform=ax_map.transAxes,
                    ha="right", va="bottom", fontsize=8.5, color="0.3")
        T = max(c["rows"][-1]["t"] for c in sc["cars"])
        for c in sc["cars"]:
            rows = c["rows"][: min(k, len(c["rows"]) - 1) + 1]
            t = [q["t"] for q in rows]
            ax_b.plot(t, np.degrees(np.unwrap(np.radians([q["beta"] for q in rows]))), color=c["color"])
            ax_r.plot(t, [q["r"] for q in rows], color=c["color"])
            ax_r.plot(t, [q["r_kin"] for q in rows], color=c["color"], ls=":", lw=1)
            ax_u.plot(t, [q["uf"] for q in rows], color=c["color"])
            ax_u.plot(t, [q["ur"] for q in rows], color=c["color"], ls="--")
        ax_b.set_ylabel("侧偏角 [°]")
        lim = max(30.0, max(np.abs(np.degrees(np.unwrap(np.radians([q["beta"] for q in c["rows"]])))).max() for c in sc["cars"]) + 15)
        ax_b.set_ylim(-lim, lim)
        ax_r.set_ylabel("横摆角速度 [rad/s]")
        ax_r.text(0.01, 0.95, "实线：实际；点线：按转角应有的 v·δ/L", transform=ax_r.transAxes, fontsize=8, va="top")
        ax_u.set_ylabel("轴的附着利用 F/(μFz)")
        ax_u.set_ylim(0, 1.3)
        ax_u.text(0.01, 0.95, "实线：前轴；虚线：后轴", transform=ax_u.transAxes, fontsize=8, va="top")
        for ax in (ax_b, ax_r, ax_u):
            ax.set_xlim(0, T)
            ax.axvline(t_now, color="0.5", lw=0.8)
            ax.grid(alpha=0.3)
        ax_u.set_xlabel("时间 [s]")

    out, writer = writer_for(out, fps)
    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    anim.save(str(out), writer=writer)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="logs/skid_demo.mp4")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args(argv)
    dyn = VehicleDynamics(dt=DT, substeps=10, assist=AssistCfg(abs=False, esc=False))
    results, summary = [], {}
    for sc in scenarios():
        cars = []
        for label, color, cog, roll_off, steer_fn, accel_fn in sc["cars"]:
            rows = simulate(dyn, sc["mu"], sc["v0"], sc["T"], cog, roll_off, steer_fn, accel_fn)
            cars.append({"label": label, "color": color, "cog": cog, "rows": rows})
            summary.setdefault(sc["title"], {})[label] = summarise(rows)
        results.append({**sc, "cars": cars})
    for title, cars in summary.items():
        print(title)
        for label, s in cars.items():
            print(f"  {label:<36} max |sideslip| {s['max_abs_sideslip_deg']:6.1f} deg  yaw {s['yaw_change_deg']:7.1f} deg  course {s['course_change_deg']:6.1f} deg"
                  f"  max use front {s['max_front_use']:.2f} rear {s['max_rear_use']:.2f}  {'SPUN' if s['spun'] else ''}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    if not args.no_video:
        print(f"video: {render(results, dyn.p, out, args.fps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
