"""Render the Chrono rollover runs written by ``chrono_rollover.py`` (VEHICLE_PROPOSAL.md section 11.11).

Per scenario, three 3D views (one per run) of the chassis (a box standing in for the HMMWV's body, schematic), the
roof load, the wheels (red while the tyre carries less than 100 N: lifted) and, in the trip scenario, the curb; under
them the chassis roll angle against time, thick where two wheels of one side are off the ground.

    PYTHONPATH=src python scripts/vehicle/render_rollover.py --data logs/rollover.json --out logs/rollover.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_scurve import writer_for

BODY = ((-2.4, 2.4), (-1.05, 1.05), (-0.15, 1.3))  # schematic body box in the chassis reference frame [m]
WHEEL_R, WHEEL_W = 0.47, 0.32
COLORS = ["tab:blue", "tab:orange", "tab:red"]
NOTES = {
    "fishhook": ("鱼钩测试：左打方向，再反打到右满舵（硬地面 μ 0.9，Chrono HMMWV）",
                 "原车 SSF ≈ 1.2 > μ，只会侧滑；车顶载重 800 kg 把质心抬到约 1.1 m（SSF ≈ 0.8 < μ）：低速两轮离地后落回，高速翻车"),
    "trip": ("绊倒式翻车：在冰面（μ 0.15）上横向侧滑",
             "没有障碍时一直滑；撞上 20 cm 路肩：4 m/s 被挡住，8 m/s 车轮被绊住，车身绕路肩翻滚"),
}


def box_faces(center, R, ranges):
    (x0, x1), (y0, y1), (z0, z1) = ranges
    c = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    w = center + c @ R.T
    idx = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    return [w[list(f)] for f in idx]


def wheel_faces(center, R, n=14):
    """A cylinder along the spindle's y axis."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([WHEEL_R * np.cos(ang), np.zeros(n), WHEEL_R * np.sin(ang)], -1)
    a = center + (ring + [0, WHEEL_W / 2, 0]) @ R.T
    b = center + (ring - [0, WHEEL_W / 2, 0]) @ R.T
    faces = [a, b]
    for i in range(n):
        j = (i + 1) % n
        faces.append(np.array([a[i], a[j], b[j], b[i]]))
    return faces


def rot(axes):
    return np.array(axes).T  # columns: the body's x, y, z axes in the world


def render(data, out: Path, fps: int = 25):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation, font_manager, rcParams
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for name in ("WenQuanYi Zen Hei", "Noto Sans CJK SC", "Noto Sans SC", "SimHei"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            rcParams["font.family"] = name
            break
    rcParams["axes.unicode_minus"] = False
    roof_dims = data["roof_box"]
    scen = data["scenarios"]
    stride = round(1 / (fps * 0.02))
    frames = []
    for name, sc in scen.items():
        n = max(len(r["rows"]) for r in sc["runs"])
        frames += [(name, k) for k in range(0, n, stride)] + [(name, n - 1)] * fps
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1], left=0.03, right=0.98, top=0.80, bottom=0.08, hspace=0.12, wspace=0.02)
    ax3 = [fig.add_subplot(gs[0, i], projection="3d") for i in range(3)]
    axr = fig.add_subplot(gs[1, :])

    def draw(frame):
        name, k = frame
        sc = scen[name]
        fig.texts.clear()
        fig.text(0.03, 0.955, NOTES[name][0], fontsize=15, weight="bold")
        fig.text(0.03, 0.905, NOTES[name][1], fontsize=10.5, color="0.25")
        for i, (ax, run) in enumerate(zip(ax3, sc["runs"])):
            ax.clear()
            rows = run["rows"]
            r = rows[min(k, len(rows) - 1)]
            c = np.array(r["pos"])
            ax.set_xlim(c[0] - 4, c[0] + 4)
            ax.set_ylim(c[1] - 4, c[1] + 4)
            ax.set_zlim(0, 4)
            ax.set_box_aspect((1, 1, 0.5), zoom=1.35)
            ax.set_axis_off()
            if name == "fishhook":
                ax.view_init(elev=22, azim=-135)
            else:
                ax.view_init(elev=24, azim=-65)
            # ground grid
            g0x, g0y = math.ceil(c[0] - 4), math.ceil(c[1] - 4)
            for gx in range(g0x, g0x + 8):
                ax.plot([gx, gx], [c[1] - 4, c[1] + 4], [0, 0], color="0.85", lw=0.6)
            for gy in range(g0y, g0y + 8):
                ax.plot([c[0] - 4, c[0] + 4], [gy, gy], [0, 0], color="0.85", lw=0.6)
            trail = np.array([q["pos"] for q in rows[: min(k, len(rows) - 1) + 1]])
            trail = trail[(np.abs(trail[:, 0] - c[0]) < 4) & (np.abs(trail[:, 1] - c[1]) < 4)]
            ax.plot(trail[:, 0], trail[:, 1], np.zeros(len(trail)) + 0.01, color=COLORS[i], lw=1.5, alpha=0.7)
            if name == "trip" and run["curb"] > 0:
                cx = sc["curb_x"]
                ax.add_collection3d(Poly3DCollection(box_faces(np.zeros(3), np.eye(3), ((cx - 0.2, cx + 0.2), (c[1] - 4, c[1] + 4), (0, run["curb"]))),
                                                     facecolor="0.55", edgecolor="0.3", lw=0.5))
            R = rot(r["axes"])
            ax.add_collection3d(Poly3DCollection(box_faces(c, R, BODY), facecolor=COLORS[i], edgecolor="0.2", lw=0.6, alpha=0.55))
            if r["roof"] is not None:
                lx, ly, lz, _ = roof_dims
                ax.add_collection3d(Poly3DCollection(box_faces(np.array(r["roof"]["pos"]), rot(r["roof"]["axes"]),
                                                               ((-lx / 2, lx / 2), (-ly / 2, ly / 2), (-lz / 2, lz / 2))),
                                                     facecolor="saddlebrown", edgecolor="0.2", lw=0.6, alpha=0.8))
            for sp, sa, fz in zip(r["spindles"], r["spindle_axes"], r["fz"]):
                ax.add_collection3d(Poly3DCollection(wheel_faces(np.array(sp), rot(sa)), facecolor="crimson" if fz < 100 else "0.25",
                                                     edgecolor="0.1", lw=0.3))
            ax.set_title(f"{run['label']}\n倾斜 {r['tilt_deg']:.0f}°   车速 {r['speed']:.1f} m/s", fontsize=10.5, color=COLORS[i], pad=0)
        axr.clear()
        for i, run in enumerate(sc["runs"]):
            rows = run["rows"][: min(k, len(run["rows"]) - 1) + 1]
            t = np.array([q["t"] for q in rows])
            roll = np.degrees(np.unwrap(np.radians([q["roll_deg"] for q in rows])))
            lift = np.array([(q["fz"][0] < 100 and q["fz"][2] < 100) or (q["fz"][1] < 100 and q["fz"][3] < 100) for q in rows])
            axr.plot(t, roll, color=COLORS[i], lw=1.2, label=run["label"])
            axr.plot(t, np.where(lift, roll, np.nan), color=COLORS[i], lw=4, alpha=0.6)
        T = max(run["rows"][-1]["t"] for run in sc["runs"])
        axr.set_xlim(0, T)
        axr.set_ylim(-200, 200)
        axr.axhline(90, color="0.5", ls=":", lw=0.8)
        axr.axhline(-90, color="0.5", ls=":", lw=0.8)
        axr.set_ylabel("车身侧倾角 [°]")
        axr.set_xlabel("时间 [s]" + ("（从开始打方向算起）" if name == "fishhook" else "（从开始侧滑算起）"))
        axr.grid(alpha=0.3)
        axr.legend(loc="upper left", fontsize=8.5, ncol=3)
        axr.text(0.995, 0.04, "粗线：同侧两个车轮都离地（垂向力 < 100 N）；红色车轮：离地；±90°：侧翻", transform=axr.transAxes, ha="right", fontsize=8.5, color="0.3")

    out, writer = writer_for(out, fps)
    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    anim.save(str(out), writer=writer)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="logs/rollover.mp4")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args(argv)
    data = json.loads(Path(args.data).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"video: {render(data, out, args.fps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
