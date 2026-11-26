"""G-AS1 step 1 (VEHICLE_PROPOSAL.md section 11.15): the rough-road, active-suspension Chrono environment.

Checks, in order:
1. actuator signs: pushing the left corners up rolls the body to the right side down (roll_b > 0), pushing the front
   corners up pitches it nose up (pitch_b < 0);
2. preview geometry: the road height Chrono reports under each wheel equals ``Road.height`` at the wheel positions
   computed from the centre-of-mass pose, a, b and the track width;
3. throughput: Chrono steps per second on flat ground, on the rough road, and on the rough road with the actuators;
4. gate: straight driving at 15 m/s over random rough roads, passive suspension vs a corner skyhook controller
   (each actuator damps the body's vertical velocity above its corner). If skyhook does not cut the body's RMS
   vertical acceleration by at least 20 %, the actuators barely matter on this road and the task is not worth
   building.

    SUPERMIMIC_CHRONO_PYTHON=chrono_env/bin/python PYTHONPATH=src python scripts/vehicle/chrono_gas1_env_check.py \\
        --road-dir logs/roads --out logs/chrono_gas1_env_check.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from supermimic.vehicles.chrono_env import ChronoPool
from supermimic.vehicles.road import road_files

CFG = {"terrain": "rigid", "roof_kg": 800.0, "powertrain": "map", "chassis_collision": True, "max_roll": 1.2, "settle": 2.0}
SIDE = np.array([1.0, -1.0, 1.0, -1.0])  # FL FR RL RR: +1 left
AXLE = np.array([1.0, 1.0, -1.0, -1.0])  # +1 front


def wheel_xy(o, geo):
    c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
    dx = np.array([geo["a"], geo["a"], -geo["b"], -geo["b"]])
    dy = SIDE * geo["track"] / 2
    return o["x"] + c * dx - s * dy, o["y"] + s * dx + c * dy


def skyhook(o, gain):
    """Corner skyhook: force against the body's vertical velocity above each corner (heave, roll and pitch rates)."""
    v = o["vz"] + SIDE * 0.9 * o["p"] - AXLE * 1.65 * o["q"]
    return np.clip(-gain * v, -1.0, 1.0)


def preview_ff(o, geo, road, k_ff, t_p=0.06):
    """Road-preview feedforward: retract a corner ahead of a rise in the road relative to the local mean road height
    under that wheel (+-5 m along the heading), cancelling the spring force the bump would create; t_p covers the
    actuator lag."""
    c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
    wx, wy = wheel_xy(o, geo)
    ahead = road.height(wx + c * o["vx"] * t_p, wy + s * o["vx"] * t_p)
    d = np.linspace(-5.0, 5.0, 21)[:, None]
    ref = road.height(wx[None] + c * d, wy[None] + s * d).mean(0)
    return -k_ff * (ahead - ref)


def straight(o):
    return max(-1.0, min(1.0, -1.5 * o["yaw"] - 0.3 * o["y"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--road-dir", default="logs/roads")
    ap.add_argument("--roads", type=int, default=4)
    ap.add_argument("--speed", type=float, default=15.0)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--gains", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--preview-gains", type=float, nargs="*", default=[], help="feedforward gains [1/m] for skyhook (gain 1) + preview")
    ap.add_argument("--rms", type=float, default=0.04, help="road roughness [m]")
    ap.add_argument("--hole-depth", type=float, nargs=2, default=[0.05, 0.10], help="pothole / bump depth range [m]")
    ap.add_argument("--python", default=None)
    ap.add_argument("--out", default="logs/chrono_gas1_env_check.json")
    args = ap.parse_args(argv)
    roads = road_files(range(1000, 1000 + args.roads), args.road_dir, rms=args.rms, hole_depth=tuple(args.hole_depth))
    rep = {}
    n = args.roads
    with ChronoPool(n, args.python) as pool:
        # 1. signs on flat ground
        sign = {}
        for name, f in (("left_up", [1, -1, 1, -1]), ("front_up", [1, 1, -1, -1])):
            pool.reset([{**CFG, "v": 5.0, "active_susp": True}], ids=[0])
            for _ in range(50):
                o = pool.step([0.0], [0.0], ids=[0], f_cmd=[f])[0]
            sign[name] = {"roll_deg": math.degrees(o["roll_b"]), "pitch_deg": math.degrees(o["pitch_b"])}
        rep["signs"] = sign
        print("signs", {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in sign.items()}, flush=True)
        geo = pool.info(ids=[0])[0]
        rep["geometry"] = {k: geo[k] for k in ("a", "b", "track")}

        def run(mode, gain=0.0, flat=False, k_ff=0.0):
            cfgs = [{**CFG, "v": args.speed, "active_susp": mode != "passive", **({} if flat else road.chrono_cfg(p))} for road, p in roads]
            obs = pool.reset(cfgs, ids=range(n))
            t0 = time.perf_counter()
            rows = [[] for _ in range(n)]
            zerr = []
            steps = round(args.seconds / 0.02)
            for _ in range(steps):
                f = [np.clip(skyhook(o, gain) + (preview_ff(o, geo, roads[i][0], k_ff) if k_ff else 0.0), -1, 1).tolist() if mode == "skyhook" else [0.0] * 4
                     for i, o in enumerate(obs)]
                obs = pool.step([straight(o) for o in obs], [0.0] * n, ids=range(n), f_cmd=f)
                for i, o in enumerate(obs):
                    rows[i].append([o.get("az", 0.0), o["p"], o["q"], *[o[f"fz{k}"] for k in range(4)], *[abs(o.get(f"fa{k}", 0.0)) for k in range(4)]])
                    if not flat:
                        wx, wy = wheel_xy(o, geo)
                        zerr.append(np.abs(roads[i][0].height(wx, wy) - np.array([o[f"zr{k}"] for k in range(4)])).max())
            wall = time.perf_counter() - t0
            R = np.array(rows)  # (roads, steps, 11)
            R = R[:, 50:]  # skip the flat lead-in
            fz = R[:, :, 3:7]
            out = {"rms_az": float(np.sqrt((R[:, :, 0] ** 2).mean())), "rms_roll_rate": float(np.sqrt((R[:, :, 1] ** 2).mean())),
                   "rms_pitch_rate": float(np.sqrt((R[:, :, 2] ** 2).mean())),
                   "tyre_load_cv": float((fz.std(1) / fz.mean(1)).mean()), "min_tyre_load_n": float(fz.min()),
                   "mean_abs_force": float(R[:, :, 7:].mean()), "steps_per_s": n * steps / wall}
            if zerr:
                out["max_road_height_error_m"] = float(np.max(zerr))
            return out

        rep["flat_passive"] = run("passive", flat=True)
        print("flat passive", rep["flat_passive"], flush=True)
        runs = [("rough_passive", "passive", 0.0, 0.0), ("rough_active_zero", "zero", 0.0, 0.0)] + [(f"rough_skyhook_{g:g}", "skyhook", g, 0.0) for g in args.gains]
        runs += [(f"rough_skyhook_1_preview_{k:g}", "skyhook", 1.0, k) for k in args.preview_gains]
        for name, mode, g, k in runs:
            rep[name] = run(mode, g, k_ff=k)
            print(name, {k: round(v, 4) for k, v in rep[name].items()}, flush=True)
    base = rep["rough_passive"]["rms_az"]
    best = min(v["rms_az"] for k, v in rep.items() if k.startswith("rough_skyhook"))
    rep["gate"] = {"rms_az_reduction": 1 - best / base, "passed": bool(1 - best / base >= 0.2)}
    print("gate", rep["gate"], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
