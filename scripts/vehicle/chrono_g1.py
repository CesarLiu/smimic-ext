"""Gate G1 of VEHICLE_PROPOSAL.md section 11.4: how much does single-track NMPC lose on deformable soil?

For each terrain (Chrono HMMWV on rigid ground with TMeasy tyres; on SCM soils with rigid tyres):

1. identification: random smooth steering and speed sweeps; the steering ratio (wheel angle per unit command) by
   least squares, and each axle's lateral force curve from single-track inverse dynamics (``chrono_env.axle_forces``),
   fitted with the NMPC's own magic-formula form (``nmpc.fit_axle``). This is the best single-track model of that
   terrain: the NMPC gets the curves identified on the terrain it drives on;
2. tracking: an S-bend and random smooth paths at constant speed, with
   * ``nmpc``: ``baselines.nmpc.NMPC`` (privileged state: Chrono's true pose, velocities, yaw rate and wheel angle)
     with the vehicle's mass, yaw inertia and axle positions from Chrono and the identified tyres;
   * ``nmpc_rigid_model`` (soils only): the same, but with the tyres identified on rigid ground (the nominal model);
   * ``pure_pursuit``: a geometric controller without any vehicle model, as a yardstick.

Gate: on soil, NMPC's mean lateral error is at least 3x its rigid-ground value on the same paths, or it terminates.

    SUPERMIMIC_CHRONO_PYTHON=chrono_env/bin/python PYTHONPATH=src python scripts/vehicle/chrono_g1.py \\
        --out docs/results/vehicle_chrono_g1.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from supermimic.baselines.nmpc import NMPC, AxleTyre, NMPCCfg, fit_axle
from supermimic.vehicles.chrono_env import (
    ChronoPool,
    axle_forces,
    path_errors,
    pure_pursuit,
    random_path,
    s_bend,
)
from supermimic.vehicles.chrono_worker import SOILS
from supermimic.vehicles.env import VehicleEnvCfg
from supermimic.vehicles.params import VehicleParams

TERRAINS = {
    "rigid": {"terrain": "rigid", "friction": 0.9},
    "soft": {"terrain": "scm", "soil": "soft", "grid": 0.10},
    "firm": {"terrain": "scm", "soil": "firm", "grid": 0.10},
}
KEYS = ("t", "x", "y", "yaw", "vx", "vy", "r", "ax", "ay", "rdot", "delta", "s_act", "throttle", "brake", "sink", "roll")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terrains", nargs="+", default=list(TERRAINS), choices=list(TERRAINS))
    ap.add_argument("--speed", type=float, default=4.0, help="reference speed [m/s] (soft soil allows about 4.4 at full throttle)")
    ap.add_argument("--ay", type=float, default=1.6, help="peak lateral acceleration of the paths [m/s^2]")
    ap.add_argument("--random-paths", type=int, default=4)
    ap.add_argument("--path-length", type=float, default=100.0)
    ap.add_argument("--id-envs", type=int, default=4)
    ap.add_argument("--id-seconds", type=float, default=25.0)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--interval", type=int, default=3, help="control steps per NMPC interval (25 x 3 x 0.02 = 1.5 s)")
    ap.add_argument("--max-lat", type=float, default=2.5, help="terminate beyond this lateral error [m]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--python", default=None, help="PyChrono interpreter (default: $SUPERMIMIC_CHRONO_PYTHON)")
    ap.add_argument("--out", default="logs/chrono_g1.json")
    ap.add_argument("--plot", default=None, help="PNG of the S-bend tracks per terrain")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------- identification


def identify(args, terrain: str, log=print) -> dict:
    rng = np.random.default_rng(args.seed + 17)
    n = args.id_envs
    with ChronoPool(n, args.python) as pool:
        obs = pool.reset([{**TERRAINS[terrain], "v": args.speed} for _ in range(n)])  # one Chrono system per process
        veh = pool.info()[0]
        amp = rng.uniform(0.3, 0.8, (n, 2))
        frq = rng.uniform(0.08, 0.35, (n, 2))
        ph = rng.uniform(0, 2 * math.pi, (n, 2))
        vph = rng.uniform(0, 2 * math.pi, n)
        rows = []
        steps = round(args.id_seconds / 0.02)
        t0 = time.perf_counter()
        for k in range(steps):
            t = k * 0.02
            s = 0.5 * (amp[:, 0] * np.sin(2 * np.pi * frq[:, 0] * t + ph[:, 0]) + amp[:, 1] * np.sin(2 * np.pi * frq[:, 1] * t + ph[:, 1]))
            # near-constant speed at the tracking speed: at lower speeds the slip angles shrink toward the
            # measurement noise and least squares underestimates the cornering stiffness (errors in variables)
            v_tgt = args.speed * (0.95 + 0.05 * np.sin(2 * np.pi * t / 8 + vph))
            a = np.clip(1.0 * (v_tgt - np.array([o["vx"] for o in obs])), -2, 2)
            obs = pool.step(s, a)
            if t >= 1.0:
                rows.append([[o[kk] for kk in KEYS] + [o["ok"]] for o in obs])
        wall = time.perf_counter() - t0
    d = np.array(rows, dtype=float)  # (time, env, key)
    L = {kk: d[:, :, i] for i, kk in enumerate(KEYS)}
    # Chrono's instantaneous chassis accelerations are dominated by contact vibration (the angular one is
    # uncorrelated with the yaw-rate trend); the single-track model needs the control-rate trend: central differences
    L["rdot"] = np.gradient(L["r"], 0.02, axis=0)
    L["ay"] = np.gradient(L["vy"], 0.02, axis=0) + L["r"] * L["vx"]
    ok = (d[:, :, -1] > 0) & (L["vx"] > 0.6 * args.speed) & (np.abs(L["vy"] / np.maximum(L["vx"], 1.0)) < 0.3)
    L = {kk: v[ok] for kk, v in L.items()}
    k_s = float(np.sum(L["delta"] * L["s_act"]) / np.sum(L["s_act"] ** 2))
    f = axle_forces(L, veh["mass"], veh["I_z"], veh["a"], veh["b"])
    # The excitation stays in the linear range (|alpha| < ~0.05 at these speeds; the steering limit prevents more on
    # rigid ground), so a free magic-formula fit cannot locate the peak and collapses onto a tiny D at its bounds.
    # Fit the cornering stiffness from the data and take the peak from a physical prior: axle load times the
    # friction coefficient (rigid: the contact material's; soil: tan of the Mohr-Coulomb friction angle).
    mu = prior_friction(terrain)
    L_wb = veh["a"] + veh["b"]
    loads = [veh["mass"] * 9.81 * veh["b"] / L_wb, veh["mass"] * 9.81 * veh["a"] / L_wb]
    front, rear = (linear_with_peak(f["alpha_" + ax], f["fy" + ax], mu * fz) for ax, fz in zip("fr", loads))
    free = {"front": fit_axle(f["alpha_f"], f["fyf"]).__dict__, "rear": fit_axle(f["alpha_r"], f["fyr"]).__dict__}
    rms = lambda tyre, al, fy: float(np.sqrt(np.mean((tyre.force(al) - fy) ** 2)) / np.abs(fy).max())
    out = {"vehicle": veh, "k_s": k_s, "steer_ratio_rel_rms": float(np.sqrt(np.mean((L["delta"] - k_s * L["s_act"]) ** 2)) / np.abs(L["delta"]).max()),
           "front": front.__dict__, "rear": rear.__dict__, "prior_friction": mu, "axle_loads": loads,
           "free_magic_formula_fit": free,
           "cornering_stiffness": [front.B * front.C * front.D, rear.B * rear.C * rear.D],
           "rel_rms_front": rms(front, f["alpha_f"], f["fyf"]), "rel_rms_rear": rms(rear, f["alpha_r"], f["fyr"]),
           "samples": int(ok.sum()), "max_abs_alpha": [float(np.abs(f["alpha_f"]).max()), float(np.abs(f["alpha_r"]).max())],
           "max_abs_ay": float(np.abs(L["ay"]).max()), "mean_speed": float(L["vx"].mean()), "mean_sink": float(L["sink"].mean()),
           "wall_s": round(wall, 1)}
    log(f"[{terrain}] identified from {out['samples']} samples ({wall:.0f} s): steering {k_s:.3f} rad/unit, cornering stiffness "
        f"front {out['cornering_stiffness'][0]:.0f} rear {out['cornering_stiffness'][1]:.0f} N/rad, fit rel RMS "
        f"{out['rel_rms_front']:.3f} / {out['rel_rms_rear']:.3f}, mean speed {out['mean_speed']:.2f} m/s")
    return out


def prior_friction(terrain: str) -> float:
    c = TERRAINS[terrain]
    if c["terrain"] == "rigid":
        return c["friction"]
    return math.tan(math.radians(SOILS[c["soil"]][4]))


def linear_with_peak(alpha: np.ndarray, force: np.ndarray, peak: float, C: float = 1.3) -> AxleTyre:
    """Magic formula with E = 0 and shape C, the cornering stiffness K = B C D fitted through the origin on
    |alpha| < 0.06, and the peak |D| given."""
    sel = np.abs(alpha) < 0.06
    K = float(np.sum(alpha[sel] * force[sel]) / np.sum(alpha[sel] ** 2))
    D = math.copysign(peak, K)
    return AxleTyre(B=K / (C * D), C=C, D=D, E=0.0)


def make_nmpc(ident: dict, args) -> NMPC:
    veh = ident["vehicle"]
    k_s = ident["k_s"]
    p = replace(VehicleParams(), m=veh["mass"], I_z=veh["I_z"], a=veh["a"], b=veh["b"], steer_max=k_s,
                steer_rate_max=1.2 * k_s, a_max=3.0, v_switch=1e3)
    ec = replace(VehicleEnvCfg(num_envs=1), dt=0.02, steer_scale=k_s, accel_scale=3.0, steer_tau=0.05, accel_tau=0.5)
    return NMPC(ec, p, NMPCCfg(horizon=args.horizon, interval=args.interval), tyres=(AxleTyre(**ident["front"]), AxleTyre(**ident["rear"])))


# ---------------------------------------------------------------------- tracking


def paths(args):
    gens = [("s_bend", lambda x, y, yaw: s_bend(args.speed, args.ay, x0=x, y0=y, yaw0=yaw))]
    for i in range(args.random_paths):
        gens.append((f"random_{i}", lambda x, y, yaw, i=i: random_path(args.speed, args.ay, args.path_length, seed=args.seed + i, x0=x, y0=y, yaw0=yaw)))
    return gens


def track(args, terrain: str, controller: str, ident: dict, log=print) -> dict:
    gens = paths(args)
    n = len(gens)
    ctl = make_nmpc(ident, args) if controller.startswith("nmpc") else None
    N, m = args.horizon, args.interval
    with ChronoPool(n, args.python) as pool:
        # every process has its own Chrono system; starting at x = -40 keeps a 100 m path inside the 300 m patch
        obs = pool.reset([{**TERRAINS[terrain], "v": args.speed, "x": -40.0} for _ in range(n)])
        refs = [g(o["x"], o["y"], o["yaw"]) for (_, g), o in zip(gens, obs)]
        active = list(range(n))
        hint = [0] * n
        warm, u_prev = {}, {i: np.array([obs[i]["delta"], 0.0]) for i in range(n)}
        a_est = [0.0] * n
        rec = {i: {"lat": [], "head": [], "dv": [], "x": [], "y": []} for i in range(n)}
        end = {i: "completed" for i in range(n)}
        steps = max(len(r) for r in refs) - 1
        t0 = time.perf_counter()
        for k in range(steps):
            s_cmd, a_cmd, ids = [], [], []
            for i in list(active):
                o, ref = obs[i], refs[i]
                if k >= len(ref) - 1:
                    active.remove(i)
                    continue
                lat, head, hint[i] = path_errors(ref, o["x"], o["y"], o["yaw"], hint[i])
                r = rec[i]
                r["lat"].append(lat)
                r["head"].append(head)
                r["dv"].append(o["vx"] - ref.v)
                r["x"].append(o["x"])
                r["y"].append(o["y"])
                if not o["ok"] or abs(lat) > args.max_lat or (k > 50 and o["vx"] < 0.5):
                    end[i] = "rolled" if not o["ok"] else ("off_path" if abs(lat) > args.max_lat else "stuck")
                    active.remove(i)
                    continue
                a_est[i] = 0.8 * a_est[i] + 0.2 * o["ax"]
                if ctl is None:
                    s, a = pure_pursuit(ref, o, hint[i], ident["vehicle"]["wheelbase"], ident["k_s"])
                else:
                    x0 = np.array([o["x"], o["y"], o["yaw"], o["vx"], o["vy"], o["r"], o["delta"], a_est[i]])
                    nodes = ref.node(k + m * np.arange(1, N + 1))
                    nodes[:, 2] += 2 * math.pi * round((o["yaw"] - nodes[0, 2]) / (2 * math.pi))
                    if i not in warm:
                        kk = np.minimum(k + m * np.arange(N + 1), len(ref) - 1)
                        X = np.stack([ref.x[kk], ref.y[kk], ref.yaw[kk], np.full(N + 1, ref.v), np.zeros(N + 1),
                                      ref.v * ref.kappa[kk], np.zeros(N + 1), np.zeros(N + 1)], -1)
                        X[0] = x0
                        guess = {"x": np.concatenate([X.ravel(), np.zeros(2 * N)])}
                    else:
                        guess = warm[i]
                    u0, sol, _ok = ctl.solve(x0, nodes, u_prev[i], 1.0, guess)
                    warm[i] = {**ctl._shift(sol), "lam_g": sol["lam_g"], "lam_x": sol["lam_x"]}
                    u_prev[i] = u0
                    s, a = float(u0[0]) / ident["k_s"], float(u0[1])
                s_cmd.append(s)
                a_cmd.append(a)
                ids.append(i)
            if not ids:
                break
            for i, o in zip(ids, pool.step(s_cmd, a_cmd, ids)):
                obs[i] = o
        wall = time.perf_counter() - t0
    per = {}
    for i, (name, _) in enumerate(gens):
        r = rec[i]
        lat = np.abs(np.array(r["lat"]))
        per[name] = {"end": end[i], "steps": len(lat), "of": len(refs[i]) - 1, "mean_abs_lat": float(lat.mean()), "max_abs_lat": float(lat.max()),
                     "rms_lat": float(np.sqrt(np.mean(lat**2))), "mean_abs_head": float(np.abs(r["head"]).mean()),
                     "mean_speed_err": float(np.mean(r["dv"])),
                     "track": {"x": [round(v, 2) for v in r["x"][::5]], "y": [round(v, 2) for v in r["y"][::5]]},
                     "ref": {"x": [round(v, 2) for v in refs[i].x[::5]], "y": [round(v, 2) for v in refs[i].y[::5]]}}
    done = [p for p in per.values() if p["end"] == "completed"]
    summary = {"mean_abs_lat": float(np.mean([p["mean_abs_lat"] for p in per.values()])),
               "max_abs_lat": float(max(p["max_abs_lat"] for p in per.values())),
               "completed": f"{len(done)}/{len(per)}", "wall_s": round(wall, 1)}
    if ctl is not None:
        summary["solver"] = ctl.timing()
    log(f"[{terrain}] {controller:>17}: mean |lat| {summary['mean_abs_lat']:.3f} m, max {summary['max_abs_lat']:.2f} m, completed "
        f"{summary['completed']} ({wall:.0f} s)" + (f", solve {summary['solver']['mean_ms']:.1f} ms, {summary['solver']['failures']} failures" if ctl else ""))
    return {"summary": summary, "paths": per}


def main(argv=None) -> int:
    args = parse(argv)
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    report = {"args": vars(args), "terrains": {}}
    idents = {}
    for terrain in args.terrains:
        idents[terrain] = identify(args, terrain, log)
        res = {"identification": idents[terrain], "controllers": {}}
        ctls = ["nmpc", "pure_pursuit"] + (["nmpc_rigid_model"] if terrain != "rigid" and "rigid" in idents else [])
        for c in ctls:
            ident = idents["rigid"] if c == "nmpc_rigid_model" else idents[terrain]
            res["controllers"][c] = track(args, terrain, c, ident, log)
        report["terrains"][terrain] = res
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))

    if "rigid" in report["terrains"]:
        base = report["terrains"]["rigid"]["controllers"]["nmpc"]["summary"]["mean_abs_lat"]
        gate = {}
        for terrain, res in report["terrains"].items():
            if terrain == "rigid":
                continue
            s = res["controllers"]["nmpc"]["summary"]
            ratio = s["mean_abs_lat"] / base
            comp = s["completed"].split("/")
            gate[terrain] = {"ratio_to_rigid": ratio, "terminated": int(comp[1]) - int(comp[0]), "passes": bool(ratio >= 3 or comp[0] != comp[1])}
        report["gate_G1"] = gate
        Path(args.out).write_text(json.dumps(report, indent=1))
        print("\nG1 (NMPC mean |lat| on soil / on rigid ground >= 3, or terminations):")
        for terrain, g in gate.items():
            print(f"  {terrain}: ratio {g['ratio_to_rigid']:.2f}, terminated {g['terminated']} -> {'PASS' if g['passes'] else 'not passed'}")

    print("\nmean |lateral error| [m] (completed paths)")
    names = sorted({c for r in report["terrains"].values() for c in r["controllers"]})
    print(f"{'terrain':>8} " + " ".join(f"{c:>22}" for c in names))
    for terrain, res in report["terrains"].items():
        cells = []
        for c in names:
            s = res["controllers"].get(c, {}).get("summary")
            cells.append(f"{s['mean_abs_lat']:.3f} ({s['completed']})" if s else "-")
        print(f"{terrain:>8} " + " ".join(f"{x:>22}" for x in cells))
    if args.plot:
        plot(report, args.plot)
    print(f"results: {args.out}")
    return 0


def plot(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    terrains = list(report["terrains"])
    fig, ax = plt.subplots(1, len(terrains), figsize=(5.5 * len(terrains), 4.5), squeeze=False)
    colors = {"nmpc": "tab:green", "nmpc_rigid_model": "tab:orange", "pure_pursuit": "tab:purple"}
    for j, terrain in enumerate(terrains):
        a = ax[0, j]
        ctls = report["terrains"][terrain]["controllers"]
        ref = next(iter(ctls.values()))["paths"]["s_bend"]["ref"]
        a.plot(ref["x"], ref["y"], "k--", lw=1, label="reference")
        for c, res in ctls.items():
            p = res["paths"]["s_bend"]
            a.plot(p["track"]["x"], p["track"]["y"], color=colors.get(c), label=f"{c} ({p['mean_abs_lat']:.2f} m)")
        a.set_title(f"{terrain}: S-bend")
        a.axis("equal")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
