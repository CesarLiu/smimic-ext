"""Gate G0 of VEHICLE_PROPOSAL.md section 11.4: how expensive is a step of Project Chrono on deformable ground?

Drives Chrono's HMMWV (full model: double-wishbone suspensions, shafts powertrain, 4WD) with a scripted driver
(throttle ramp, then a sinusoidal steering sweep) and times ``Advance`` alone, in several configurations:

- ``rigid_tmeasy``: rigid ground, TMeasy handling tyres (closest to our own simulator);
- ``scm_<delta>``: SCM deformable soil (Bekker / Mohr-Coulomb / Janosi, the parameters of Chrono's
  ``demo_VEH_HMMWV_DefSoil``), rigid tyres, grid spacing ``delta`` m, active domain around the vehicle.

Each configuration runs in 1, 2, ... parallel processes (one Chrono thread each), and the result is counted in 50 Hz
control steps per second, the unit of ``bench_sim.py``. Needs PyChrono with the vehicle module, from the
``projectchrono`` conda channel (the ``pychrono`` on PyPI and the conda-forge build lack it):

    micromamba create -p chrono_env -c projectchrono -c conda-forge "projectchrono::pychrono=10.0.0" python=3.12
    chrono_env/bin/python scripts/vehicle/chrono_throughput.py --out docs/results/vehicle_chrono_throughput.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import time


def run_case(case: dict) -> dict:
    """One vehicle in its own Chrono system; returns timings of the stepping loop only."""
    import pychrono as chrono
    import pychrono.vehicle as veh

    t_setup = time.perf_counter()
    step = case["step"]
    hmmwv = veh.HMMWV_Full()
    hmmwv.SetContactMethod(chrono.ChContactMethod_SMC)
    hmmwv.SetChassisFixed(False)
    hmmwv.SetInitPosition(chrono.ChCoordsysd(chrono.ChVector3d(-case["length"] / 2 + 10, 0, 0.6), chrono.ChQuaterniond(1, 0, 0, 0)))
    hmmwv.SetEngineType(veh.EngineModelType_SHAFTS)
    hmmwv.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SHAFTS)
    hmmwv.SetDriveType(veh.DrivelineTypeWV_AWD)
    if case["terrain"] == "rigid":
        hmmwv.SetTireType(veh.TireModelType_TMEASY)
        hmmwv.SetTireStepSize(case["tire_step"])
    else:
        hmmwv.SetTireType(veh.TireModelType_RIGID)
    hmmwv.Initialize()
    system = hmmwv.GetSystem()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetNumThreads(1, 1, 1)

    if case["terrain"] == "rigid":
        terrain = veh.RigidTerrain(system)
        mat = chrono.ChContactMaterialSMC()
        mat.SetFriction(0.9)
        mat.SetRestitution(0.01)
        mat.SetYoungModulus(2e7)
        terrain.AddPatch(mat, chrono.CSYSNORM, case["length"], case["width"])
        terrain.Initialize()
    else:
        terrain = veh.SCMTerrain(system)
        terrain.SetSoilParameters(2e6, 0, 1.1, 0, 30, 0.01, 2e8, 3e4)
        terrain.AddActiveDomain(hmmwv.GetChassisBody(), chrono.ChVector3d(0, 0, 0), chrono.ChVector3d(5, 3, 1))
        terrain.Initialize(case["length"], case["width"], case["delta"])

    inputs = veh.DriverInputs()
    vehicle = hmmwv.GetVehicle()
    setup_s = time.perf_counter() - t_setup

    def drive(t):
        inputs.m_throttle = min(1.0, 3.5 * t) * case["throttle"]
        inputs.m_steering = 0.0 if t < 1.0 else 0.5 * math.sin(2 * math.pi * (t - 1.0) / 4.0)
        inputs.m_braking = 0.0

    n_warm = round(case["warmup_s"] / step)
    n_meas = round(case["sim_s"] / step)
    speeds, wall = [], 0.0
    for k in range(n_warm + n_meas):
        t = system.GetChTime()
        drive(t)
        t0 = time.perf_counter()
        terrain.Synchronize(t)
        hmmwv.Synchronize(t, inputs, terrain)
        terrain.Advance(step)
        hmmwv.Advance(step)
        if k >= n_warm:
            wall += time.perf_counter() - t0
            if k % round(0.02 / step) == 0:
                speeds.append(vehicle.GetSpeed())
    pos = vehicle.GetPos()
    return {
        "setup_s": round(setup_s, 3),
        "sim_s": case["sim_s"],
        "wall_s": round(wall, 3),
        "real_time_factor": round(wall / case["sim_s"], 3),  # wall time per simulated second (> 1: slower than real time)
        "control_steps_per_s": round(case["sim_s"] * 50 / wall, 1),
        "mean_speed_mps": round(sum(speeds) / max(1, len(speeds)), 2),
        "final_x_m": round(pos.x, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=["rigid_tmeasy", "scm_0.10", "scm_0.05"])
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4], help="parallel processes (one vehicle each)")
    ap.add_argument("--sim-s", type=float, default=6.0, help="timed simulated seconds per vehicle")
    ap.add_argument("--warmup-s", type=float, default=1.0)
    ap.add_argument("--throttle", type=float, default=0.8)
    ap.add_argument("--ours", type=float, default=87360.0,
                    help="our simulator's env steps / s on the same machine (bench_sim.py --device cpu, best of --num-envs)")
    ap.add_argument("--out", default="logs/chrono_throughput.json")
    args = ap.parse_args(argv)

    import pychrono

    report = {"machine": {"cpus": os.cpu_count(), "platform": platform.platform(), "pychrono": getattr(pychrono, "__version__", "?")},
              "ours_steps_per_s": args.ours, "settings": vars(args), "cases": {}}
    for name in args.cases:
        base = {"length": 400.0, "width": 40.0, "sim_s": args.sim_s, "warmup_s": args.warmup_s, "throttle": args.throttle}
        if name == "rigid_tmeasy":
            case = {**base, "terrain": "rigid", "step": 2e-3, "tire_step": 1e-3}
        elif name.startswith("scm_"):
            case = {**base, "terrain": "scm", "step": 2e-3, "delta": float(name.split("_")[1])}
        else:
            raise SystemExit(f"unknown case {name}")
        report["cases"][name] = {"config": case, "runs": {}}
        for n in args.procs:
            t0 = time.perf_counter()
            with mp.get_context("spawn").Pool(n) as pool:
                res = pool.map(run_case, [case] * n)
            total = sum(r["control_steps_per_s"] for r in res)
            report["cases"][name]["runs"][str(n)] = {"per_process": res, "aggregate_control_steps_per_s": round(total, 1),
                                                     "ours_over_chrono": round(args.ours / total, 1), "elapsed_s": round(time.perf_counter() - t0, 1)}
            r0 = res[0]
            print(f"{name:>13}  procs {n}:  {total:9.1f} control steps/s in total  (per process {r0['control_steps_per_s']:.1f}, "
                  f"wall/sim {r0['real_time_factor']:.2f}, mean speed {r0['mean_speed_mps']:.1f} m/s)  ours / chrono = {args.ours / total:,.0f}x",
                  flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    print(f"results: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
