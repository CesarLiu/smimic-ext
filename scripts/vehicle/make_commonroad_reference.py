"""Record the CommonRoad multi-body model's responses to the validation manoeuvres.

Needs ``pip install commonroad-vehicle-models`` (BSD; Althoff & Würsching, "CommonRoad: Vehicle
Models"). It also records CommonRoad's tyre equations on a grid of slips, slip angles, camber angles and
loads. The multi-body model (29 states, vehicle 2 = BMW 320i, PAC2002-type tyres) is integrated with
LSODA (rtol 1e-6, atol 1e-8), the inputs held over each 20 ms control step. ``init_mb`` does not start
the suspension at equilibrium, so each run first settles for 1 s driving straight at the initial speed
(not recorded). Output: per manoeuvre, at every
control step, the quantities the simulator tests compare (``tests/test_vehicle_sim.py``).

    PYTHONPATH=src python scripts/vehicle/make_commonroad_reference.py --out tests/data/commonroad_mb_reference.pt
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp
from vehiclemodels.init_mb import init_mb
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.utils import tire_model
from vehiclemodels.vehicle_dynamics_mb import vehicle_dynamics_mb

from supermimic.vehicles.maneuvers import DT, MANEUVERS

SETTLE = 1.0


def _step(x, u, p):
    sol = solve_ivp(lambda _t, y: vehicle_dynamics_mb(list(y), list(u), p), (0.0, DT), x, method="LSODA", rtol=1e-6, atol=1e-8)
    if sol.status != 0 or not np.isfinite(sol.y[:, -1]).all():
        raise RuntimeError(f"integration failed: {sol.message}")
    return sol.y[:, -1]


def run(man, p):
    x = np.array(init_mb([0.0, 0.0, 0.0, man.v0, 0.0, 0.0, 0.0], p), dtype=float)
    for _ in range(round(SETTLE / DT)):
        x = _step(x, (0.0, 2.0 * (man.v0 - x[3])), p)
    x[0] = x[1] = x[4] = 0.0  # restart the pose at the origin
    rows = []
    for i in range(man.steps + 1):
        t = i * DT
        f = vehicle_dynamics_mb(list(x), [0.0, 0.0], p)  # for the logged accelerations
        rows.append([t, x[0], x[1], x[4], x[3], x[10], x[5], x[2], x[6], x[8], f[3] - x[5] * x[10], f[10] + x[5] * x[3], *x[23:27]])
        if i == man.steps:
            break
        x = _step(x, man.inputs(t, x[3], x[2]), p)
    return np.array(rows)


def tyre_grid(p):
    rows = list(itertools.product([-0.5, -0.2, -0.05, -0.01, 0.0, 0.01, 0.05, 0.2, 0.6], [-0.3, -0.1, -0.02, 0.0, 0.03, 0.12, 0.4], [-0.05, 0.0, 0.04], [1500.0, 3000.0, 5200.0]))
    out = []
    for k, a, g, fz in rows:
        fx0 = tire_model.formula_longitudinal(k, g, fz, p.tire)
        fy0, mu_y = tire_model.formula_lateral(a, g, fz, p.tire)
        out.append([k, a, g, fz, tire_model.formula_longitudinal_comb(k, a, fx0, p.tire), tire_model.formula_lateral_comb(k, a, g, mu_y, fz, fy0, p.tire)])
    return torch.tensor(out, dtype=torch.float64)


COLUMNS = ("t", "x", "y", "yaw", "vx", "vy", "r", "delta", "roll", "pitch", "ax", "ay", "w_lf", "w_rf", "w_lr", "w_rr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/data/commonroad_mb_reference.pt")
    args = ap.parse_args()
    p = parameters_vehicle2()
    out = {"columns": COLUMNS, "vehicle": "commonroad vehicle 2 (BMW 320i), multi-body model", "maneuvers": {}, "tyre_grid": tyre_grid(p)}
    for man in MANEUVERS:
        t0 = time.time()
        a = run(man, p)
        out["maneuvers"][man.name] = torch.tensor(a, dtype=torch.float32)
        last = a[-1]
        print(f"{man.name:28s} {time.time() - t0:5.1f}s  vx {last[4]:6.2f}  r {last[6]:+.4f}  ay {last[11]:+6.2f}  beta {math.degrees(math.atan2(last[5], last[4])):+5.2f}deg  roll {math.degrees(last[8]):+5.2f}deg")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
