"""Probe 2: (a) a wider lift window (friction 1.1, faster steering) where avoiding the obstacle may need two-wheel lift;
(b) the original task with the MPC re-solving only every 3rd control step (holding its command), as a real-time budget (VEHICLE_PROPOSAL.md section 11.14).

    SUPERMIMIC_CHRONO_PYTHON=chrono_env/bin/python PYTHONPATH=src python scripts/vehicle/mpc_probes/lift_window_and_budget.py \\
        lift|rt logs/chrono_lift_data.npz logs/mpc_probe_<part>.json
"""
import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts/vehicle")
import chrono_roll_mpc as crm
import chrono_rollover_control as crc

from supermimic.vehicles.chrono_env import ChronoPool


class Every:
    """Re-solve every k-th step, hold the command in between (real-time budget of k control periods)."""
    def __init__(self, mpc, k):
        self.mpc, self.k = mpc, k
    def reset(self, n):
        self.mpc.reset(n); self.t, self.u = [0] * n, [None] * n
    def act(self, i, o, path, j, v_ref, a_prev):
        if self.t[i] % self.k == 0 or self.u[i] is None:
            self.u[i] = self.mpc.act(i, o, path, j, v_ref, a_prev)
        self.t[i] += 1
        return self.u[i]

part = sys.argv[1]; data = sys.argv[2]; out = sys.argv[3]
args = types.SimpleNamespace(workers=4)
res = {}
if part == "lift":
    crc.LAT_FAIL, crc.A_MIN, crc.A_MAX, crc.LOOK = 0.4, -0.5, 0.5, 0.4
    crc.LIFT_CFG = {**crc.LIFT_CFG, "friction": 1.1, "steer_rate": 2.0}
    crm.MU_PRIOR = 1.1
    ident = crm.identify(data)
    rng = np.random.default_rng(7); cases = []
    crc.OFFSET, crc.TRANSITION = (3.0, 3.8), (14.0, 22.0)
    while len(cases) < 24:
        p, info = crc.lane_change_path(rng, lead=30.0)
        if not info["double"]:
            cases += [(p, v) for v in (12.0, 14.0, 16.0)]
    with ChronoPool(4) as pool:
        for soft in (0.6, 1.0, 1.3, 10.0):
            m = crm.RollMPC(ident, w_ltr=500.0, ltr_soft=soft)
            m.lbx[m.nd + 1::2], m.ubx[m.nd + 1::2] = crc.A_MIN, crc.A_MAX
            gap = 2.0 * crm.K_STEER * 0.05
            m.lbg[-m.N:], m.ubg[-m.N:] = -gap, gap
            ev, _ = crc.evaluate(pool, cases, m, args)
            res[f"mpc_ltr{soft:g}"] = ev
            print(f"mpc_ltr{soft:g}", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in ev.items() if k != "per_episode"}, flush=True)
    for i, (p, v) in enumerate(cases):
        print(f"ay_path {v*v*float(np.abs(p[:,3]).max()):5.1f} v {v:4.0f} " + "  ".join(
            f"{n}: {r['per_episode'][i]['end'][:4]} LTR {r['per_episode'][i]['max_ltr']:.2f} lift {r['per_episode'][i]['two_wheel_lift_s']:.2f} tilt {r['per_episode'][i]['max_tilt_deg']:.0f}" for n, r in res.items()))
else:  # real-time budget on the original task
    ident = crm.identify(data)
    rng = np.random.default_rng(2000); cases = []
    for _ in range(8):
        p, _ = crc.lane_change_path(rng); cases += [(p, v) for v in crc.EVAL_SPEEDS]
    with ChronoPool(4) as pool:
        for k in (3, 5):
            ev, _ = crc.evaluate(pool, cases, Every(crm.RollMPC(ident, w_ltr=500.0), k), args)
            res[f"every_{k}"] = ev
            print(f"mpc every {k} steps", {kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in ev.items() if kk != "per_episode"}, flush=True)
Path(out).write_text(json.dumps(res))
