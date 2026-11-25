"""Probe for VEHICLE_PROPOSAL.md section 11.14: a single evasive lane change (offset 3-3.8 m over 14-22 m, corridor
0.4 m, acceleration limited to +-0.5 m/s^2), pure pursuit against the roll MPC with LTR limits 0.6 and 1.0.

    # arguments: path seed (7 in section 11.14), lift data, output
    SUPERMIMIC_CHRONO_PYTHON=chrono_env/bin/python PYTHONPATH=src python scripts/vehicle/mpc_probes/evasive.py \\
        7 logs/chrono_lift_data.npz logs/mpc_probe_evasive.json
"""
import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts/vehicle")
import chrono_rollover_control as crc
from chrono_roll_mpc import RollMPC, identify

from supermimic.vehicles.chrono_env import ChronoPool

crc.LAT_FAIL, crc.A_MIN, crc.A_MAX, crc.LOOK = 0.4, -0.5, 0.5, 0.4

def evade_path(rng):
    rng_state = crc.OFFSET, crc.TRANSITION
    crc.OFFSET, crc.TRANSITION = (3.0, 3.8), (14.0, 22.0)
    while True:
        p, info = crc.lane_change_path(rng, lead=30.0)
        if not info["double"]:
            break
    crc.OFFSET, crc.TRANSITION = rng_state
    return p, info

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
cases = []
for _ in range(8):
    p, info = evade_path(rng)
    for v in (12.0, 14.0, 16.0):
        cases.append((p, v))
ident = identify(sys.argv[2])
args = types.SimpleNamespace(workers=4)
res = {}
with ChronoPool(4) as pool:
    for name in ("pure_pursuit", "mpc_ltr0.6", "mpc_ltr1.0"):
        if name == "pure_pursuit":
            ctl = "pure_pursuit"
        else:
            ctl = RollMPC(ident, w_ltr=500.0, ltr_soft=float(name.split("ltr")[1]))
            ctl.lbx[ctl.nd + 1::2], ctl.ubx[ctl.nd + 1::2] = crc.A_MIN, crc.A_MAX
        ev, _ = crc.evaluate(pool, cases, ctl, args)
        res[name] = ev
        print(name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in ev.items() if k != "per_episode"}, flush=True)
for i, (p, v) in enumerate(cases):
    need = v * v * float(np.abs(p[:, 3]).max())
    print(f"ay_need {need:5.1f}  v {v:4.0f}  " + "  ".join(f"{n}: {res[n]['per_episode'][i]['end'][:4]} LTR {res[n]['per_episode'][i]['max_ltr']:.2f} lift {res[n]['per_episode'][i]['two_wheel_lift_s']:.2f}" for n in res))
Path(sys.argv[3]).write_text(json.dumps(res))
