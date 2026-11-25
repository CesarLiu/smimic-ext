"""VEHICLE_PROPOSAL.md section 11.15, check G-AS0: does per-corner active suspension force (+-7 kN, set at 50 Hz) change the rollover margin of the HMMWV with an
800 kg roof load?  (1) sign check; (2) slow ramp steer at 13 m/s: lateral acceleration at LTR 0.9 and at two-wheel lift;
(3) fishhook: two-wheel lift time / rollover; passive vs roll feedback vs lean-in.

    SUPERMIMIC_CHRONO_PYTHON is not used: run it with the Chrono interpreter directly, e.g.
    PYTHONPATH=src chrono_env/bin/python scripts/vehicle/chrono_active_suspension_check.py sign
    PYTHONPATH=src chrono_env/bin/python scripts/vehicle/chrono_active_suspension_check.py ramp logs/as_ramp.json
    PYTHONPATH=src chrono_env/bin/python scripts/vehicle/chrono_active_suspension_check.py fish logs/as_fish.json 8,9,10,10.5,11,13,15
"""
import json
import math
import sys
from pathlib import Path

import pychrono as chrono
import pychrono.vehicle as veh

from supermimic.vehicles.chrono_worker import Sim

FMAX = 7000.0
CFG = {"roof_kg": 800.0, "powertrain": "map", "chassis_collision": True, "max_roll": 1.2, "settle": 1.5}

class Active(chrono.ForceFunctor):
    """The shock's own force plus an actuator force (the HMMWV shocks use a force functor, which ignores SetActuatorForce)."""
    def __init__(self, orig):
        super().__init__()
        self.orig, self.F = orig, 0.0
    def evaluate(self, time, rest_length, length, vel, link):
        return self.orig.evaluate(time, rest_length, length, vel, link) + self.F
    def SetActuatorForce(self, f):
        self.F = f

def shocks(s):
    out = []
    for ax in (0, 1):
        dw = veh.CastToChDoubleWishbone(s.v.GetSuspension(ax))
        for side in (veh.LEFT, veh.RIGHT):
            sh = dw.GetShock(side)
            a = Active(sh.GetForceFunctor())
            sh.RegisterForceFunctor(a)
            s._keep = getattr(s, "_keep", []) + [a]
            out.append(a)
    return out  # FL, FR, RL, RR

def ltr(o):
    fz = [o[f"fz{i}"] for i in range(4)]
    return ((fz[0] + fz[2]) - (fz[1] + fz[3])) / max(sum(fz), 1e3)

def lifted(o):
    off = [o[f"fz{i}"] < 100 for i in range(4)]
    return (off[0] and off[2]) or (off[1] and off[3])

def run(mode, steer_fn, T, v=13.0, sign=1.0):
    s = Sim(); o = s.reset({**CFG, "v": v}); sh = shocks(s)
    rows = []
    for k in range(round(T / 0.02)):
        t = k * 0.02
        if mode == "passive":
            u = 0.0
        elif mode == "roll_fb":   # resist body roll (roll moment from roll angle and rate)
            u = -(8.0 * o["roll_b"] + 1.0 * o["p"])
        elif mode == "lean_in":   # roll the body into the turn in proportion to lateral acceleration, plus roll damping
            u = -(8.0 * o["roll_b"] + 1.0 * o["p"]) - 0.12 * o["ay"]
        u = max(-1.0, min(1.0, u)) * sign
        # u > 0: push the left side up and the right side down (a roll moment)
        for i, f in enumerate((u, -u, u, -u)):
            sh[i].SetActuatorForce(FMAX * f)
        o = s.step(steer_fn(t), 0.0)
        rows.append((t, o["ay"], ltr(o), o["roll_b"], lifted(o), o["tilt"], o["ok"]))
        if not o["ok"]:
            break
    return rows

if __name__ == "__main__":
    part = sys.argv[1]
    res = {}
    if part == "sign":
        for sg in (1.0, -1.0):
            s = Sim(); s.reset({**CFG, "v": 5.0}); sh = shocks(s)
            for f, x in zip(sh, (1, -1, 1, -1)):
                f.SetActuatorForce(FMAX * sg * x)
            for _ in range(50):
                o = s.step(0.0, 0.0)
            print(f"u={sg:+.0f}: roll_b {math.degrees(o['roll_b']):+.2f} deg, LTR {ltr(o):+.3f}")
    elif part == "ramp":
        for mode in ("passive", "roll_fb", "lean_in"):
            r = run(mode, lambda t: min(0.95, 0.06 * t), 18.0)
            alift = next((a for t, a, l, ro, li, *_ in r if li), None)
            maxroll = max(abs(x[3]) for x in r)
            res[mode] = {"ay_at_lift": alift, "max_roll_deg": math.degrees(maxroll), "rolled": not r[-1][6]}
            print(mode, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res[mode].items()}, flush=True)
    else:  # fishhook at several speeds
        def fish(t):
            if t < 0.5: return 0.0
            if t < 1.5: return 0.6
            if t < 2.0: return 0.6 - 3.1 * (t - 1.5)
            return -0.95
        for v in [float(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ("11", "13", "15"))]:
            for mode in ("passive", "roll_fb", "lean_in"):
                r = run(mode, fish, 5.0, v=v)
                res[f"{mode}_{v:g}"] = {"lift_s": 0.02 * sum(x[4] for x in r), "max_tilt_deg": math.degrees(max(x[5] for x in r)), "rolled": not r[-1][6],
                                        "max_ltr": max(abs(x[2]) for x in r)}
                print(v, mode, {k: (round(x, 2) if isinstance(x, float) else x) for k, x in res[f"{mode}_{v:g}"].items()}, flush=True)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
