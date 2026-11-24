"""Wheel lift and rollover with Project Chrono's HMMWV (VEHICLE_PROPOSAL.md section 11.11).

Our own simulator cannot roll a car over (small-angle roll, wheel loads clamped above zero, flat road). Chrono's
full multibody vehicle can: the chassis moves in all six degrees of freedom, a wheel can leave the ground, and the
chassis collides with the ground (primitive collision shapes). Two scenarios, each with three runs:

* ``fishhook`` (rigid ground, mu 0.9, TMeasy tyres): steer left, then reverse to full right lock and hold, throttle
  released at the start (the shape of NHTSA's fishhook test). The stock HMMWV has a static stability factor
  SSF = track / (2 x CoG height) of about 1.82 / (2 x 0.78) = 1.2, above the tyres' friction, so it slides rather
  than rolls; with an 800 kg roof load (CoG about 1.1 m, SSF about 0.8) two wheels lift at low speed and it rolls
  over at higher speed;
* ``trip`` (ice, mu 0.15, rigid tyres so that the wheels can strike an obstacle): the vehicle slides sideways;
  without an obstacle it slides on, into a 0.2 m curb at 4 m/s it is stopped, at 8 m/s it trips and rolls.

Runs under the PyChrono interpreter (VEHICLE_RUNBOOK.md step 10) and writes the poses of the chassis, wheels and
roof load at 50 Hz plus the tyre loads to a JSON file; ``render_rollover.py`` (project environment) draws them.

    chrono_env/bin/python -I scripts/vehicle/chrono_rollover.py --out logs/rollover.json
"""

from __future__ import annotations

import argparse
import json
import math
import time

import pychrono as chrono
import pychrono.vehicle as veh

STEP = 1e-3
LOG_EVERY = 20  # steps: 50 Hz
ROOF = (1.6, 1.4, 0.3, 1.6)  # roof load box size [m] and height of its centre above the chassis reference frame


def vec(v):
    return [round(v.x, 4), round(v.y, 4), round(v.z, 4)]


def axes(q):
    return [vec(q.GetAxisX()), vec(q.GetAxisY()), vec(q.GetAxisZ())]


def build(*, tyres: str, mu: float, v0: float, roof_kg: float = 0.0, yaw: float = 0.0, curb: float = 0.0):
    h = veh.HMMWV_Full()
    h.SetContactMethod(chrono.ChContactMethod_SMC)
    h.SetChassisCollisionType(veh.CollisionType_PRIMITIVES)  # the chassis hits the ground once it rolls
    q = chrono.ChQuaterniond()
    q.SetFromAngleZ(yaw)
    h.SetInitPosition(chrono.ChCoordsysd(chrono.ChVector3d(0, 0, 0.6), q))
    h.SetInitFwdVel(v0)
    # map-based engine and gearbox: the shafts powertrain starts in first gear and brakes hard at speed
    h.SetEngineType(veh.EngineModelType_SIMPLE_MAP)
    h.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SIMPLE_MAP)
    h.SetDriveType(veh.DrivelineTypeWV_AWD)
    if tyres == "tmeasy":
        h.SetTireType(veh.TireModelType_TMEASY)
        h.SetTireStepSize(STEP)
    else:
        h.SetTireType(veh.TireModelType_RIGID)
    h.Initialize()
    sys_ = h.GetSystem()
    sys_.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    ch = h.GetChassisBody()
    roof = None
    if roof_kg > 0:
        lx, ly, lz, dz = ROOF
        roof = chrono.ChBodyEasyBox(lx, ly, lz, roof_kg / (lx * ly * lz), True, False)
        roof.SetPos(ch.GetPos() + ch.GetRot().Rotate(chrono.ChVector3d(0, 0, dz)))
        roof.SetRot(ch.GetRot())
        sys_.Add(roof)
        link = chrono.ChLinkLockLock()
        link.Initialize(roof, ch, chrono.ChFramed(roof.GetPos(), roof.GetRot()))
        sys_.Add(link)
    terrain = veh.RigidTerrain(sys_)
    mat = chrono.ChContactMaterialSMC()
    mat.SetFriction(mu)
    mat.SetRestitution(0.01)
    mat.SetYoungModulus(2e7)
    terrain.AddPatch(mat, chrono.CSYSNORM, 600, 300)
    terrain.Initialize()
    if curb > 0:
        cm = chrono.ChContactMaterialSMC()
        cm.SetFriction(0.8)
        cm.SetRestitution(0.01)
        cm.SetYoungModulus(2e7)
        body = chrono.ChBodyEasyBox(0.4, 30, curb, 2000, True, True, cm)
        body.SetPos(chrono.ChVector3d(CURB_X, 0, curb / 2))
        body.SetFixed(True)
        sys_.Add(body)
    return h, terrain, roof


CURB_X = 6.0


def record(h, terrain, roof, t, inputs):
    v = h.GetVehicle()
    wheels = [(a, s) for a in (0, 1) for s in (veh.LEFT, veh.RIGHT)]
    ez = v.GetRot().GetAxisZ()
    vel = h.GetChassisBody().GetPosDt()
    return {"t": round(t, 3), "pos": vec(v.GetPos()), "axes": axes(v.GetRot()), "speed": round(vel.Length(), 3),
            "tilt_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, ez.z)))), 2),
            "spindles": [vec(v.GetSpindlePos(a, s)) for a, s in wheels], "spindle_axes": [axes(v.GetSpindleRot(a, s)) for a, s in wheels],
            "fz": [round(v.GetTire(a, s).ReportTireForce(terrain).force.z, 1) for a, s in wheels],
            "roof": None if roof is None else {"pos": vec(roof.GetPos()), "axes": axes(roof.GetRot())},
            "steering": round(inputs.m_steering, 3)}


def roll_angle(rec):
    """Signed roll of the chassis about its own x axis [deg]: positive when the left side goes up."""
    _ex, ey, ez = rec["axes"]
    return math.degrees(math.atan2(ey[2], ez[2]))


def fishhook(v0: float, roof_kg: float, A: float = 0.5, run_up: float = 2.0, T: float = 5.0):
    h, terrain, roof = build(tyres="tmeasy", mu=0.9, v0=v0, roof_kg=roof_kg)
    v, sys_ = h.GetVehicle(), h.GetSystem()
    inp = veh.DriverInputs()
    integ, rows = 0.0, []
    for k in range(round((run_up + T) / STEP)):
        t = sys_.GetChTime()
        tm = t - run_up
        if tm < 0:  # run-up: hold the speed, straight
            err = v0 - v.GetSpeed()
            integ += err * STEP
            u = 0.5 * err + 0.3 * integ
            inp.m_throttle, inp.m_braking, inp.m_steering = max(0.0, min(1.0, u)), max(0.0, min(1.0, -u)), 0.0
        else:  # fishhook: left to A, dwell, reverse to full right, hold; throttle released
            inp.m_throttle = inp.m_braking = 0.0
            inp.m_steering = (A * tm / 0.25 if tm < 0.25 else A if tm < 0.5 else A - (A + 1.0) * (tm - 0.5) / 0.5 if tm < 1.0 else -1.0)
        terrain.Synchronize(t)
        h.Synchronize(t, inp, terrain)
        terrain.Advance(STEP)
        h.Advance(STEP)
        if tm >= 0 and k % LOG_EVERY == 0:
            rows.append(record(h, terrain, roof, tm, inp))
    return rows


def trip(v_side: float, curb: float, settle: float = 0.5, T: float = 4.0):
    h, terrain, roof = build(tyres="rigid", mu=0.15, v0=0.0, yaw=math.pi / 2, curb=curb)
    sys_ = h.GetSystem()
    inp = veh.DriverInputs()
    for _ in range(round(settle / STEP)):
        t = sys_.GetChTime()
        terrain.Synchronize(t)
        h.Synchronize(t, inp, terrain)
        terrain.Advance(STEP)
        h.Advance(STEP)
    for b in sys_.GetBodies():  # an already sliding vehicle: every body moves sideways (+x), wheels not spinning
        if not b.IsFixed():
            b.SetPosDt(chrono.ChVector3d(v_side, 0, 0))
    rows = []
    for k in range(round(T / STEP)):
        t = sys_.GetChTime()
        terrain.Synchronize(t)
        h.Synchronize(t, inp, terrain)
        terrain.Advance(STEP)
        h.Advance(STEP)
        if k % LOG_EVERY == 0:
            rows.append(record(h, terrain, roof, k * STEP, inp))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="logs/rollover.json")
    args = ap.parse_args(argv)
    t0 = time.time()
    scenarios = {
        "fishhook": {"ground": "rigid, mu 0.9, TMeasy tyres", "runs": [
            {"label": "原车，16 m/s", "v0": 16.0, "roof_kg": 0.0},
            {"label": "车顶载重 800 kg，9 m/s", "v0": 9.0, "roof_kg": 800.0},
            {"label": "车顶载重 800 kg，16 m/s", "v0": 16.0, "roof_kg": 800.0}]},
        "trip": {"ground": "ice, mu 0.15, rigid tyres", "curb_x": CURB_X, "curb_size": [0.4, 30.0], "runs": [
            {"label": "无路肩，横滑 8 m/s", "v_side": 8.0, "curb": 0.0},
            {"label": "20 cm 路肩，横滑 4 m/s", "v_side": 4.0, "curb": 0.2},
            {"label": "20 cm 路肩，横滑 8 m/s", "v_side": 8.0, "curb": 0.2}]},
    }
    for run in scenarios["fishhook"]["runs"]:
        run["rows"] = fishhook(run["v0"], run["roof_kg"])
    for run in scenarios["trip"]["runs"]:
        run["rows"] = trip(run["v_side"], run["curb"])
    for name, sc in scenarios.items():
        print(name)
        for run in sc["runs"]:
            rows = run["rows"]
            for r in rows:
                r["roll_deg"] = round(roll_angle(r), 2)
            # wheels: front left, front right, rear left, rear right; both wheels of one side below 100 N
            two = [r["t"] for r in rows if (r["fz"][0] < 100 and r["fz"][2] < 100) or (r["fz"][1] < 100 and r["fz"][3] < 100)]
            run["summary"] = {"max_tilt_deg": max(r["tilt_deg"] for r in rows), "rolled_over": max(r["tilt_deg"] for r in rows) > 90,
                              "two_wheel_lift_s": round(len(two) * LOG_EVERY * STEP, 2), "first_two_wheel_lift_t": two[0] if two else None,
                              "speed_at_start": rows[0]["speed"]}
            print(f"  {run['label']:<24} {run['summary']}")
    with open(args.out, "w") as f:
        json.dump({"scenarios": scenarios, "roof_box": ROOF, "note": __doc__.split("\n\n")[0]}, f, ensure_ascii=False)
    print(f"written {args.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
