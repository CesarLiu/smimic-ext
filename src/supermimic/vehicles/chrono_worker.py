"""One Project Chrono vehicle, driven over a pipe (VEHICLE_PROPOSAL.md section 11, gate G1).

Runs under the PyChrono interpreter (the ``projectchrono`` conda build, see VEHICLE_RUNBOOK.md step 10), not the
project's: it imports only the standard library and ``pychrono``. ``chrono_env.ChronoPool`` starts one process per
vehicle and talks to it with length-prefixed pickles on stdin / stdout (Chrono's own prints are moved to stderr).

The vehicle is Chrono's HMMWV (full multibody model, 4WD, shafts powertrain) on rigid ground (TMeasy tyres) or SCM
deformable soil (rigid tyres). Low-level actuation lives here, the same for every terrain, so a controller commands
what our own environment's actions command:

* steering: a normalised command s_cmd in [-1, 1] through a servo identical to ``VehicleTrackingEnv``'s (per control
  step a constant rate (s_cmd - s) / tau, clamped to a rate limit); the driver's steering input is the servo state;
* longitudinal: an acceleration command a_cmd [m/s^2], integrated into a target speed that a PI loop on the measured
  speed tracks through throttle and brake (the integral term absorbs rolling and bulldozing resistance).

Messages (dicts): {"op": "reset", "cfg": {...}} -> observation; {"op": "step", "s_cmd": f, "a_cmd": f} ->
observation; {"op": "info"} -> vehicle constants; {"op": "close"}.
"""

from __future__ import annotations

import math
import os
import pickle
import struct
import sys
import time

SOILS = {
    # Bekker Kphi, Kc, n, Mohr cohesion [Pa], friction angle [deg], Janosi shear [m], elastic stiffness, damping
    "soft": (2e6, 0.0, 1.1, 0.0, 30.0, 0.01, 2e8, 3e4),  # Chrono's demo_VEH_HMMWV_DefSoil
    "very_soft": (0.2e6, 0.0, 1.1, 0.0, 30.0, 0.01, 4e7, 3e4),  # Chrono's demo_VEH_DeformableSoil
    "firm": (2e7, 0.0, 1.1, 3e3, 30.0, 0.01, 2e9, 3e4),  # stiffer, slightly cohesive variant (ours)
}

DEFAULTS = {
    "terrain": "rigid",  # "rigid" (TMeasy tyres) or "scm" (rigid tyres on deformable soil)
    "soil": "soft",
    "grid": 0.10,  # SCM grid spacing [m]
    "friction": 0.9,  # rigid ground
    "size": 300.0,  # square terrain patch centred on the origin [m]
    "x": 0.0, "y": 0.0, "yaw": 0.0, "v": 5.0,
    "step": 2e-3, "dt": 0.02,
    "settle": 2.0,  # [s] straight driving toward speed v after the drop, before the first observation
    "steer_tau": 0.05, "steer_rate": 1.2,  # servo time constant [s], rate limit [1/s] in normalised units
    "kp": 0.4, "ki": 0.4, "kff": 0.08,  # speed loop: throttle per (m/s) error, per (m/s s), per (m/s^2) command
}


class Sim:
    def __init__(self):
        import pychrono as chrono
        import pychrono.vehicle as veh

        self.chrono, self.veh = chrono, veh
        self.hmmwv = None

    def reset(self, cfg: dict) -> dict:
        chrono, veh = self.chrono, self.veh
        c = {**DEFAULTS, **cfg}
        self.c = c
        h = veh.HMMWV_Full()
        h.SetContactMethod(chrono.ChContactMethod_SMC)
        h.SetChassisFixed(False)
        q = chrono.ChQuaterniond()
        q.SetFromAngleZ(c["yaw"])
        h.SetInitPosition(chrono.ChCoordsysd(chrono.ChVector3d(c["x"], c["y"], 0.6), q))
        h.SetInitFwdVel(c["v"])
        h.SetEngineType(veh.EngineModelType_SHAFTS)
        h.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SHAFTS)
        h.SetDriveType(veh.DrivelineTypeWV_AWD)
        if c["terrain"] == "rigid":
            h.SetTireType(veh.TireModelType_TMEASY)
            h.SetTireStepSize(1e-3)
        else:
            h.SetTireType(veh.TireModelType_RIGID)
        h.Initialize()
        sys_ = h.GetSystem()
        sys_.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        sys_.SetNumThreads(1, 1, 1)
        if c["terrain"] == "rigid":
            terrain = veh.RigidTerrain(sys_)
            mat = chrono.ChContactMaterialSMC()
            mat.SetFriction(c["friction"])
            mat.SetRestitution(0.01)
            mat.SetYoungModulus(2e7)
            terrain.AddPatch(mat, chrono.CSYSNORM, c["size"], c["size"])
            terrain.Initialize()
        else:
            terrain = veh.SCMTerrain(sys_)
            soil = c["soil"] if isinstance(c["soil"], (list, tuple)) else SOILS[c["soil"]]
            terrain.SetSoilParameters(*soil)
            terrain.AddActiveDomain(h.GetChassisBody(), chrono.ChVector3d(0, 0, 0), chrono.ChVector3d(5, 3, 1))
            terrain.Initialize(c["size"], c["size"], c["grid"])
        self.hmmwv, self.terrain, self.sys = h, terrain, sys_
        self.v, self.chassis = h.GetVehicle(), h.GetChassisBody()
        self.inputs = veh.DriverInputs()
        self.s_act, self.v_target, self.integ = 0.0, c["v"], 0.0
        self.throttle = self.brake = 0.0
        self.z0 = None
        self.wall = 0.0
        # settle: the vehicle is dropped onto the ground; drive straight toward the initial speed before handing over
        for _ in range(round(c["settle"] / c["dt"])):
            self.step(0.0, max(-2.0, min(2.0, 1.5 * (c["v"] - self.v.GetSpeed()))))
        self.z0 = None
        self.v_target = c["v"]
        self.wall = 0.0
        return self.observe()

    def info(self) -> dict:
        v, veh = self.v, self.veh
        com = v.GetCOMFrame().GetPos()  # chassis reference frame
        pos, ex = v.GetPos(), v.GetRot().GetAxisX()

        def sp(ax):  # spindle position along the chassis x axis, relative to the chassis reference frame
            d = v.GetSpindlePos(ax, veh.LEFT) - pos
            return d.x * ex.x + d.y * ex.y + d.z * ex.z

        I = v.GetInertia().to_numpy()
        return {"mass": v.GetMass(), "I_z": float(I[2][2]), "a": sp(0) - com.x, "b": com.x - sp(1),
                "wheelbase": v.GetWheelbase(), "max_steer": v.GetMaxSteeringAngle(), "min_turn_radius": v.GetMinTurningRadius()}

    def observe(self) -> dict:
        v, veh = self.v, self.veh
        com = v.GetCOMFrame().GetPos()
        p, vel, acc = v.GetPointLocation(com), v.GetPointVelocity(com), v.GetPointAcceleration(com)
        ex = v.GetRot().GetAxisX()
        ez = v.GetRot().GetAxisZ()
        yaw = math.atan2(ex.y, ex.x)
        c, s = math.cos(yaw), math.sin(yaw)
        if self.z0 is None:
            self.z0 = p.z  # "sink": height lost since the end of the settle phase (not the absolute sinkage)
        delta = 0.5 * (v.GetSteeringAngle(0, veh.LEFT) + v.GetSteeringAngle(0, veh.RIGHT))
        roll = math.asin(max(-1.0, min(1.0, -(ez.x * -s + ez.y * c))))
        obs = {
            "t": self.sys.GetChTime(), "x": p.x, "y": p.y, "z": p.z, "yaw": yaw,
            "vx": c * vel.x + s * vel.y, "vy": -s * vel.x + c * vel.y, "r": self.chassis.GetAngVelParent().z,
            "ax": c * acc.x + s * acc.y, "ay": -s * acc.x + c * acc.y, "rdot": self.chassis.GetAngAccParent().z,
            "delta": delta, "s_act": self.s_act, "v_target": self.v_target, "throttle": self.throttle, "brake": self.brake,
            "roll": roll, "sink": self.z0 - p.z, "wall": self.wall,
        }
        obs["ok"] = all(math.isfinite(x) for x in obs.values()) and abs(roll) < 0.7
        return obs

    def step(self, s_cmd: float, a_cmd: float) -> dict:
        c = self.c
        dt, h = c["dt"], c["step"]
        n = round(dt / h)
        s_cmd = max(-1.0, min(1.0, float(s_cmd)))
        rate = max(-c["steer_rate"], min(c["steer_rate"], (s_cmd - self.s_act) / c["steer_tau"]))
        s0 = self.s_act
        self.v_target = max(0.0, min(40.0, self.v_target + float(a_cmd) * dt))
        speed = self.v.GetSpeed()
        err = self.v_target - speed
        u = c["kp"] * err + c["ki"] * self.integ + c["kff"] * float(a_cmd)
        if -1.0 < u < 1.0 or (u >= 1.0 and err < 0) or (u <= -1.0 and err > 0):  # anti-windup
            self.integ += err * dt
        self.throttle, self.brake = max(0.0, min(1.0, u)), max(0.0, min(1.0, -u))
        t0 = time.perf_counter()
        for k in range(n):
            self.s_act = max(-1.0, min(1.0, s0 + rate * (k + 1) * h))
            self.inputs.m_steering = self.s_act
            self.inputs.m_throttle = self.throttle
            self.inputs.m_braking = self.brake
            t = self.sys.GetChTime()
            self.terrain.Synchronize(t)
            self.hmmwv.Synchronize(t, self.inputs, self.terrain)
            self.terrain.Advance(h)
            self.hmmwv.Advance(h)
        self.wall += time.perf_counter() - t0
        return self.observe()


def main() -> int:
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # anything Chrono prints goes to stderr, not into the protocol
    inp = sys.stdin.buffer

    def send(obj):
        data = pickle.dumps(obj, protocol=4)
        out.write(struct.pack("<I", len(data)) + data)
        out.flush()

    sim = Sim()
    while True:
        hdr = inp.read(4)
        if len(hdr) < 4:
            return 0
        msg = pickle.loads(inp.read(struct.unpack("<I", hdr)[0]))
        try:
            op = msg["op"]
            if op == "reset":
                send(sim.reset(msg.get("cfg", {})))
            elif op == "step":
                send(sim.step(msg["s_cmd"], msg["a_cmd"]))
            elif op == "info":
                send(sim.info())
            elif op == "close":
                send({"closed": True})
                return 0
            else:
                send({"error": f"unknown op {op}"})
        except Exception as e:  # noqa: BLE001  (report instead of dying: the client raises with the message)
            send({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    raise SystemExit(main())
