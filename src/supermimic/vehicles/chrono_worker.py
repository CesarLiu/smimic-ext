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
observation; {"op": "rollout", "cfg": {...}, "actions": [[s_cmd, a_cmd], ...]} -> {"obs0": observation after the
reset, "obs": one observation per action, cut where the observation stops being ok}; {"op": "info"} -> vehicle
constants; {"op": "close"}. A rollout is deterministic: the same cfg and actions give the same observations, which is
how finite differences are taken without copying Chrono's state.

Options for wheel-lift studies (defaults keep the G1 setting): ``roof_kg`` (a box rigidly fixed 1.6 m above the
chassis reference frame), ``powertrain`` "map" (map-based engine and gearbox; the shafts model starts in first gear
and brakes hard at speed), ``chassis_collision`` (the chassis lands when it rolls), ``max_roll`` [rad].

Options for the active-chassis study (section 11.15, G-AS1): ``heightmap`` (a BMP written by
``supermimic.vehicles.road``, placed by ``hm_center``, ``hm_size``, ``hm_range``) replaces the flat rigid patch;
``active_susp`` adds a force actuator in parallel with each of the four shocks (``susp_fmax`` [N], first-order lag
``susp_tau`` [s]), commanded by ``f_cmd`` (four values in [-1, 1], FL FR RL RR, positive = push the body up) in the
step message or as action elements 3-6 of a rollout. Either option adds observations: suspension deflection and
deflection rate ``dz*`` / ``dzd*`` (relative to the settled state), actuator force ``fa*`` (normalised), road height
under each wheel ``zr*``, and the body's vertical acceleration ``az``.
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
    "roof_kg": 0.0, "powertrain": "shafts", "chassis_collision": False,
    # sloshing payload: the roof load hangs on a pendulum about the chassis x axis (length [m], damping [N m s/rad]),
    # its centre at the same static height as the rigid load; 0 = rigid
    "slosh_length": 0.0, "slosh_damping": 2000.0,
    "max_roll": 0.7,  # [rad] the observation is not ok beyond this roll
    "heightmap": None, "hm_center": [110.0, 0.0], "hm_size": [240.0, 40.0], "hm_range": [-0.25, 0.25],
    "active_susp": False, "susp_fmax": 7000.0, "susp_tau": 0.02,
}


def _active_functor(chrono):
    class Active(chrono.ForceFunctor):
        """The shock's own force plus an actuator force (the HMMWV shocks use a force functor, which ignores
        ``ChLinkTSDA.SetActuatorForce``)."""

        def __init__(self, orig):
            super().__init__()
            self.orig, self.F = orig, 0.0

        def evaluate(self, time, rest_length, length, vel, link):
            return self.orig.evaluate(time, rest_length, length, vel, link) + self.F

    return Active


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
        if c["powertrain"] == "map":
            h.SetEngineType(veh.EngineModelType_SIMPLE_MAP)
            h.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SIMPLE_MAP)
        else:
            h.SetEngineType(veh.EngineModelType_SHAFTS)
            h.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SHAFTS)
        if c["chassis_collision"]:
            h.SetChassisCollisionType(veh.CollisionType_PRIMITIVES)
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
        self.slosh = None
        if c["roof_kg"] > 0:
            ch = h.GetChassisBody()
            roof = chrono.ChBodyEasyBox(1.6, 1.4, 0.3, c["roof_kg"] / (1.6 * 1.4 * 0.3), True, False)
            roof.SetPos(ch.GetPos() + ch.GetRot().Rotate(chrono.ChVector3d(0, 0, 1.6)))
            roof.SetRot(ch.GetRot())
            sys_.Add(roof)
            if c["slosh_length"] > 0:  # pendulum: pivot slosh_length above the load's centre, axis = chassis x
                qy = chrono.ChQuaterniond()
                qy.SetFromAngleY(math.pi / 2)
                frame = chrono.ChFramed(ch.GetPos() + ch.GetRot().Rotate(chrono.ChVector3d(0, 0, 1.6 + c["slosh_length"])), ch.GetRot() * qy)
                joint = chrono.ChLinkLockRevolute()
                joint.Initialize(roof, ch, frame)
                sys_.Add(joint)
                damper = chrono.ChLinkRSDA()
                damper.Initialize(roof, ch, frame)
                damper.SetSpringCoefficient(0.0)
                damper.SetDampingCoefficient(c["slosh_damping"])
                sys_.Add(damper)
                self.slosh = damper
            else:
                link = chrono.ChLinkLockLock()
                link.Initialize(roof, ch, chrono.ChFramed(roof.GetPos(), roof.GetRot()))
                sys_.Add(link)
        self.act = []
        if c["active_susp"]:
            Active = _active_functor(chrono)
            for ax in (0, 1):
                dw = veh.CastToChDoubleWishbone(h.GetVehicle().GetSuspension(ax))
                for side in (veh.LEFT, veh.RIGHT):
                    sh = dw.GetShock(side)
                    a = Active(sh.GetForceFunctor())
                    sh.RegisterForceFunctor(a)
                    self.act.append(a)
        self.f_cmd = [0.0] * 4
        if c["terrain"] == "rigid":
            terrain = veh.RigidTerrain(sys_)
            mat = chrono.ChContactMaterialSMC()
            mat.SetFriction(c["friction"])
            mat.SetRestitution(0.01)
            mat.SetYoungModulus(2e7)
            if c["heightmap"]:
                (cx, cy), (sx, sy), (h0, h1) = c["hm_center"], c["hm_size"], c["hm_range"]
                terrain.AddPatch(mat, chrono.ChCoordsysd(chrono.ChVector3d(cx, cy, 0), chrono.QUNIT), c["heightmap"], sx, sy, h0, h1)
            else:
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
        self.dz0 = None
        # settle: the vehicle is dropped onto the ground; drive straight toward the initial speed before handing over
        for _ in range(round(c["settle"] / c["dt"])):
            self.step(0.0, max(-2.0, min(2.0, 1.5 * (c["v"] - self.v.GetSpeed()))))
        self.z0 = None
        self.v_target = c["v"]
        self.wall = 0.0
        if self.act or c["heightmap"]:
            self.dz0 = [self._shock(i).GetLength() for i in range(4)]
        return self.observe()

    def _shock(self, i):
        return self.veh.CastToChDoubleWishbone(self.v.GetSuspension(i // 2)).GetShock(self.veh.LEFT if i % 2 == 0 else self.veh.RIGHT)

    def info(self) -> dict:
        v, veh = self.v, self.veh
        com = v.GetCOMFrame().GetPos()  # chassis reference frame
        pos, ex = v.GetPos(), v.GetRot().GetAxisX()

        def sp(ax):  # spindle position along the chassis x axis, relative to the chassis reference frame
            d = v.GetSpindlePos(ax, veh.LEFT) - pos
            return d.x * ex.x + d.y * ex.y + d.z * ex.z

        I = v.GetInertia().to_numpy()
        dl = v.GetSpindlePos(0, veh.LEFT) - v.GetSpindlePos(0, veh.RIGHT)
        return {"mass": v.GetMass(), "I_z": float(I[2][2]), "a": sp(0) - com.x, "b": com.x - sp(1), "track": math.hypot(dl.x, dl.y),
                "wheelbase": v.GetWheelbase(), "max_steer": v.GetMaxSteeringAngle(), "min_turn_radius": v.GetMinTurningRadius()}

    def observe(self) -> dict:
        v, veh = self.v, self.veh
        com = v.GetCOMFrame().GetPos()
        p, vel, acc = v.GetPointLocation(com), v.GetPointVelocity(com), v.GetPointAcceleration(com)
        ex = v.GetRot().GetAxisX()
        ey = v.GetRot().GetAxisY()
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
            # wheel-lift study: body-frame roll / pitch rates, vertical speed, roll and pitch valid at large angles,
            # tilt of the body's z axis from vertical, speed-loop integrator, tyre normal loads (FL, FR, RL, RR)
            "vz": vel.z, "p": self.chassis.GetAngVelLocal().x, "q": self.chassis.GetAngVelLocal().y,
            "roll_b": math.atan2(ey.z, ez.z), "pitch_b": math.atan2(-ex.z, math.hypot(ex.x, ex.y)),
            "tilt": math.acos(max(-1.0, min(1.0, ez.z))), "integ": self.integ,
        }
        for i, (a, side) in enumerate((a, s) for a in (0, 1) for s in (veh.LEFT, veh.RIGHT)):
            obs[f"fz{i}"] = v.GetTire(a, side).ReportTireForce(self.terrain).force.z
        if self.slosh is not None:
            obs["slosh"] = self.slosh.GetAngle()  # diagnostics only: no controller or world model sees it
        if self.dz0 is not None:
            obs["az"] = acc.z
            for i in range(4):
                sh = self._shock(i)
                obs[f"dz{i}"] = self.dz0[i] - sh.GetLength()  # compression positive
                obs[f"dzd{i}"] = -sh.GetVelocity()
                obs[f"fa{i}"] = self.act[i].F / self.c["susp_fmax"] if self.act else 0.0
                sp = v.GetSpindlePos(i // 2, veh.LEFT if i % 2 == 0 else veh.RIGHT)
                obs[f"zr{i}"] = self.terrain.GetHeight(self.chrono.ChVector3d(sp.x, sp.y, sp.z + 1.0))
        obs["ok"] = all(math.isfinite(x) for x in obs.values()) and abs(roll) < self.c["max_roll"]
        return obs

    def step(self, s_cmd: float, a_cmd: float, f_cmd=None) -> dict:
        c = self.c
        if f_cmd is not None:
            self.f_cmd = [max(-1.0, min(1.0, float(f))) for f in f_cmd]
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
            if self.act:
                alpha = 1.0 - math.exp(-h / c["susp_tau"])
                for a, f in zip(self.act, self.f_cmd):
                    a.F += alpha * (c["susp_fmax"] * f - a.F)
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
                send(sim.step(msg["s_cmd"], msg["a_cmd"], msg.get("f_cmd")))
            elif op == "rollout":
                obs0, rows = sim.reset(msg.get("cfg", {})), []
                for act in msg["actions"]:
                    rows.append(sim.step(act[0], act[1], act[2:6] if len(act) > 2 else None))
                    if not rows[-1]["ok"]:
                        break
                send({"obs0": obs0, "obs": rows})
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
