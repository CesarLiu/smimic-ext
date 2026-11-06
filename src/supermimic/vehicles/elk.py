"""Elk test (moose test): an ISO 3888-2-style obstacle-avoidance lane change, as a reference and a set of cones.

Layout, along the track from the first gate: a lane of ``sections[0]`` m, a transition of ``sections[1]`` m to a
second lane offset sideways by ``offset``, ``sections[2]`` m in it, a transition back of ``sections[3]`` m, a
last lane of ``sections[4]`` m in line with the first. Lane widths scale with the vehicle width ``b``: 1.1 b + 0.25
(first), 1.2 b + 0.25 (second), 1.3 b + 0.25 (last). The numbers are the ones the author remembers of ISO
3888-2 (12 / 13.5 / 11 / 12.5 / 12 m, 1 m offset, those width factors); they were NOT checked against the
standard text (paywalled), so treat the track as "ISO-style". The car is driven at constant speed.

The reference (``elk_reference``) is the lane-centre line. The scripted open-loop driver used for the other
references cannot follow such an abrupt path (it is open loop in yaw and the path needs a steering reversal in under a
second), so the reference is what the privileged NMPC baseline (``supermimic.baselines.nmpc``, needs casadi) achieves
when it tracks the ideal lane-centre path under the nominal context: feasible by construction, with its recorded
commands. Scoring does not depend on it, though: a run is scored against the cones, not against the reference. A run is scored against the cones, not against the reference: any corner of the body
footprint outside a lane while inside that lane's length counts as a hit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from supermimic.vehicles.env import ReferenceLibrary, VehicleEnvCfg
from supermimic.vehicles.params import VehicleParams


@dataclass
class ElkTrack:
    sections: tuple[float, float, float, float, float] = (12.0, 13.5, 11.0, 12.5, 12.0)
    offset: float = 1.0
    width_factors: tuple[float, float, float] = (1.1, 1.2, 1.3)  # lane widths: factor * b + 0.25
    vehicle_width: float = 1.8
    front_overhang: float = 0.9  # body beyond the front axle [m]
    rear_overhang: float = 0.9
    lead: float = 1.0  # [s] driven before the first gate
    tail: float = 1.5  # [s] after the last
    x0: float = 0.0  # x of the first gate, set by ``elk_reference``
    # body extents ahead of / behind the CoG (set from the vehicle by ``elk_reference``; BMW 320i: 2.06 / 2.32 m)
    _front: float = field(default=2.06, repr=False)
    _rear: float = field(default=2.32, repr=False)

    @property
    def lane_widths(self) -> tuple[float, float, float]:
        return tuple(f * self.vehicle_width + 0.25 for f in self.width_factors)

    def lanes(self):
        """[(x_start, x_end, centre_y, width)] of the three lanes."""
        s, w = self.sections, self.lane_widths
        a = self.x0
        return [(a, a + s[0], 0.0, w[0]), (a + s[0] + s[1], a + s[0] + s[1] + s[2], self.offset, w[1]),
                (a + sum(s[:4]), a + sum(s), 0.0, w[2])]

    def cones(self) -> list[tuple[float, float]]:
        """Cone positions: both sides of each lane at its start and end, and every 3 m along it."""
        out = []
        for xa, xb, c, w in self.lanes():
            n = max(2, int(round((xb - xa) / 3.0)) + 1)
            for k in range(n):
                x = xa + (xb - xa) * k / (n - 1)
                out += [(x, c - w / 2), (x, c + w / 2)]
        return out

    def hits(self, x, y, yaw) -> list[int]:
        """Per step, the number of lanes the body footprint sticks out of (corners outside the lane's cones while
        inside its length). x, y, yaw: sequences of the CoG pose."""
        half = self.vehicle_width / 2
        front, rear = self._front, self._rear
        lanes = self.lanes()
        out = []
        for px, py, ph in zip(x, y, yaw):
            c, s = math.cos(ph), math.sin(ph)
            n = 0
            for xa, xb, cy, w in lanes:
                bad = False
                for u, v in ((front, half), (front, -half), (-rear, half), (-rear, -half)):
                    cx_, cy_ = px + c * u - s * v, py + s * u + c * v
                    if xa <= cx_ <= xb and abs(cy_ - cy) > w / 2:
                        bad = True
                n += bad
            out.append(n)
        return out



def _ideal_library(cfg: VehicleEnvCfg, tr: "ElkTrack", v: float, steps: int, t_edges, mu: float, p: VehicleParams, device) -> ReferenceLibrary:
    """The textbook lane-centre path at constant speed as a reference (kinematic quantities from the path)."""
    T1, T2, h = tr.sections[1] / v, tr.sections[3] / v, tr.offset
    a = 0.5 * h
    t = torch.arange(steps, dtype=torch.float64) * cfg.dt
    y, dy, ddy = (torch.zeros_like(t) for _ in range(3))
    m1 = (t >= t_edges[1]) & (t < t_edges[2])
    u1 = (t - t_edges[1]) / T1
    y[m1], dy[m1], ddy[m1] = a * (1 - torch.cos(math.pi * u1[m1])), a * (math.pi / T1) * torch.sin(math.pi * u1[m1]), a * (math.pi / T1) ** 2 * torch.cos(math.pi * u1[m1])
    m2 = (t >= t_edges[2]) & (t < t_edges[3])
    y[m2] = h
    m3 = (t >= t_edges[3]) & (t < t_edges[4])
    u3 = (t - t_edges[3]) / T2
    y[m3], dy[m3], ddy[m3] = a * (1 + torch.cos(math.pi * u3[m3])), -a * (math.pi / T2) * torch.sin(math.pi * u3[m3]), -a * (math.pi / T2) ** 2 * torch.cos(math.pi * u3[m3])
    yaw = torch.atan2(dy, torch.full_like(dy, v))
    r = ddy / v / (1 + (dy / v) ** 2)
    f = lambda x: x.float().to(device)  # noqa: E731
    z = torch.zeros(steps, device=device)
    delta = f(p.L * r / v)
    return ReferenceLibrary(x=f(t * v), y=f(y), yaw=f(yaw), v=torch.full((steps,), v, device=device), beta=z.clone(), r=f(r), vx=torch.full((steps,), v, device=device),
                            vy=z.clone(), delta=delta, a_act=z.clone(), actions=torch.stack([torch.atanh((delta / cfg.steer_scale).clamp(-0.95, 0.95)), z], -1),
                            lengths=(steps,), util=f(v * v * r.abs() / (mu * 9.81)), seq_mu=(float(mu),))


@torch.no_grad()
def elk_reference(cfg: VehicleEnvCfg, speed: float, track: ElkTrack | None = None, mu: float = 1.0, device="cpu",
                  params: VehicleParams | None = None, max_beta: float = 0.2, max_dev: float = 0.15):
    """-> (``ReferenceLibrary`` of one trajectory, ``ElkTrack`` with ``x0`` set): the trajectory the privileged NMPC
    drives along the lane-centre path at ``speed`` [m/s] on road friction ``mu`` (needs casadi). Raises if it cannot follow
    the path to ``max_dev`` metres, or ends the run (a termination), or needs more sideslip than ``max_beta``."""
    from supermimic.baselines.nmpc import NMPC, NMPCCfg
    from supermimic.vehicles.dynamics import AX, AY, R, VY
    from supermimic.vehicles.env import VehicleTrackingEnv

    tr0 = track or ElkTrack()
    tr = ElkTrack(**{k: getattr(tr0, k) for k in ("sections", "offset", "width_factors", "vehicle_width", "front_overhang", "rear_overhang", "lead", "tail")})
    p = params or VehicleParams()
    tr._front, tr._rear = p.a + tr.front_overhang, p.b + tr.rear_overhang
    v = float(speed)
    t_edges = [tr.lead]
    for L in tr.sections:
        t_edges.append(t_edges[-1] + L / v)
    steps = int(round((t_edges[-1] + tr.tail) / cfg.dt))
    ideal = _ideal_library(cfg, tr, v, steps, t_edges, mu, p, device)
    env_cfg_ = VehicleEnvCfg(num_envs=1, episode_steps=10_000, replay_capacity=2, domain_randomization=False, reset_noise=False, match_friction=True,
                             assist=cfg.assist, preview=cfg.preview, tracking=cfg.tracking, substeps=cfg.substeps, dt=cfg.dt)
    env = VehicleTrackingEnv(env_cfg_, ideal, device=device, params=p, record=False)
    env.forced_xi = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    env.forced_frame[:] = ideal.layout.seq_start[:1]
    env.reset()
    ctl = NMPC(env_cfg_, p, NMPCCfg(information="privileged"))
    rows = {k: [] for k in ("x", "y", "yaw", "v", "beta", "r", "vx", "vy", "delta", "a_act", "actions", "util")}
    for k in range(steps):
        a = ctl(env) if k < steps - 1 else torch.zeros(1, 2, device=device)
        s = env.s[0]
        for key, val in (("x", s[0]), ("y", s[1]), ("yaw", s[2]), ("v", torch.hypot(s[3], s[VY])), ("beta", torch.atan2(s[VY], s[3].clamp_min(0.1))), ("r", s[R]),
                         ("vx", s[3]), ("vy", s[VY]), ("delta", s[6]), ("a_act", env.act.a_act[0]), ("util", torch.hypot(s[AX], s[AY]) / (mu * 9.81))):
            rows[key].append(val.clone())
        rows["actions"].append(a[0].clone())
        if k < steps - 1:
            _o, _r, term, _tr, info = env.step(a)
            if info["done"][0] and k < steps - 2:
                raise RuntimeError(f"elk test at {speed * 3.6:.0f} km/h, mu {mu}: the NMPC leaves the track at step {k}")
    out = {key: torch.stack(val).to(device) for key, val in rows.items()}
    dev = float((out["y"] - ideal.y).abs().max())
    if float(out["beta"].abs().max()) >= max_beta:
        raise RuntimeError(f"elk test at {speed * 3.6:.0f} km/h, mu {mu}: needs sideslip {float(out['beta'].abs().max()):.2f}")
    if dev > max_dev:
        raise RuntimeError(f"elk test at {speed * 3.6:.0f} km/h, mu {mu}: even the NMPC follows the lane-centre path only to {dev:.2f} m")
    tr.x0 = float(out["x"][int(round(tr.lead / cfg.dt))])
    return ReferenceLibrary(**{k: out[k] for k in ReferenceLibrary._FIELDS}, lengths=(steps,), util=out["util"], seq_mu=(float(mu),)), tr
