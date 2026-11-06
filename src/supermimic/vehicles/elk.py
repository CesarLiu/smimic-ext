"""Elk test (moose test): the ISO 3888-2 obstacle-avoidance lane change, as a reference and a set of cones.

Layout along the track, from the first gate (ISO 3888-2:2011, as reproduced in secondary sources; the standard
itself was not available):

=========  ==========  ==========================================  =========================================
section    length [m]  lane width                                   position
=========  ==========  ==========================================  =========================================
1 entry    12          1.1 b + 0.25                                 centred on y = 0
2          13.5        (transition, no lane)
3 avoid    11          b + 1                                        to the left, 1 m gap to lane 1
4          12.5        (transition, no lane)
5 exit     12          1.3 b + 0.25, at least 3                     right edge in line with lane 1 (see below)
=========  ==========  ==========================================  =========================================

b is the vehicle width; the total is 61 m. The "1 m" lateral offset is read as the gap between lane 1's left
boundary and lane 3's right one, which puts lane 3's centre 1.1b/2 + 0.125 + 1 + (b + 1)/2 = 3.5 m (b = 1.8) to the
left; this reading matches a secondary source's "3.5" and the speeds real cars reach (pass benchmark 72 km/h,
record about 91 km/h), whereas a 1 m centre-to-centre offset would be passable far beyond them. Not verified against the
standard text: the reading of the offset, and how the exit lane is aligned (``exit_align``: its right boundary in line
with lane 1's, default, or its centre).

The standard's procedure (enter in the highest gear that holds the speed, release the throttle after the entry) is
replaced by a constant reference speed. Scoring follows its pass criterion, no cone displaced: the track limits are
tangent to the cones' base circles, so each cone sits ``cone_radius`` outside a lane boundary, and a cone counts as hit
when the body footprint (a rectangle) touches its base circle. The cone spacing along the lanes (about 3 m) and the
base radius (0.15 m) are assumptions. ``overhang`` is a stricter secondary measure: steps with any body corner outside a
lane while inside that lane's length.

The reference (``elk_reference``) follows ``best_line``: the path through the lanes with the least peak curvature,
a small optimisation over the CoG path with the body corners kept inside the lanes (small-angle approximation). The
car cannot follow even that line beyond some speed; the reference is what the privileged NMPC baseline
(``supermimic.baselines.nmpc``) drives along it under the nominal context, feasible by construction, and a speed only
counts as a case if that drive passes the cones itself. Both need casadi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from supermimic.vehicles.env import ReferenceLibrary, VehicleEnvCfg
from supermimic.vehicles.params import VehicleParams


@dataclass
class ElkTrack:
    sections: tuple[float, float, float, float, float] = (12.0, 13.5, 11.0, 12.5, 12.0)
    gap: float = 1.0  # lateral gap between lane 1 and lane 3 [m]
    vehicle_width: float = 1.8
    front_overhang: float = 0.9  # body beyond the front axle [m]
    rear_overhang: float = 0.9
    exit_align: str = "right_edge"
    cone_radius: float = 0.15  # base radius of the cones [m] (assumed)
    cone_spacing: float = 3.0  # cones along each lane about this far apart [m] (assumed)  # exit lane: "right_edge" in line with lane 1's right boundary, or "centre"
    lead: float = 1.0  # [s] driven before the first gate
    tail: float = 1.5  # [s] after the last
    x0: float = 0.0  # x of the first gate, set by ``elk_reference``
    clearance: float = float("nan")  # the reference line's clearance to lanes and cones [m], set by ``elk_reference``
    # body extents ahead of / behind the CoG (set from the vehicle by ``elk_reference``; BMW 320i: 2.06 / 2.32 m)
    _front: float = field(default=2.06, repr=False)
    _rear: float = field(default=2.32, repr=False)

    @property
    def lane_widths(self) -> tuple[float, float, float]:
        b = self.vehicle_width
        return 1.1 * b + 0.25, b + 1.0, max(1.3 * b + 0.25, 3.0)

    def lanes(self):
        """[(x_start, x_end, centre_y, width)] of the entry, avoidance and exit lanes."""
        s, (w1, w2, w3) = self.sections, self.lane_widths
        a = self.x0
        c2 = w1 / 2 + self.gap + w2 / 2
        c3 = -w1 / 2 + w3 / 2 if self.exit_align == "right_edge" else 0.0
        return [(a, a + s[0], 0.0, w1), (a + s[0] + s[1], a + s[0] + s[1] + s[2], c2, w2), (a + sum(s[:4]), a + sum(s), c3, w3)]

    def cones(self, sides: bool = False) -> list[tuple]:
        """Cone centres: both sides of each lane at its start and end and about every ``cone_spacing`` along it, each a
        base radius outside the boundary (the track limit is tangent to the base circle). ``sides``: also +1 for a cone
        on the left of its lane, -1 on the right."""
        out = []
        r = self.cone_radius
        for xa, xb, c, w in self.lanes():
            n = max(2, int(round((xb - xa) / self.cone_spacing)) + 1)
            for k in range(n):
                x = xa + (xb - xa) * k / (n - 1)
                out += [(x, c - w / 2 - r, -1), (x, c + w / 2 + r, 1)]
        return out if sides else [(x, y) for x, y, _s in out]

    def cone_hits(self, x, y, yaw) -> list[set]:
        """Per step, the indices (into ``cones()``) of the cones whose base circle the body footprint touches."""
        half, front, rear, r = self.vehicle_width / 2, self._front, self._rear, self.cone_radius
        cones = np.asarray(self.cones())
        out = []
        for px, py, ph in zip(x, y, yaw):
            c, s = math.cos(ph), math.sin(ph)
            dx, dy = cones[:, 0] - px, cones[:, 1] - py
            u, v = c * dx + s * dy, -s * dx + c * dy  # cone centre in the body frame
            du = np.maximum(np.maximum(-rear - u, u - front), 0.0)
            dv = np.maximum(np.abs(v) - half, 0.0)
            out.append(set(np.nonzero(np.hypot(du, dv) < r)[0].tolist()))
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


def best_line(tr: ElkTrack, ds: float = 0.25, before: float = 25.0, after: float = 30.0, margin: float = 0.05, kappa_max: float | None = None):
    """A CoG path y(x) through the lanes, x measured from ``before`` metres ahead of the first gate (``tr.x0`` is not
    used); straight with zero slope at both ends. Every body corner stays inside the lane whose length it is in, and the
    body's sides stay clear of every cone's base circle, by a clearance (small-angle geometry: a point of the body at
    distance u ahead of the CoG is at CoG + slope * u).

    * ``kappa_max`` None: the least peak curvature with the clearance fixed at ``margin`` (the line the fastest car
      would need; its peak curvature gives the point-mass speed limit);
    * ``kappa_max`` given: the largest clearance with the curvature bounded by it (what a driver aims for at a given
      speed: as far from the cones as the grip allows).

    -> (x grid relative to the first gate, y, peak curvature [1/m], clearance [m])."""
    import casadi as ca

    xs = np.arange(-before, sum(tr.sections) + after + ds / 2, ds)
    n = len(xs)
    at0 = ElkTrack(**{**{k: getattr(tr, k) for k in ("sections", "gap", "vehicle_width", "front_overhang", "rear_overhang", "exit_align", "cone_radius", "cone_spacing")}, "x0": 0.0})
    lanes, cones = at0.lanes(), at0.cones(sides=True)
    opti = ca.Opti()
    y = opti.variable(n)
    peak = opti.variable()
    clear = opti.variable() if kappa_max is not None else margin
    curv = (y[2:] - 2 * y[1:-1] + y[:-2]) / ds**2
    slope = (y[2:] - y[:-2]) / (2 * ds)
    half = tr.vehicle_width / 2
    for i in range(1, n - 1):
        for u in (tr._front, -tr._rear):
            for xa, xb, c, w in lanes:
                if xa <= xs[i] + u <= xb:
                    for v in (half, -half):
                        corner = y[i] + slope[i - 1] * u + v
                        opti.subject_to(opti.bounded(c - w / 2 + clear, corner, c + w / 2 - clear))
        for xc, yc, side in cones:  # the body side passing a cone stays clear of its base circle
            u = xc - xs[i]
            if -tr._rear <= u <= tr._front:
                edge = y[i] + slope[i - 1] * u + side * half
                if side > 0:
                    opti.subject_to(edge <= yc - tr.cone_radius - clear)
                else:
                    opti.subject_to(edge >= yc + tr.cone_radius + clear)
    opti.subject_to(opti.bounded(-peak, curv, peak))
    opti.subject_to(y[0] == 0)
    opti.subject_to(y[1] == 0)
    opti.subject_to(y[-1] == y[-2])
    jerk = (curv[1:] - curv[:-1]) / ds
    if kappa_max is None:
        opti.minimize(peak + 1e-3 * ca.sumsqr(jerk) * ds)
    else:
        opti.subject_to(peak <= kappa_max)
        opti.subject_to(clear <= 0.5)
        opti.minimize(-clear + 1e-3 * ca.sumsqr(jerk) * ds)
    opti.set_initial(peak, 0.1)
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "sb": "yes", "max_iter": 3000})
    try:
        sol = opti.solve()
    except RuntimeError:  # infeasible within the curvature bound
        return xs, np.zeros(n), float("inf"), float("-inf")
    return xs, np.asarray(sol.value(y)).ravel(), float(sol.value(peak)), float(sol.value(clear)) if kappa_max is not None else margin


def _line_library(cfg: VehicleEnvCfg, v: float, steps: int, x_gate: float, line, mu: float, p: VehicleParams, device) -> ReferenceLibrary:
    """The best line driven at constant speed ``v`` as a reference (kinematic quantities from the path); the first
    gate is at x = ``x_gate``."""
    xs, ys = line[0], line[1]
    t = np.arange(steps) * cfg.dt
    x = v * t
    xr = x - x_gate
    y = np.interp(xr, xs, ys, left=0.0, right=ys[-1])
    dydx = np.gradient(np.interp(xr, xs, ys, left=0.0, right=ys[-1]), x)
    d2 = np.gradient(dydx, x)
    yaw = np.arctan(dydx)
    kappa = d2 / (1 + dydx**2) ** 1.5
    r = v * kappa
    f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    z = torch.zeros(steps, device=device)
    delta = f(p.L * kappa)
    return ReferenceLibrary(x=f(x), y=f(y), yaw=f(yaw), v=torch.full((steps,), v, device=device), beta=z.clone(), r=f(r), vx=torch.full((steps,), v, device=device),
                            vy=z.clone(), delta=delta, a_act=z.clone(), actions=torch.stack([torch.atanh((delta / cfg.steer_scale).clamp(-0.95, 0.95)), z], -1),
                            lengths=(steps,), util=f(v * v * np.abs(kappa) / (mu * 9.81)), seq_mu=(float(mu),))


@torch.no_grad()
def elk_reference(cfg: VehicleEnvCfg, speed: float, track: ElkTrack | None = None, mu: float = 1.0, device="cpu",
                  params: VehicleParams | None = None, max_beta: float = 0.2, sigma_lat: float = 0.1, ay_frac: float = 0.85, min_clearance: float = 0.02,
                  target_clearance: float = 0.10):
    """-> (``ReferenceLibrary`` of one trajectory, ``ElkTrack`` with ``x0`` and ``clearance`` set): the trajectory the
    privileged NMPC drives along the speed's line at ``speed`` [m/s] on road friction ``mu`` (needs casadi). The line:
    the largest clearance a lateral acceleration of ``ay_frac`` mu g allows is found first; the line is then the
    least-curvature one with the clearance at that or ``target_clearance``, whichever is smaller (gentle when there is
    room, at the grip budget near the limit). Raises if the clearance would be under ``min_clearance``, or if the drive
    is terminated, hits a cone itself, or needs more sideslip than ``max_beta``. ``sigma_lat``: the NMPC driving the reference weights lateral error
    with this scale (the environment's 0.5 m tolerates errors of several cm, too many for these lanes); heading and
    sideslip, whose references here are kinematic, are de-emphasised."""
    from supermimic.baselines.nmpc import NMPC, NMPCCfg
    from supermimic.vehicles.dynamics import AX, AY, R, VY
    from supermimic.vehicles.env import VehicleTrackingEnv

    tr0 = track or ElkTrack()
    tr = ElkTrack(**{k: getattr(tr0, k) for k in ("sections", "gap", "vehicle_width", "front_overhang", "rear_overhang", "exit_align", "cone_radius", "cone_spacing", "lead", "tail")})
    p = params or VehicleParams()
    tr._front, tr._rear = p.a + tr.front_overhang, p.b + tr.rear_overhang
    v = float(speed)
    best_clear = best_line(tr, kappa_max=ay_frac * mu * 9.81 / v**2)[3]
    if best_clear < min_clearance:
        raise RuntimeError(f"elk test at {speed * 3.6:.0f} km/h, mu {mu}: no line through the cones within {ay_frac} mu g")
    tr.clearance = min(best_clear, target_clearance)
    line = best_line(tr, margin=tr.clearance)
    x_gate = v * tr.lead
    steps = int(round((tr.lead + sum(tr.sections) / v + tr.tail) / cfg.dt))
    ideal = _line_library(cfg, v, steps, x_gate, line, mu, p, device)
    env_cfg_ = VehicleEnvCfg(num_envs=1, episode_steps=10_000, replay_capacity=2, domain_randomization=False, reset_noise=False, match_friction=True,
                             assist=cfg.assist, preview=cfg.preview, tracking=cfg.tracking, substeps=cfg.substeps, dt=cfg.dt)
    env = VehicleTrackingEnv(env_cfg_, ideal, device=device, params=p, record=False)
    env.forced_xi = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    env.forced_frame[:] = ideal.layout.seq_start[:1]
    env.reset()
    from dataclasses import replace

    tight = replace(cfg.tracking, sigma_lat=sigma_lat, sigma_yaw=0.3, sigma_beta=1.0)
    ctl = NMPC(env_cfg_, p, NMPCCfg(information="privileged"), tracking=tight)
    rows = {k: [] for k in ("x", "y", "yaw", "v", "beta", "r", "vx", "vy", "delta", "a_act", "actions", "util")}
    for k in range(steps):
        a = ctl(env) if k < steps - 1 else torch.zeros(1, 2, device=device)
        s = env.s[0]
        for key, val in (("x", s[0]), ("y", s[1]), ("yaw", s[2]), ("v", torch.hypot(s[3], s[VY])), ("beta", torch.atan2(s[VY], s[3].clamp_min(0.1))), ("r", s[R]),
                         ("vx", s[3]), ("vy", s[VY]), ("delta", s[6]), ("a_act", env.act.a_act[0]), ("util", torch.hypot(s[AX], s[AY]) / (mu * 9.81))):
            rows[key].append(val.clone())
        rows["actions"].append(a[0].clone())
        if k < steps - 1:
            _o, _r, _term, _tr, info = env.step(a)
            if info["done"][0] and k < steps - 2:
                raise RuntimeError(f"elk test at {speed * 3.6:.0f} km/h, mu {mu}: the NMPC leaves the track at step {k}")
    out = {key: torch.stack(val).to(device) for key, val in rows.items()}
    tr.x0 = x_gate
    if float(out["beta"].abs().max()) >= max_beta:
        raise RuntimeError(f"elk test at {speed * 3.6:.0f} km/h, mu {mu}: needs sideslip {float(out['beta'].abs().max()):.2f}")
    x_end = tr.lanes()[-1][1]
    n_end = int((out["x"] <= x_end + 3.0).sum())
    hit = set().union(*tr.cone_hits(out["x"][:n_end].tolist(), out["y"][:n_end].tolist(), out["yaw"][:n_end].tolist()))
    if hit:
        raise RuntimeError(f"elk test at {speed * 3.6:.0f} km/h, mu {mu}: even the NMPC on the line (clearance {tr.clearance:.2f} m) hits {len(hit)} cone(s)")
    return ReferenceLibrary(**{k: out[k] for k in ReferenceLibrary._FIELDS}, lengths=(steps,), util=out["util"], seq_mu=(float(mu),)), tr
