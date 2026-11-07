"""Batched double-track vehicle dynamics (torch; CPU or GPU, one row per environment).

State (``STATE_DIM`` = 17), CommonRoad sign conventions (roll positive = left side down, pitch positive
= nose up, so a left turn rolls negative and acceleration pitches positive):

    x, y, yaw           pose of the CoG in the world frame
    vx, vy, r           body-frame velocity and yaw rate
    delta               front-wheel steering angle
    roll, roll_rate, pitch, pitch_rate     sprung-mass attitude (drives the load transfer)
    w_lf, w_rf, w_lr, w_rr                 wheel spin rates (WHEELS)
    ax, ay              body-frame accelerations of the previous substep (unsprung load transfer)

Inputs per control step, as CommonRoad's: front-wheel steering rate and longitudinal acceleration
request; the request becomes engine / brake torques split per axle like CommonRoad's multi-body model.

Model (what differs from CommonRoad's 29-state multi-body model, which ``tests/test_vehicle_sim.py``
compares against):
* tyres: the same PAC2002 combined-slip equations (``tire.py``), camber from roll and pitch included;
* load transfer: sprung-mass roll and pitch as second-order systems about axes at the ground, with the
  destabilising moment of gravity on the displaced sprung CoG; the roll stiffness of each axle is its
  suspension stiffness (springs and auxiliary torsion) in series with the tyres' vertical stiffness, and
  the suspension moment carries the load transfer; unsprung masses transfer load quasi-statically; no
  heave, no compliant joints;
* wheels: spin dynamics integrated semi-implicitly (the slip ODE is stiff: its time constant is a few ms
  at 20 m/s and shrinks with speed); slip and slip angles use speeds floored at ``V_EPS`` instead of
  CommonRoad's switch to a kinematic model below 0.1 m/s;
* optional ABS (per-wheel brake release on excessive slip, with hysteresis) and ESC (single-wheel braking
  on yaw-rate error, with hysteresis): switching logic, deliberately non-differentiable.

Per-environment sim context (``Context``): road friction scale, tyre stiffness scale, mass scale and CoG
shift; everything else comes from ``VehicleParams``.

Optional chassis actuators (``chassis``, (n, 3), held over the control step; VEHICLE_PROPOSAL.md §10): rear-wheel
angle [rad], yaw moment [N m] from a left / right drive-torque difference on the rear axle, and an offset of the
front axle's share of the roll stiffness (the total stays the same, so the offset moves lateral load transfer
between the axles). ``chassis=None`` is the plain vehicle, bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.vehicles import tire as tyre
from supermimic.vehicles.params import VehicleParams

G = 9.81
V_EPS = 1.0  # [m/s] floor of the wheel / contact speeds in slip denominators

# Wheel order and names follow CommonRoad's multi-body model, whose "left" wheels are the ones at
# y = -T/2 (their contact speed is vx + T/2 r): the outer side of a left turn.
WHEELS = ("lf", "rf", "lr", "rr")

X, Y, YAW, VX, VY, R, DELTA, ROLL, ROLL_RATE, PITCH, PITCH_RATE = range(11)
W = slice(11, 15)
AX, AY = 15, 16
STATE_DIM = 17
STATE_NAMES = ("x", "y", "yaw", "vx", "vy", "r", "delta", "roll", "roll_rate", "pitch", "pitch_rate", "w_lf", "w_rf", "w_lr", "w_rr", "ax", "ay")


@dataclass
class Context:
    """Per-environment physical variation (the sim context xi), tensors of shape (n,)."""

    mu: torch.Tensor  # road friction scale on the tyres' peak coefficients
    tire_stiffness: torch.Tensor  # scale on the tyres' slip stiffness
    mass: torch.Tensor  # scale on sprung mass and inertias
    cog_shift: torch.Tensor  # [m] forward shift of the CoG (a -= shift, b += shift)

    @classmethod
    def nominal(cls, n: int, device="cpu", dtype=torch.float32) -> Context:
        one, zero = torch.ones(n, device=device, dtype=dtype), torch.zeros(n, device=device, dtype=dtype)
        return cls(one.clone(), one.clone(), one.clone(), zero)

    def index(self, ids) -> Context:
        return Context(self.mu[ids], self.tire_stiffness[ids], self.mass[ids], self.cog_shift[ids])


@dataclass
class AssistCfg:
    abs: bool = True
    abs_slip_on: float = 0.15  # release the brake above this longitudinal slip
    abs_slip_off: float = 0.06  # re-apply below it
    abs_release: float = 0.1  # brake torque factor while released
    abs_min_speed: float = 2.0
    esc: bool = True
    esc_err_on: float = 0.08  # [rad/s] yaw-rate error that activates ESC
    esc_err_off: float = 0.04
    esc_gain: float = 4000.0  # [N m per rad/s]
    esc_torque_max: float = 1500.0  # [N m]
    esc_min_speed: float = 5.0
    esc_char_speed: float = 22.0  # [m/s] characteristic speed of the reference yaw rate


class VehicleDynamics:
    """One control step = ``substeps`` semi-implicit Euler substeps of ``dt / substeps``."""

    def __init__(self, params: VehicleParams | None = None, dt: float = 0.02, substeps: int = 20, assist: AssistCfg | None = None):
        self.p = params or VehicleParams()
        self.dt, self.substeps = dt, substeps
        self.h = dt / substeps
        self.assist = assist or AssistCfg()
        p = self.p
        # Roll stiffness per axle: suspension (springs + auxiliary torsion, K_ts < 0 adds stiffness in
        # CommonRoad's convention) in series with the tyres' vertical stiffness.
        susp_f = p.K_sf * p.T_f**2 / 2 - p.K_tsf
        susp_r = p.K_sr * p.T_r**2 / 2 - p.K_tsr
        self.tyre_roll_f = p.K_zt * p.T_f**2 / 2
        self.tyre_roll_r = p.K_zt * p.T_r**2 / 2
        self.k_roll_f = 1 / (1 / susp_f + 1 / self.tyre_roll_f)
        self.k_roll_r = 1 / (1 / susp_r + 1 / self.tyre_roll_r)
        self.c_roll_f = p.K_sdf * p.T_f**2 / 2
        self.c_roll_r = p.K_sdr * p.T_r**2 / 2
        kf = 1 / (1 / (2 * p.K_sf) + 1 / (2 * p.K_zt))  # axle vertical stiffness
        kr = 1 / (1 / (2 * p.K_sr) + 1 / (2 * p.K_zt))
        self._k_pitch_axle = (kf, kr)
        self.c_pitch_axle = (2 * p.K_sdf, 2 * p.K_sdr)

    # ------------------------------------------------------------------ helpers

    def initial_state(self, vx, vy=None, r=None, delta=None, x=None, y=None, yaw=None) -> torch.Tensor:
        """Steady straight-line-like start: wheels rolling at vx / R, no roll / pitch."""
        vx = torch.as_tensor(vx)
        n = vx.shape[0]
        s = torch.zeros(n, STATE_DIM, dtype=vx.dtype, device=vx.device)
        s[:, VX] = vx
        for i, v in ((VY, vy), (R, r), (DELTA, delta), (X, x), (Y, y), (YAW, yaw)):
            if v is not None:
                s[:, i] = v
        s[:, W] = (vx / self.p.R_w)[:, None]
        return s

    def _geometry(self, ctx: Context):
        p = self.p
        a = p.a - ctx.cog_shift
        b = p.b + ctx.cog_shift
        m_s = p.m_s * ctx.mass
        m = m_s + p.m_uf + p.m_ur
        kf, kr = self._k_pitch_axle
        return a, b, m_s, m, kf * a**2 + kr * b**2, self.c_pitch_axle[0] * a**2 + self.c_pitch_axle[1] * b**2

    # ------------------------------------------------------------------ forces

    def tyre_state(self, s: torch.Tensor, ctx: Context, chassis: torch.Tensor | None = None):
        """Vertical loads, slips, slip angles, camber and the combined-slip tyre forces, (n, 4) each (LF, RF, LR, RR).
        ``chassis``: see the module docstring (rear-wheel angle and roll-stiffness split are used here)."""
        p = self.p
        a, b, m_s, _m, k_pitch, c_pitch = self._geometry(ctx)
        L = a + b
        vx, vy, r, delta = s[:, VX], s[:, VY], s[:, R], s[:, DELTA]
        roll, roll_rate, pitch, pitch_rate = s[:, ROLL], s[:, ROLL_RATE], s[:, PITCH], s[:, PITCH_RATE]
        ax, ay = s[:, AX], s[:, AY]

        # static loads, then longitudinal (pitch + unsprung) and lateral (roll + unsprung) transfer
        fz0_f = (m_s * G * b / L + p.m_uf * G) / 2
        fz0_r = (m_s * G * a / L + p.m_ur * G) / 2
        d_long = (k_pitch * pitch + c_pitch * pitch_rate) / L + (p.m_uf + p.m_ur) * ax * p.R_w / L  # off the front axle
        # roll < 0 (left turn) loads wheels 0 / 2, which sit at y = -T/2 (see WHEELS)
        if chassis is None:
            k_f, k_r = self.k_roll_f, self.k_roll_r
        else:  # active anti-roll: shift the front axle's share of the (unchanged) total roll stiffness
            k_tot = self.k_roll_f + self.k_roll_r
            share = (self.k_roll_f / k_tot + chassis[:, 2]).clamp(0.05, 0.95)
            k_f, k_r = share * k_tot, (1 - share) * k_tot
        d_lat_f = (k_f * roll + self.c_roll_f * roll_rate) / p.T_f - p.m_uf * ay * p.R_w / p.T_f
        d_lat_r = (k_r * roll + self.c_roll_r * roll_rate) / p.T_r - p.m_ur * ay * p.R_w / p.T_r
        fz = torch.stack([fz0_f - d_long / 2 - d_lat_f, fz0_f - d_long / 2 + d_lat_f, fz0_r + d_long / 2 - d_lat_r, fz0_r + d_long / 2 + d_lat_r], -1).clamp_min(1.0)

        # camber (CommonRoad's suspension kinematics, heave neglected); unsprung roll from tyre compliance
        droll_f = roll - self.k_roll_f * roll / self.tyre_roll_f
        droll_r = roll - self.k_roll_r * roll / self.tyre_roll_r
        gamma = torch.stack([
            roll + p.D_f * (0.5 * droll_f * p.T_f + a * pitch),
            roll + p.D_f * (0.5 * droll_f * p.T_f - a * pitch),
            roll + p.D_r * (0.5 * droll_r * p.T_r - b * pitch),
            roll + p.D_r * (0.5 * droll_r * p.T_r + b * pitch),
        ], -1)

        cd, sd = torch.cos(delta), torch.sin(delta)
        vy_f, vy_r = vy + a * r, vy - b * r
        vx_l_f, vx_r_f = vx + 0.5 * p.T_f * r, vx - 0.5 * p.T_f * r
        vx_l_r, vx_r_r = vx + 0.5 * p.T_r * r, vx - 0.5 * p.T_r * r
        if chassis is None:
            u = torch.stack([vx_l_f * cd + vy_f * sd, vx_r_f * cd + vy_f * sd, vx_l_r, vx_r_r], -1).clamp_min(V_EPS)
            dr = 0.0
        else:  # rear-wheel steering
            dr = chassis[:, 0]
            cr, sr = torch.cos(dr), torch.sin(dr)
            u = torch.stack([vx_l_f * cd + vy_f * sd, vx_r_f * cd + vy_f * sd, vx_l_r * cr + vy_r * sr, vx_r_r * cr + vy_r * sr], -1).clamp_min(V_EPS)
        kappa = 1 - p.R_w * s[:, W] / u
        alpha = torch.stack([
            torch.atan(vy_f / vx_l_f.clamp_min(V_EPS)) - delta,
            torch.atan(vy_f / vx_r_f.clamp_min(V_EPS)) - delta,
            torch.atan(vy_r / vx_l_r.clamp_min(V_EPS)) - dr,
            torch.atan(vy_r / vx_r_r.clamp_min(V_EPS)) - dr,
        ], -1)
        tp = p.tire
        mu = ctx.mu[:, None]
        ks = ctx.tire_stiffness[:, None]
        fx0 = tyre.longitudinal(kappa, gamma, fz, tp, mu, ks)
        fy0, mu_y = tyre.lateral(alpha, gamma, fz, tp, mu, ks)
        fx = tyre.longitudinal_comb(kappa, alpha, fx0, tp)
        fy = tyre.lateral_comb(kappa, alpha, gamma, mu_y, fz, fy0, tp)
        return {"fz": fz, "kappa": kappa, "alpha": alpha, "gamma": gamma, "u": u, "fx": fx, "fy": fy}

    # ------------------------------------------------------------------ inputs

    def constrain(self, s, steer_rate, accel):
        """CommonRoad's steering and acceleration constraints."""
        p = self.p
        d = s[:, DELTA]
        at_limit = ((d <= -p.steer_max) & (steer_rate <= 0)) | ((d >= p.steer_max) & (steer_rate >= 0))
        steer_rate = torch.where(at_limit, torch.zeros_like(steer_rate), steer_rate.clamp(-p.steer_rate_max, p.steer_rate_max))
        vx = s[:, VX]
        pos_limit = torch.where(vx > p.v_switch, p.a_max * p.v_switch / vx.clamp_min(1e-3), torch.full_like(vx, p.a_max))
        accel = torch.minimum(accel.clamp(-p.a_max, None), pos_limit)
        accel = torch.where((vx >= p.v_max) & (accel >= 0), torch.zeros_like(accel), accel)
        return steer_rate, accel

    def wheel_torques(self, accel, m):
        """Engine and brake torques from the acceleration request, split per axle as CommonRoad does: (n, 4) each."""
        p = self.p
        t = m * p.R_w * accel
        t_e, t_b = t.clamp_min(0.0), t.clamp_max(0.0)
        drive = torch.stack([0.5 * p.T_se * t_e, 0.5 * p.T_se * t_e, 0.5 * (1 - p.T_se) * t_e, 0.5 * (1 - p.T_se) * t_e], -1)
        brake = torch.stack([0.5 * p.T_sb * t_b, 0.5 * p.T_sb * t_b, 0.5 * (1 - p.T_sb) * t_b, 0.5 * (1 - p.T_sb) * t_b], -1)
        return drive, brake

    # ------------------------------------------------------------------ assists (non-differentiable)

    def _abs(self, brake, ts, s, flags):
        cfg = self.assist
        slip = ts["kappa"]
        on = torch.where(flags[:, 0:4] > 0.5, slip > cfg.abs_slip_off, slip > cfg.abs_slip_on)
        on &= (brake < 0) & (s[:, VX] > cfg.abs_min_speed)[:, None]
        return torch.where(on, brake * cfg.abs_release, brake), on.to(flags.dtype)

    def _esc(self, s, flags, ctx, a, b):
        cfg = self.assist
        vx, r, delta = s[:, VX], s[:, R], s[:, DELTA]
        r_ref = vx * delta / ((a + b) * (1 + (vx / cfg.esc_char_speed) ** 2))
        r_cap = 0.85 * ctx.mu * G / vx.clamp_min(1.0)
        r_ref = torch.maximum(torch.minimum(r_ref, r_cap), -r_cap)
        err = r - r_ref
        on = torch.where(flags[:, 4] > 0.5, err.abs() > cfg.esc_err_off, err.abs() > cfg.esc_err_on) & (vx > cfg.esc_min_speed)
        over = err * torch.where(r_ref.abs() > 1e-3, r_ref, r) > 0  # turning more than the reference: oversteer
        left = torch.where(r_ref.abs() > 1e-3, r_ref, r) > 0  # turning left
        # oversteer: brake the outer front wheel; understeer: brake the inner rear wheel (wheels 0 / 2 are
        # the outer side of a left turn, see WHEELS)
        wheel = torch.where(over, torch.where(left, 0, 1), torch.where(left, 3, 2))
        torque = -(cfg.esc_gain * err.abs()).clamp_max(cfg.esc_torque_max) * on
        extra = torch.zeros(s.shape[0], 4, dtype=s.dtype, device=s.device).scatter(1, wheel[:, None], torque[:, None])
        return extra, on.to(flags.dtype)

    # ------------------------------------------------------------------ integration

    def substep(self, s, flags, steer_rate, accel, ctx: Context, chassis: torch.Tensor | None = None):
        p, h = self.p, self.h
        a, b, m_s, m, k_pitch, c_pitch = self._geometry(ctx)
        I_z = p.I_z * ctx.mass
        I_roll = (p.I_phi_s + p.m_s * p.h_s**2) * ctx.mass
        I_pitch = (p.I_y_s + p.m_s * p.h_s**2) * ctx.mass
        ts = self.tyre_state(s, ctx, chassis)
        drive, brake = self.wheel_torques(accel, m)
        if chassis is not None:  # torque vectoring: +dT on wheel 2, -dT on wheel 3 gives the yaw moment (see sum_n)
            dT = chassis[:, 1] * p.R_w / p.T_r
            drive = drive + torch.stack([torch.zeros_like(dT), torch.zeros_like(dT), dT, -dT], -1)
        flags = flags.clone()
        if self.assist.abs:
            brake, flags[:, 0:4] = self._abs(brake, ts, s, flags)
        if self.assist.esc:
            extra, flags[:, 4] = self._esc(s, flags, ctx, a, b)
            brake = brake + extra
        fx, fy = ts["fx"], ts["fy"]
        delta = s[:, DELTA]
        cd, sd = torch.cos(delta), torch.sin(delta)
        if chassis is None:
            fx_r, fy_r = fx[:, 2:], fy[:, 2:]
        else:  # rear wheel forces into the body frame
            cr, sr = torch.cos(chassis[:, 0:1]), torch.sin(chassis[:, 0:1])
            fx_r, fy_r = fx[:, 2:] * cr - fy[:, 2:] * sr, fy[:, 2:] * cr + fx[:, 2:] * sr
        sum_x = fx_r[:, 0] + fx_r[:, 1] + (fx[:, 0] + fx[:, 1]) * cd - (fy[:, 0] + fy[:, 1]) * sd
        sum_y = (fy[:, 0] + fy[:, 1]) * cd + (fx[:, 0] + fx[:, 1]) * sd + fy_r[:, 0] + fy_r[:, 1]
        sum_n = ((fy[:, 0] + fy[:, 1]) * a * cd + (fx[:, 0] + fx[:, 1]) * a * sd
                 + (fy[:, 1] - fy[:, 0]) * 0.5 * p.T_f * sd + (fx[:, 0] - fx[:, 1]) * 0.5 * p.T_f * cd
                 + (fx_r[:, 0] - fx_r[:, 1]) * 0.5 * p.T_r - (fy_r[:, 0] + fy_r[:, 1]) * b)
        ax, ay = sum_x / m, sum_y / m
        vx, vy, r = s[:, VX], s[:, VY], s[:, R]
        roll, roll_rate, pitch, pitch_rate = s[:, ROLL], s[:, ROLL_RATE], s[:, PITCH], s[:, PITCH_RATE]
        k_roll, c_roll = self.k_roll_f + self.k_roll_r, self.c_roll_f + self.c_roll_r

        out = s.clone()
        out[:, VX] = vx + h * (ax + r * vy)
        out[:, VY] = vy + h * (ay - r * vx)
        out[:, R] = r + h * sum_n / I_z
        # roll / pitch about axes at the ground; gravity acting on the displaced sprung CoG destabilises both
        g_moment = m_s * G * p.h_s
        out[:, ROLL_RATE] = roll_rate + h * (-m_s * ay * p.h_s - (k_roll - g_moment) * roll - c_roll * roll_rate) / I_roll
        out[:, PITCH_RATE] = pitch_rate + h * (m_s * ax * p.h_s - (k_pitch - g_moment) * pitch - c_pitch * pitch_rate) / I_pitch
        # wheels: semi-implicit in the slip stiffness (secant slope of Fx over slip, a stable damping term)
        w = s[:, W]
        torque = drive + brake - p.R_w * fx
        slope = fx.abs() / ts["kappa"].abs().clamp_min(1e-3)
        w_new = w + h * torque / p.I_y_w / (1 + h * p.R_w**2 * slope / (p.I_y_w * ts["u"]))
        out[:, W] = w_new.clamp_min(0.0)  # negative wheel spin forbidden (as in CommonRoad)
        # positions with the updated velocities
        yaw = s[:, YAW]
        out[:, X] = s[:, X] + h * (out[:, VX] * torch.cos(yaw) - out[:, VY] * torch.sin(yaw))
        out[:, Y] = s[:, Y] + h * (out[:, VX] * torch.sin(yaw) + out[:, VY] * torch.cos(yaw))
        out[:, YAW] = yaw + h * out[:, R]
        out[:, ROLL] = roll + h * out[:, ROLL_RATE]
        out[:, PITCH] = pitch + h * out[:, PITCH_RATE]
        out[:, DELTA] = (delta + h * steer_rate).clamp(-p.steer_max, p.steer_max)
        out[:, AX], out[:, AY] = ax, ay
        return out, flags

    def step(self, s, flags, steer_rate, accel, ctx: Context, chassis: torch.Tensor | None = None):
        """One control step with the inputs held (zero-order hold). ``flags`` (n, 5): ABS per wheel, ESC.
        ``chassis`` (n, 3) or None: see the module docstring."""
        for _ in range(self.substeps):
            sr, ac = self.constrain(s, steer_rate, accel)
            s, flags = self.substep(s, flags, sr, ac, ctx, chassis)
        return s, flags

    def new_flags(self, n: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
        return torch.zeros(n, 5, device=device, dtype=dtype)
