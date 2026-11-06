"""NMPC baseline (VEHICLE_PROPOSAL.md §6, E6): nonlinear model-predictive tracking of the same references,
scored by the same environment, with a single-track (bicycle) prediction model and a simplified magic-formula
tyre per axle whose parameters are identified from the simulator.

What the controller knows, and how that differs from the learned policies (state this when reporting):

* the full simulator state (pose, body velocities, yaw rate, wheel angle, longitudinal actuator state) without
  noise; the policies see noisy on-board measurements and a short history;
* the road friction ``mu`` (``NMPCCfg.friction = "known"``; ``"nominal"`` gives it 1.0 instead); the policies
  have to infer it;
* the vehicle's nominal mass, inertia and geometry, and the actuator model (servo time constant, rate limit,
  longitudinal lag) exactly; no command delay is modelled (the evaluation runs with delay 0).

What it does not model (the plant/model mismatch): roll, pitch and load transfer, camber, combined slip,
wheel spin dynamics, the brake / drive split. ``identify_axle_tyres`` fits each axle's lateral force against
its slip angle on data from the full simulator, so load transfer is absorbed into the fitted curve on average.

Optimisation: multiple shooting over ``horizon`` intervals of ``interval`` control steps (RK4), quadratic
tracking cost on the reward's own error terms (lateral, longitudinal, heading, speed, sideslip, each weighted
w / sigma^2 as in ``TrackingCfg``), input-rate penalties, solved by IPOPT through CasADi, warm-started from the
shifted previous solution. Needs ``casadi`` (``pip install casadi``); imported lazily.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from supermimic.vehicles.dynamics import AX, AY, DELTA, VX, VY, YAW, Context, G, R, VehicleDynamics, X, Y
from supermimic.vehicles.env import VehicleEnvCfg
from supermimic.vehicles.params import VehicleParams
from supermimic.vehicles.tracking import TrackingCfg

NX, NU = 8, 2  # x, y, yaw, vx, vy, r, delta, a_act | delta_cmd, a_cmd
NREF = 5  # x, y, yaw, v, beta per node


@dataclass
class NMPCCfg:
    horizon: int = 25  # shooting intervals
    interval: int = 2  # control steps per interval (horizon * interval * dt = 1 s by default)
    rk4_substeps: int = 2
    friction: str = "known"  # "known": the true road friction; "nominal": 1.0
    w_steer_rate: float = 50.0  # on (delta_cmd_k - delta_cmd_{k-1}) [rad], per interval
    w_accel_rate: float = 0.05  # on (a_cmd_k - a_cmd_{k-1}) [m/s^2]
    terminal_scale: float = 2.0  # extra weight on the last node
    cmd_margin: float = 0.97  # commands stay within this share of the actuator range (tanh is inverted)
    max_iter: int = 60
    tol: float = 1e-4


@dataclass
class AxleTyre:
    """Lateral force of one axle against its slip angle: D sin(C atan(B a - E (B a - atan(B a)))) at mu = 1.
    Road friction scales the peak and leaves the cornering stiffness B C D unchanged (as in ``tire.lateral``)."""

    B: float
    C: float
    D: float
    E: float

    def force(self, alpha, mu=1.0, lib=np):
        b = self.B / mu
        x = b * alpha
        return mu * self.D * lib.sin(self.C * lib.atan(x - self.E * (x - lib.atan(x))))


# ---------------------------------------------------------------------- identification


def _axle_slip(s, a, b):
    vx = s[:, VX].clamp_min(1.0)
    alpha_f = torch.atan((s[:, VY] + a * s[:, R]) / vx) - s[:, DELTA]
    alpha_r = torch.atan((s[:, VY] - b * s[:, R]) / vx)
    return alpha_f, alpha_r


@torch.no_grad()
def excitation_data(params: VehicleParams | None = None, env_cfg: VehicleEnvCfg | None = None, n: int = 256, steps: int = 150, seed: int = 0):
    """Full-simulator data under random smooth steering at speeds 6-30 m/s, nominal context, mu = 1, no ABS/ESC:
    per sample the bicycle slip angles of both axles and the axles' summed lateral tyre forces (wheel frame)."""
    from supermimic.vehicles.dynamics import AssistCfg

    cfg = env_cfg or VehicleEnvCfg()
    dyn = VehicleDynamics(params, cfg.dt, cfg.substeps, AssistCfg(abs=False, esc=False))
    p = dyn.p
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *shape: torch.rand(*shape, generator=g)  # noqa: E731
    v0 = 6.0 + 24.0 * rnd(n)
    s = dyn.initial_state(v0)
    flags = dyn.new_flags(n)
    ctx = Context.nominal(n)
    amp, f, ph = 0.03 + 0.2 * rnd(n), 0.1 + 0.6 * rnd(n), 2 * math.pi * rnd(n)
    ok = torch.ones(n, dtype=torch.bool)
    rows = []
    for i in range(steps):
        target = (amp * torch.sin(2 * math.pi * f * i * cfg.dt + ph)).clamp(-p.steer_max, p.steer_max)
        steer_rate = ((target - s[:, DELTA]) / cfg.steer_tau).clamp(-p.steer_rate_max, p.steer_rate_max)
        s, flags = dyn.step(s, flags, steer_rate, torch.zeros(n), ctx)
        beta = torch.atan2(s[:, VY], s[:, VX].clamp_min(0.1))
        ok &= torch.isfinite(s).all(-1) & (beta.abs() < 0.3) & (s[:, VX] > 3.0)
        ts = dyn.tyre_state(s, ctx)
        af, ar = _axle_slip(s, p.a, p.b)
        rows.append(torch.stack([af, ar, ts["fy"][:, 0] + ts["fy"][:, 1], ts["fy"][:, 2] + ts["fy"][:, 3]], -1)[ok])
    return torch.cat(rows).double().numpy()


def fit_axle(alpha: np.ndarray, force: np.ndarray) -> AxleTyre:
    """Least-squares magic-formula fit, with C and E kept in the physical range (an unbounded fit can trade a
    tiny C against a huge D and lose the peak)."""
    from scipy.optimize import least_squares

    scale = float(np.abs(force).max())
    sign = float(np.sign(np.sum(alpha * force))) or 1.0
    small = np.abs(alpha) < 0.02
    k0 = float(np.polyfit(alpha[small], force[small], 1)[0]) if small.sum() > 10 else sign * 10 * scale

    def resid(q):
        b, c, dn, e = q
        return (AxleTyre(b, c, sign * dn * scale, e).force(alpha) - force) / scale

    x0 = [abs(k0) / (1.3 * scale), 1.3, 1.0, 0.0]
    res = least_squares(resid, x0, bounds=([0.1, 0.8, 0.2, -5.0], [100.0, 2.5, 2.0, 1.0]), max_nfev=2000)
    b, c, dn, e = map(float, res.x)
    return AxleTyre(b, c, sign * dn * scale, e)


def identify_axle_tyres(params: VehicleParams | None = None, env_cfg: VehicleEnvCfg | None = None, seed: int = 0) -> tuple[AxleTyre, AxleTyre, dict]:
    """Front and rear ``AxleTyre`` fitted to ``excitation_data``; also the fit's relative RMS error per axle."""
    d = excitation_data(params, env_cfg, seed=seed)
    front, rear = fit_axle(d[:, 0], d[:, 2]), fit_axle(d[:, 1], d[:, 3])
    rms = lambda t, a, f: float(np.sqrt(np.mean((t.force(a) - f) ** 2)) / np.abs(f).max())  # noqa: E731
    info = {"samples": int(d.shape[0]), "max_abs_alpha": [float(np.abs(d[:, 0]).max()), float(np.abs(d[:, 1]).max())],
            "rel_rms_front": rms(front, d[:, 0], d[:, 2]), "rel_rms_rear": rms(rear, d[:, 1], d[:, 3])}
    return front, rear, info


# ---------------------------------------------------------------------- controller


class NMPC:
    """One NLP, solved per environment and control step. ``__call__(env)`` returns the actions of all of a
    ``VehicleTrackingEnv``'s environments (raw, pre-tanh, as the policies output), keeping a warm start per env."""

    def __init__(self, env_cfg: VehicleEnvCfg, params: VehicleParams | None = None, cfg: NMPCCfg | None = None,
                 tyres: tuple[AxleTyre, AxleTyre] | None = None, tracking: TrackingCfg | None = None):
        import casadi as ca

        self.ca = ca
        self.cfg = cfg = cfg or NMPCCfg()
        self.env_cfg, self.p = env_cfg, params or VehicleParams()
        self.tr = tracking or env_cfg.tracking
        if tyres is None:
            front, rear, self.fit_info = identify_axle_tyres(self.p, env_cfg)
        else:
            (front, rear), self.fit_info = tyres, {}
        self.front, self.rear = front, rear
        self.T = cfg.interval * env_cfg.dt
        self._build()
        self.warm: dict[int, dict] = {}
        self.solve_ms: list[float] = []
        self.failures = 0
        self.solves = 0

    # -------------------------------------------------------------- model

    def _ode(self, x, u, mu):
        ca, p, ec = self.ca, self.p, self.env_cfg
        _px, _py, yaw, vx, vy, r, delta, a_act = (x[i] for i in range(NX))
        vxs = ca.fmax(vx, 1.0)
        alpha_f = ca.atan((vy + p.a * r) / vxs) - delta
        alpha_r = ca.atan((vy - p.b * r) / vxs)
        fyf, fyr = self.front.force(alpha_f, mu, ca), self.rear.force(alpha_r, mu, ca)
        # the env's steering servo and longitudinal lag are discrete first-order steps; matching time constants
        tau_d = -ec.dt / math.log(1.0 - ec.dt / ec.steer_tau)
        tau_a = ec.accel_tau
        return ca.vertcat(
            vx * ca.cos(yaw) - vy * ca.sin(yaw),
            vx * ca.sin(yaw) + vy * ca.cos(yaw),
            r,
            a_act - fyf * ca.sin(delta) / p.m + r * vy,
            (fyf * ca.cos(delta) + fyr) / p.m - r * vx,
            (p.a * fyf * ca.cos(delta) - p.b * fyr) / p.I_z,
            (u[0] - delta) / tau_d,
            (u[1] - a_act) / tau_a,
        )

    def _step(self, x, u, mu):
        h = self.T / self.cfg.rk4_substeps
        for _ in range(self.cfg.rk4_substeps):
            k1 = self._ode(x, u, mu)
            k2 = self._ode(x + h / 2 * k1, u, mu)
            k3 = self._ode(x + h / 2 * k2, u, mu)
            k4 = self._ode(x + h * k3, u, mu)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    def _build(self):
        ca, cfg, tr, ec, p = self.ca, self.cfg, self.tr, self.env_cfg, self.p
        N = cfg.horizon
        xs = ca.SX.sym("x", NX)
        us = ca.SX.sym("u", NU)
        ms = ca.SX.sym("mu")
        self.F = ca.Function("F", [xs, us, ms], [self._step(xs, us, ms)])
        Xv = ca.SX.sym("X", NX, N + 1)
        Uv = ca.SX.sym("U", NU, N)
        P = ca.SX.sym("P", NX + NREF * N + NU + 2)
        x0, refs = P[:NX], ca.reshape(P[NX:NX + NREF * N], NREF, N)
        u_prev, mu, a_hi = P[NX + NREF * N:NX + NREF * N + NU], P[-2], P[-1]
        Fm = self.F.map(N)
        Xn = Fm(Xv[:, :N], Uv, ca.repmat(mu, 1, N))
        g = [Xv[:, 0] - x0, ca.reshape(Xv[:, 1:] - Xn, -1, 1)]
        # steering rate limit, exact for the env's first control step: |delta_cmd - delta| <= rate_max * tau
        g.append(ca.reshape(Uv[0, :] - Xv[6, :N], -1, 1))
        wq = [tr.w_lat / tr.sigma_lat**2, tr.w_lon / tr.sigma_lon**2, tr.w_yaw / tr.sigma_yaw**2, tr.w_v / tr.sigma_v**2, tr.w_beta / tr.sigma_beta**2]
        J = 0
        for k in range(N):
            xk, rk = Xv[:, k + 1], refs[:, k]
            dx, dy = xk[0] - rk[0], xk[1] - rk[1]
            c, s = ca.cos(rk[2]), ca.sin(rk[2])
            vx, vy = xk[3], xk[4]
            e = [-s * dx + c * dy, c * dx + s * dy, xk[2] - rk[2], ca.sqrt(vx**2 + vy**2 + 1e-6) - rk[3], ca.atan(vy / ca.fmax(vx, 1.0)) - rk[4]]
            wk = cfg.terminal_scale if k == N - 1 else 1.0
            J += wk * sum(w * ei**2 for w, ei in zip(wq, e))
            du = Uv[:, k] - (u_prev if k == 0 else Uv[:, k - 1])
            J += cfg.w_steer_rate * du[0] ** 2 + cfg.w_accel_rate * du[1] ** 2
        w = ca.vertcat(ca.reshape(Xv, -1, 1), ca.reshape(Uv, -1, 1))
        G = ca.vertcat(*g)
        # a_cmd upper bound (drive limit at the current speed) as a linear constraint on the inputs
        G = ca.vertcat(G, ca.reshape(Uv[1, :] - a_hi, -1, 1))
        self.n_dyn = NX * (N + 1)
        self.n_rate = N
        self.n_ahi = N
        opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": cfg.max_iter, "ipopt.tol": cfg.tol,
                "ipopt.acceptable_tol": 1e-3, "ipopt.acceptable_iter": 3, "ipopt.warm_start_init_point": "yes",
                "ipopt.mu_init": 1e-3, "ipopt.sb": "yes", "expand": True}
        self.solver = ca.nlpsol("nmpc", "ipopt", {"x": w, "p": P, "f": J, "g": G}, opts)
        rate_gap = p.steer_rate_max * ec.steer_tau
        self.lbg = np.concatenate([np.zeros(self.n_dyn), -rate_gap * np.ones(N), -np.inf * np.ones(N)])
        self.ubg = np.concatenate([np.zeros(self.n_dyn), rate_gap * np.ones(N), np.zeros(N)])
        big = np.inf
        lbx = np.full(NX * (N + 1) + NU * N, -big)
        ubx = np.full(NX * (N + 1) + NU * N, big)
        du_ = NX * (N + 1)
        lbx[du_ + 0::2] = -cfg.cmd_margin * ec.steer_scale
        ubx[du_ + 0::2] = cfg.cmd_margin * ec.steer_scale
        lbx[du_ + 1::2] = -cfg.cmd_margin * ec.accel_scale
        ubx[du_ + 1::2] = cfg.cmd_margin * ec.accel_scale
        self.lbx, self.ubx = lbx, ubx

    # -------------------------------------------------------------- solving

    def solve(self, x0: np.ndarray, refs: np.ndarray, u_prev: np.ndarray, mu: float, guess: dict | None) -> tuple[np.ndarray, dict, bool]:
        """x0 (NX,), refs (horizon, NREF) -> first command (NU,), the solution (for warm starts), success."""
        cfg, ec, p = self.cfg, self.env_cfg, self.p
        vx = max(float(x0[3]), 1e-3)
        a_hi = min(cfg.cmd_margin * ec.accel_scale, p.a_max * p.v_switch / vx if vx > p.v_switch else p.a_max)
        P = np.concatenate([x0, refs.reshape(-1), u_prev, [mu, a_hi]])  # refs row k -> column k of the NLP's (NREF, N)
        kw = {"p": P, "lbg": self.lbg, "ubg": self.ubg, "lbx": self.lbx, "ubx": self.ubx, "x0": guess["x"]}
        if "lam_g" in guess:
            kw["lam_g0"], kw["lam_x0"] = guess["lam_g"], guess["lam_x"]
        t0 = time.perf_counter()
        sol = self.solver(**kw)
        self.solve_ms.append(1e3 * (time.perf_counter() - t0))
        self.solves += 1
        status = self.solver.stats()["return_status"]
        ok = status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
        x = np.asarray(sol["x"]).ravel()
        if not np.isfinite(x).all():
            ok, x = False, guess["x"]
        self.failures += not ok
        out = {"x": x, "lam_g": np.asarray(sol["lam_g"]).ravel(), "lam_x": np.asarray(sol["lam_x"]).ravel()}
        u0 = x[NX * (cfg.horizon + 1):NX * (cfg.horizon + 1) + NU]
        return u0, out, ok

    def _shift(self, sol: dict) -> dict:
        N = self.cfg.horizon
        X = sol["x"][:NX * (N + 1)].reshape(N + 1, NX)
        U = sol["x"][NX * (N + 1):].reshape(N, NU)
        # one control step of an interval-step grid: interpolate one ``1 / interval`` of the way forward
        f = 1.0 / self.cfg.interval
        Xs = np.vstack([X[:-1] + f * (X[1:] - X[:-1]), X[-1:]])
        return {"x": np.concatenate([Xs.ravel(), U.ravel()])}

    def _cold(self, lib, frame: int, last: int, x0: np.ndarray) -> dict:
        """Initial guess from the reference itself (its states and recorded commands)."""
        N, m, ec = self.cfg.horizon, self.cfg.interval, self.env_cfg
        idx = np.minimum(frame - 1 + m * np.arange(N + 1), last)
        f = lambda t: t[idx].detach().cpu().double().numpy()  # noqa: E731
        X = np.stack([f(lib.x), f(lib.y), f(lib.yaw), f(lib.vx), f(lib.vy), f(lib.r), f(lib.delta), f(lib.a_act)], -1)
        X[0] = x0
        acts = lib.actions[idx[:-1]].detach().cpu().double().numpy()
        U = np.stack([np.tanh(acts[:, 0]) * ec.steer_scale, np.tanh(acts[:, 1]) * ec.accel_scale], -1)
        U = np.clip(U, self.lbx[NX * (N + 1):NX * (N + 1) + 2], self.ubx[NX * (N + 1):NX * (N + 1) + 2])
        return {"x": np.concatenate([X.ravel(), U.ravel()])}

    def references(self, lib, frame: int, yaw_now: float) -> tuple[np.ndarray, int]:
        N, m = self.cfg.horizon, self.cfg.interval
        last = int(lib.layout.last_of(torch.tensor([frame], device=lib.x.device))[0])
        idx = np.minimum(frame - 1 + m * np.arange(1, N + 1), last)
        f = lambda t: t[idx].detach().cpu().double().numpy()  # noqa: E731
        refs = np.stack([f(lib.x), f(lib.y), f(lib.yaw), f(lib.v), f(lib.beta)], -1)
        refs[:, 2] += 2 * math.pi * round((yaw_now - refs[0, 2]) / (2 * math.pi))  # unwrap onto the car's yaw
        return refs, last

    def reset(self, ids=None) -> None:
        if ids is None:
            self.warm.clear()
        else:
            for i in ids:
                self.warm.pop(int(i), None)

    @torch.no_grad()
    def __call__(self, env) -> torch.Tensor:
        """Actions (n, 2) for every environment of ``env`` from its current (true) state."""
        ec = self.env_cfg
        s = env.s.detach().cpu().double().numpy()
        a_act = env.act.a_act.detach().cpu().double().numpy()
        frames = env.frame.detach().cpu().numpy()
        mus = env.ctx.mu.detach().cpu().double().numpy()
        fresh = (env.ep_steps == 0).detach().cpu().numpy()
        prev = env.prev_action.detach().cpu().double().numpy()
        out = np.zeros((env.num_envs, NU))
        for i in range(env.num_envs):
            x0 = np.array([s[i, X], s[i, Y], s[i, YAW], s[i, VX], s[i, VY], s[i, R], s[i, DELTA], a_act[i]])
            refs, last = self.references(env.lib, int(frames[i]), float(x0[2]))
            mu = float(mus[i]) if self.cfg.friction == "known" else 1.0
            if fresh[i] or i not in self.warm:
                guess = self._cold(env.lib, int(frames[i]), last, x0)
                u_prev = guess["x"][NX * (self.cfg.horizon + 1):NX * (self.cfg.horizon + 1) + NU]
            else:
                guess = self.warm[i]
                u_prev = np.array([np.tanh(prev[i, 0]) * ec.steer_scale, np.tanh(prev[i, 1]) * ec.accel_scale])
            u0, sol, _ok = self.solve(x0, refs, u_prev, mu, guess)
            self.warm[i] = {**self._shift(sol), "lam_g": sol["lam_g"], "lam_x": sol["lam_x"]}
            lim = self.cfg.cmd_margin
            out[i, 0] = math.atanh(float(np.clip(u0[0] / ec.steer_scale, -lim, lim)))
            out[i, 1] = math.atanh(float(np.clip(u0[1] / ec.accel_scale, -lim, lim)))
        return torch.as_tensor(out, dtype=env.s.dtype, device=env.s.device)

    def timing(self) -> dict:
        t = np.asarray(self.solve_ms) if self.solve_ms else np.zeros(1)
        return {"solves": self.solves, "failures": self.failures, "mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)),
                "p95_ms": float(np.percentile(t, 95)), "max_ms": float(t.max())}


# keep AX / AY / G importable from here for callers computing friction use alongside the controller
__all__ = ["NMPC", "NMPCCfg", "AxleTyre", "identify_axle_tyres", "excitation_data", "fit_axle", "AX", "AY", "G"]
