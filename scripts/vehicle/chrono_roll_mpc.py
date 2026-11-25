"""Rollover-aware NMPC: the industrial answer to ``chrono_rollover_control.py``'s task (VEHICLE_PROPOSAL.md
section 11.12).

Prediction model: single-track lateral dynamics (axle tyres identified from Chrono data: cornering stiffness fitted,
peak = 0.9 x axle load) plus a roll degree of freedom, with the roll acceleration and the signed lateral load transfer
ratio LTR = (F_left - F_right) / F_total linear in (lateral acceleration, roll, roll rate), both identified by least
squares on the grounded samples of ``chrono_lift_check.py``'s random-excitation data. The vehicle constants are
Chrono's (HMMWV + 800 kg roof load); the steering servo and the longitudinal response are modelled as in
``baselines.nmpc``.

Optimisation (CasADi + IPOPT, multiple shooting, 30 intervals of 3 control steps = 1.8 s, warm-started): path
following in the vehicle's local frame (the next 90 m of path passed as parameters through a linear interpolant),
heading, speed below the reference cheap and above it expensive, a strong penalty on |LTR| above 0.6 and on leaving a
0.4 m corridor, input-rate penalties. It sees the same true Chrono state as the learned controllers.

    SUPERMIMIC_CHRONO_PYTHON=chrono_env/bin/python PYTHONPATH=src python scripts/vehicle/chrono_roll_mpc.py \\
        --lift-data logs/chrono_lift_data.npz --out docs/results/vehicle_chrono_roll_mpc.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chrono_lift_check import STATE
from chrono_rollover_control import (
    A_MAX,
    A_MIN,
    DS,
    EVAL_SPEEDS,
    evaluate,
    lane_change_path,
    set_slosh,
)

from supermimic.baselines.nmpc import AxleTyre
from supermimic.vehicles.chrono_env import ChronoPool, axle_forces

IX = {k: i for i, k in enumerate(STATE)}
K_STEER = 0.49  # front-wheel angle per unit steering command (chrono_g1 identification)
# HMMWV (Chrono) + 800 kg roof box centred above the chassis reference frame
M = 2573.1 + 800.0
X_COM = 2573.1 * 0.051 / M  # combined CoG ahead of the chassis reference frame [m]
A_AX, B_AX = 1.649 - X_COM, 1.653 + X_COM
I_Z = 4686.3 + 800.0 * (1.6**2 + 1.4**2) / 12
MU_PRIOR = 0.9


def identify(lift_data: str) -> dict:
    d = np.load(lift_data, allow_pickle=True)
    rows = []
    for ep in d["train"]:
        x = ep["x"]
        if len(x) < 10:
            continue
        vy, r, vx, p = x[:, IX["vy"]], x[:, IX["r"]], x[:, IX["vx"]], x[:, IX["p"]]
        ay = np.gradient(vy, 0.02) + r * vx
        rdot = np.gradient(r, 0.02)
        fz = x[:, 12:16]
        ltr = ((fz[:, 0] + fz[:, 2]) - (fz[:, 1] + fz[:, 3])) / np.maximum(fz.sum(1), 1e3)
        grounded = (fz >= 100).all(1) & (ep["tilt"] < math.radians(10)) & (vx > 5)
        rows.append(np.stack([ay, x[:, IX["roll_b"]], p, np.gradient(p, 0.02), ltr, vx, vy, r, x[:, IX["delta"]], rdot, grounded], 1))
    D = np.concatenate(rows)
    D = D[D[:, -1] > 0]
    A = D[:, :3]
    out = {}
    for name, col in (("roll", 3), ("ltr", 4)):
        coef, *_ = np.linalg.lstsq(A, D[:, col], rcond=None)
        pred = A @ coef
        out[name] = {"coef_ay_roll_rate": coef.tolist(), "r2": float(1 - ((D[:, col] - pred) ** 2).mean() / D[:, col].var())}
    L = {"ay": D[:, 0], "vx": D[:, 5], "vy": D[:, 6], "r": D[:, 7], "delta": D[:, 8], "rdot": D[:, 9]}
    f = axle_forces(L, M, I_Z, A_AX, B_AX)
    loads = [M * 9.81 * B_AX / (A_AX + B_AX), M * 9.81 * A_AX / (A_AX + B_AX)]
    for ax, fz in zip("fr", loads):
        al, fy = f["alpha_" + ax], f["fy" + ax]
        sel = np.abs(al) < 0.06
        K = float(np.sum(al[sel] * fy[sel]) / np.sum(al[sel] ** 2))
        D_ = math.copysign(MU_PRIOR * fz, K)
        out["front" if ax == "f" else "rear"] = {"B": K / (1.3 * D_), "C": 1.3, "D": D_, "E": 0.0}
        out[f"cornering_stiffness_{ax}"] = K
    out["samples"] = len(D)
    return out


class RollMPC:
    N, INTERVAL, DT = 30, 3, 0.02
    NX = 10  # X, Y, yaw, vx, vy, r, roll, roll rate, wheel angle, acceleration
    GRID = np.arange(0.0, 91.0, 1.0)

    def __init__(self, ident: dict, w_ltr: float = 500.0, ltr_soft: float = 0.6):
        import casadi as ca

        self.ca, self.id = ca, ident
        self.front, self.rear = AxleTyre(**ident["front"]), AxleTyre(**ident["rear"])
        self.cr, self.cl = ident["roll"]["coef_ay_roll_rate"], ident["ltr"]["coef_ay_roll_rate"]
        self.w_ltr, self.ltr_soft = w_ltr, ltr_soft
        self.T = self.INTERVAL * self.DT
        self.tau_d = -self.DT / math.log(1.0 - self.DT / 0.05)  # the servo's discrete first-order step as a time constant
        self._build()
        self.warm, self.solve_ms, self.failures = {}, [], 0

    def _axle_forces(self, x):
        ca = self.ca
        vxs = ca.fmax(x[3], 1.0)
        af = ca.atan((x[4] + A_AX * x[5]) / vxs) - x[8]
        ar = ca.atan((x[4] - B_AX * x[5]) / vxs)
        fyf, fyr = self.front.force(af, 1.0, ca), self.rear.force(ar, 1.0, ca)
        return fyf, fyr, (fyf * ca.cos(x[8]) + fyr) / M

    def _ode(self, x, u):
        ca = self.ca
        _X, _Y, psi, vx, vy, r, phi, p, delta, a_act = (x[i] for i in range(10))
        fyf, fyr, ay = self._axle_forces(x)
        c = self.cr
        return ca.vertcat(vx * ca.cos(psi) - vy * ca.sin(psi), vx * ca.sin(psi) + vy * ca.cos(psi), r,
                          a_act - fyf * ca.sin(delta) / M + r * vy, ay - r * vx, (A_AX * fyf * ca.cos(delta) - B_AX * fyr) / I_Z,
                          p, c[0] * ay + c[1] * phi + c[2] * p, (u[0] - delta) / self.tau_d, (u[1] - a_act) / 0.5)

    def _ltr(self, x, u):
        ay = self._axle_forces(x)[2]
        c = self.cl
        return c[0] * ay + c[1] * x[6] + c[2] * x[7]

    def _build(self):
        ca, N, NG = self.ca, self.N, len(self.GRID)
        NX = self.NX
        xs, us = ca.SX.sym("x", NX), ca.SX.sym("u", 2)
        h = self.T / 2
        x = xs
        for _ in range(2):  # RK4, 2 substeps per interval
            k1 = self._ode(x, us)
            k2 = self._ode(x + h / 2 * k1, us)
            k3 = self._ode(x + h / 2 * k2, us)
            k4 = self._ode(x + h * k3, us)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        F = ca.Function("F", [xs, us], [x])
        LT = ca.Function("LT", [xs, us], [self._ltr(xs, us)])
        yl = ca.interpolant("yl", "linear", [self.GRID.tolist()], 1)
        pl = ca.interpolant("pl", "linear", [self.GRID.tolist()], 1)
        Xv, Uv = ca.SX.sym("X", NX, N + 1), ca.SX.sym("U", 2, N)
        P = ca.SX.sym("P", NX + 2 * NG + 2 + 1)  # x0, path y and yaw on the grid, previous command, v_ref
        x0, ygrid, pgrid, u_prev, v_ref = P[:NX], P[NX:NX + NG], P[NX + NG:NX + 2 * NG], P[NX + 2 * NG:NX + 2 + 2 * NG], P[-1]
        g = [Xv[:, 0] - x0]
        J = 0
        for k in range(N):
            g.append(Xv[:, k + 1] - F(Xv[:, k], Uv[:, k]))
            xk = Xv[:, k + 1]
            xq = ca.fmin(ca.fmax(xk[0], 0.0), self.GRID[-1])
            ypath, ppath = yl(xq, ygrid), pl(xq, pgrid)
            e_lat = (xk[1] - ypath) * ca.cos(ppath)
            e_head = xk[2] - ppath
            ltr = LT(xk, Uv[:, k])
            alt = ca.sqrt(ltr**2 + 1e-6)
            J += 30.0 * e_lat**2 + 10.0 * e_head**2 + 0.1 * ca.fmax(0, v_ref - xk[3]) ** 2 + 20.0 * ca.fmax(0, xk[3] - v_ref) ** 2
            J += self.w_ltr * ca.fmax(0, alt - self.ltr_soft) ** 2 + 500.0 * ca.fmax(0, ca.sqrt(e_lat**2 + 1e-6) - 0.4) ** 2
            du = Uv[:, k] - (u_prev if k == 0 else Uv[:, k - 1])
            J += 50.0 * du[0] ** 2 + 0.05 * du[1] ** 2
        g.append(ca.reshape(Uv[0, :] - Xv[8, :N], -1, 1))  # steering rate limit: |delta_cmd - delta| <= rate * tau
        w = ca.vertcat(ca.reshape(Xv, -1, 1), ca.reshape(Uv, -1, 1))
        opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": 80, "ipopt.tol": 1e-4, "ipopt.acceptable_tol": 1e-3,
                "ipopt.acceptable_iter": 3, "ipopt.warm_start_init_point": "yes", "ipopt.mu_init": 1e-3, "ipopt.sb": "yes", "expand": True}
        self.solver = ca.nlpsol("rollmpc", "ipopt", {"x": w, "p": P, "f": J, "g": ca.vertcat(*g)}, opts)
        rate_gap = 1.2 * K_STEER * 0.05
        nd = NX * (N + 1)
        self.lbg = np.concatenate([np.zeros(nd), -rate_gap * np.ones(N)])
        self.ubg = np.concatenate([np.zeros(nd), rate_gap * np.ones(N)])
        self.lbx = np.full(nd + 2 * N, -np.inf)
        self.ubx = np.full(nd + 2 * N, np.inf)
        self.lbx[nd::2], self.ubx[nd::2] = -0.97 * K_STEER, 0.97 * K_STEER
        self.lbx[nd + 1::2], self.ubx[nd + 1::2] = A_MIN, A_MAX
        self.nd = nd

    def reset(self, n):
        self.warm = {}
        self.a_est = [0.0] * n

    def _x0(self, i, o):
        self.a_est[i] = 0.8 * self.a_est[i] + 0.2 * o["ax"]
        return np.array([0.0, 0.0, 0.0, o["vx"], o["vy"], o["r"], o["roll_b"], o["p"], o["delta"], self.a_est[i]])

    def act(self, i, o, path, j, v_ref, a_prev):
        N, nd = self.N, self.nd
        c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
        seg = path[max(0, j - 8): j + int(100 / DS)]
        dx, dy = seg[:, 0] - o["x"], seg[:, 1] - o["y"]
        xl, ylc = c * dx + s * dy, -s * dx + c * dy
        psil = np.unwrap(seg[:, 2] - o["yaw"])
        order = np.argsort(xl)
        ygrid = np.interp(self.GRID, xl[order], ylc[order])
        pgrid = np.interp(self.GRID, xl[order], psil[order])
        x0 = self._x0(i, o)
        u_prev = np.array([a_prev[0] * K_STEER, a_prev[1]])
        P = np.concatenate([x0, ygrid, pgrid, u_prev, [v_ref]])
        if i in self.warm:  # shift the previous solution by one control step, re-anchored at the new state
            g = self.warm[i]
        else:
            X = np.tile(x0, (N + 1, 1))
            X[:, 0] = o["vx"] * self.T * np.arange(N + 1)
            g = {"x": np.concatenate([X.ravel(), np.tile(u_prev, N)])}
        kw = {"p": P, "lbg": self.lbg, "ubg": self.ubg, "lbx": self.lbx, "ubx": self.ubx, "x0": g["x"]}
        if "lam_g" in g:
            kw["lam_g0"], kw["lam_x0"] = g["lam_g"], g["lam_x"]
        t0 = time.perf_counter()
        sol = self.solver(**kw)
        self.solve_ms.append(1e3 * (time.perf_counter() - t0))
        ok = self.solver.stats()["return_status"] in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
        xs = np.asarray(sol["x"]).ravel()
        if not np.isfinite(xs).all():
            ok, xs = False, g["x"]
        self.failures += not ok
        X = xs[:nd].reshape(N + 1, self.NX)
        U = xs[nd:].reshape(N, 2)
        # warm start for the next step: the local frame moves with the car, so start again from the shifted inputs and
        # the predicted states expressed relative to the first predicted step
        f = 1.0 / self.INTERVAL
        Xs = np.vstack([X[:-1] + f * (X[1:] - X[:-1]), X[-1:]])
        cx, sy, ps = Xs[0, 0], Xs[0, 1], Xs[0, 2]
        dxs, dys = Xs[:, 0] - cx, Xs[:, 1] - sy
        Xs[:, 0], Xs[:, 1], Xs[:, 2] = math.cos(ps) * dxs + math.sin(ps) * dys, -math.sin(ps) * dxs + math.cos(ps) * dys, Xs[:, 2] - ps
        self.warm[i] = {"x": np.concatenate([Xs.ravel(), U.ravel()]), "lam_g": np.asarray(sol["lam_g"]).ravel(), "lam_x": np.asarray(sol["lam_x"]).ravel()}
        return [float(np.clip(U[0, 0] / K_STEER, -1.0, 1.0)), float(np.clip(U[0, 1], A_MIN, A_MAX))]

    def timing(self):
        t = np.asarray(self.solve_ms) if self.solve_ms else np.zeros(1)
        return {"solves": len(self.solve_ms), "failures": self.failures, "mean_ms": float(t.mean()), "p95_ms": float(np.percentile(t, 95))}


def identify_slosh(lift_data: str) -> dict:
    """``identify`` plus roll, LTR and payload-swing models that also see the swing angle and rate: privileged
    information (``chrono_worker``'s diagnostic ``slosh`` channel, recorded by ``chrono_lift_check.collect``)."""
    out = identify(lift_data)
    d = np.load(lift_data, allow_pickle=True)
    rows = []
    for ep in d["train"]:
        x = ep["x"]
        if len(x) < 10:
            continue
        if "slosh" not in ep:
            raise ValueError(f"{lift_data} has no payload swing angle: recollect it with the current chrono_lift_check")
        vy, r, vx, p, th = x[:, IX["vy"]], x[:, IX["r"]], x[:, IX["vx"]], x[:, IX["p"]], ep["slosh"]
        thd = np.gradient(th, 0.02)
        fz = x[:, 12:16]
        ltr = ((fz[:, 0] + fz[:, 2]) - (fz[:, 1] + fz[:, 3])) / np.maximum(fz.sum(1), 1e3)
        grounded = (fz >= 100).all(1) & (ep["tilt"] < math.radians(10)) & (vx > 5)
        rows.append(np.stack([np.gradient(vy, 0.02) + r * vx, x[:, IX["roll_b"]], p, th, thd, np.gradient(p, 0.02), ltr, np.gradient(thd, 0.02), grounded], 1))
    D = np.concatenate(rows)
    D = D[D[:, -1] > 0]
    A = D[:, :5]
    for name, col in (("roll", 5), ("ltr", 6), ("slosh", 7)):
        coef, *_ = np.linalg.lstsq(A, D[:, col], rcond=None)
        out[name] = {"coef_ay_roll_rate_swing_swingrate": coef.tolist(), "r2": float(1 - ((D[:, col] - A @ coef) ** 2).mean() / D[:, col].var())}
    return out


class SloshMPC(RollMPC):
    """``RollMPC`` with the payload's swing angle and rate as two more states, identified as linear in (lateral
    acceleration, roll, roll rate, swing, swing rate), and the measured swing angle in the initial state. An oracle:
    it is told the payload sloshes and sees the swing, which neither the plain MPC nor the learned controllers do."""

    NX = 12

    def __init__(self, ident: dict, w_ltr: float = 500.0, ltr_soft: float = 0.6):
        self.cs = ident["slosh"]["coef_ay_roll_rate_swing_swingrate"]
        ident = {**ident, "roll": {"coef_ay_roll_rate": ident["roll"]["coef_ay_roll_rate_swing_swingrate"]},
                 "ltr": {"coef_ay_roll_rate": ident["ltr"]["coef_ay_roll_rate_swing_swingrate"]}}
        super().__init__(ident, w_ltr, ltr_soft)

    def _ode(self, x, u):
        ca = self.ca
        base = super()._ode(x, u)  # uses the first three roll coefficients
        ay, phi, p, th, thd = self._axle_forces(x)[2], x[6], x[7], x[10], x[11]
        c, s = self.cr, self.cs
        return ca.vertcat(base[:7], base[7] + c[3] * th + c[4] * thd, base[8:10], thd, s[0] * ay + s[1] * phi + s[2] * p + s[3] * th + s[4] * thd)

    def _ltr(self, x, u):
        return super()._ltr(x, u) + self.cl[3] * x[10] + self.cl[4] * x[11]

    def reset(self, n):
        super().reset(n)
        self.th_prev = [None] * n

    def _x0(self, i, o):
        th = float(o["slosh"])
        thd = 0.0 if self.th_prev[i] is None else (th - self.th_prev[i]) / self.DT
        self.th_prev[i] = th
        return np.concatenate([super()._x0(i, o), [th, thd]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lift-data", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-paths", type=int, default=8)
    ap.add_argument("--w-ltr", type=float, nargs="+", default=[500.0], help="LTR penalty weights to evaluate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slosh-length", type=float, default=0.0, help="hang the roof load on a pendulum this long [m] (0: rigid)")
    ap.add_argument("--slosh-damping", type=float, default=500.0, help="pendulum damping [N m s/rad]")
    ap.add_argument("--slosh-aware", action="store_true", help="oracle SloshMPC: models the swing and sees its angle")
    ap.add_argument("--python", default=None)
    ap.add_argument("--out", default="logs/chrono_roll_mpc.json")
    args = ap.parse_args(argv)
    t0 = time.time()
    if args.slosh_length > 0:
        set_slosh(args.slosh_length, args.slosh_damping)
    ident = identify_slosh(args.lift_data) if args.slosh_aware else identify(args.lift_data)
    if args.slosh_aware:
        print(f"swing model R2 {ident['slosh']['r2']:.3f}", flush=True)
    print(f"identified: roll model R2 {ident['roll']['r2']:.3f}, LTR model R2 {ident['ltr']['r2']:.3f}, cornering stiffness "
          f"{ident['cornering_stiffness_f']:.0f} / {ident['cornering_stiffness_r']:.0f} N/rad ({ident['samples']} grounded samples)", flush=True)
    eval_rng = np.random.default_rng(args.seed + 2000)
    cases = []
    for _ in range(args.eval_paths):
        p, _ = lane_change_path(eval_rng)
        cases += [(p, v) for v in EVAL_SPEEDS]
    report = {"identification": ident, "runs": {}}
    with ChronoPool(args.workers, args.python) as pool:
        for w in args.w_ltr:
            mpc = (SloshMPC if args.slosh_aware else RollMPC)(ident, w_ltr=w)
            ev, _ = evaluate(pool, cases, mpc, types.SimpleNamespace(workers=args.workers))
            ev["solver"] = mpc.timing()
            report["runs"][f"w_ltr_{w:g}"] = ev
            print(f"roll MPC (w_ltr {w:g}): rolled {ev['rolled']}/{ev['episodes']}, off path {ev['off_path']}, return {ev['mean_return']:.1f}, "
                  f"|lat| {ev['mean_abs_lat']:.3f} m, lift {ev['two_wheel_lift_s']:.1f} s, speed {ev['mean_speed']:.1f} m/s; "
                  f"solve {ev['solver']['mean_ms']:.0f} ms (p95 {ev['solver']['p95_ms']:.0f}), failures {ev['solver']['failures']} ({time.time() - t0:.0f} s)", flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
