"""``chrono_roll_mpc.SloshMPC``: with the swing decoupled (zero coefficients) it plans exactly like ``RollMPC``, and
with a coupling the measured swing changes the plan."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vehicle"))
pytest.importorskip("casadi")

TYRE = {"B": 7.0, "C": 1.3, "D": -15000.0, "E": 0.0}
ROLL, LTR = [0.2, -33.0, -1.2], [-0.02, -11.5, -0.23]


def _obs(slosh=0.0):
    return {"x": 0.0, "y": 0.0, "yaw": 0.0, "vx": 13.0, "vy": 0.0, "r": 0.0, "roll_b": 0.0, "p": 0.0, "delta": 0.0, "ax": 0.0, "slosh": slosh}


def _act(mpc, path, slosh=0.0):
    mpc.reset(1)
    return mpc.act(0, _obs(slosh), path, 0, 13.0, [0.0, 0.0])


def test_decoupled_swing_matches_roll_mpc():
    from chrono_roll_mpc import RollMPC, SloshMPC
    from chrono_rollover_control import lane_change_path

    path, _ = lane_change_path(np.random.default_rng(0))
    base = {"front": TYRE, "rear": {**TYRE, "D": -14500.0}}
    plain = RollMPC({**base, "roll": {"coef_ay_roll_rate": ROLL}, "ltr": {"coef_ay_roll_rate": LTR}})
    oracle = SloshMPC({**base, "roll": {"coef_ay_roll_rate_swing_swingrate": [*ROLL, 0.0, 0.0]},
                       "ltr": {"coef_ay_roll_rate_swing_swingrate": [*LTR, 0.0, 0.0]},
                       "slosh": {"coef_ay_roll_rate_swing_swingrate": [0.0, 0.0, 0.0, -12.0, -1.0]}})
    a_plain, a_oracle = _act(plain, path), _act(oracle, path, slosh=0.2)
    assert np.allclose(a_plain, a_oracle, atol=1e-3)


def test_swing_coupling_changes_plan():
    from chrono_roll_mpc import SloshMPC
    from chrono_rollover_control import lane_change_path

    path, _ = lane_change_path(np.random.default_rng(0))
    ident = {"front": TYRE, "rear": {**TYRE, "D": -14500.0}, "roll": {"coef_ay_roll_rate_swing_swingrate": [*ROLL, 40.0, 2.0]},
             "ltr": {"coef_ay_roll_rate_swing_swingrate": [*LTR, 1.5, 0.1]},
             "slosh": {"coef_ay_roll_rate_swing_swingrate": [-1.2, 10.0, 0.5, -12.0, -1.0]}}
    a0, a1 = _act(SloshMPC(ident), path, 0.0), _act(SloshMPC(ident), path, 0.3)
    assert np.isfinite(a0).all() and np.isfinite(a1).all() and not np.allclose(a0, a1, atol=1e-3)


def test_swing_observer_tracks_simulated_swing():
    """On data from the observer's own (linear) model, the estimate converges to the true swing within ~2 s."""
    from chrono_roll_mpc import SwingObserver

    cr, cs = [0.56, -95.0, -6.0, 4.94, 0.32], [0.30, 0.0, 0.0, -14.3, -1.34]  # stable: roll damping, no roll-to-swing coupling
    ident = {"roll": {"coef_ay_roll_rate_swing_swingrate": cr, "resid_std": 0.05}, "slosh": {"coef_ay_roll_rate_swing_swingrate": cs, "resid_std": 0.05}}
    ob = SwingObserver(ident)
    ob.reset()
    dt, th, thd, roll, p = 0.02, 0.15, 0.0, 0.0, 0.0  # starts swinging; the observer starts at zero
    errs, ths = [], []
    for k in range(400):
        ay = 3.0 * np.sin(2 * np.pi * 0.4 * k * dt)
        est = ob.update(ay, roll, p)
        errs.append(est[0] - th)
        ths.append(th)
        pdot = cr[0] * ay + cr[1] * roll + cr[2] * p + cr[3] * th + cr[4] * thd
        thdd = cs[0] * ay + cs[1] * roll + cs[2] * p + cs[3] * th + cs[4] * thd
        roll, p, th, thd = roll + dt * p, p + dt * pdot, th + dt * thd, thd + dt * thdd
    late = np.abs(errs[100:])
    assert late.mean() < 0.1 * np.abs(ths[100:]).mean() + 1e-3
