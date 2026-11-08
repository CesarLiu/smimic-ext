"""``vehicles.chrono_env``: path references, tracking errors, pure pursuit and single-track identification (plain
numpy, always run); the Chrono worker pool itself only when ``SUPERMIMIC_CHRONO_PYTHON`` names a PyChrono
interpreter (VEHICLE_RUNBOOK.md step 10)."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from supermimic.vehicles.chrono_env import (
    axle_forces,
    path_errors,
    path_reference,
    pure_pursuit,
    random_path,
    s_bend,
)


def test_s_bend_geometry():
    ref = s_bend(v=4.0, ay=1.6)
    assert abs(ref.kappa.max() - 1.6 / 16) < 1e-9 and abs(ref.kappa.min() + 1.6 / 16) < 1e-9
    assert abs(ref.yaw[-1]) < 1e-6  # left arc then right arc: back to the start heading
    step = np.hypot(np.diff(ref.x), np.diff(ref.y))
    assert np.allclose(step, 4.0 * 0.02, rtol=1e-3)  # constant speed
    nodes = ref.node([0, 5, 10_000])
    assert nodes.shape == (3, 5) and nodes[2, 0] == ref.x[-1] and np.all(nodes[:, 3] == 4.0)


def test_random_path_peak_and_start():
    ref = random_path(v=5.0, ay=2.0, length=80.0, seed=3, x0=1.0, y0=2.0, yaw0=0.5)
    assert abs(np.abs(ref.kappa).max() - 2.0 / 25) < 2e-3
    assert ref.x[0] == 1.0 and ref.y[0] == 2.0 and ref.yaw[0] == 0.5
    assert np.all(ref.kappa[: int(10 / (5.0 * 0.02))] == 0)  # the first 10 m are straight


def test_path_errors_sign_and_heading():
    ref = path_reference(lambda s: 0.0, 50.0, 5.0)  # straight along +x
    lat, head, j = path_errors(ref, 20.0, 1.5, 0.1, hint=200)  # the search covers +-window points around hint
    assert abs(lat - 1.5) < 1e-9 and abs(head - 0.1) < 1e-9 and abs(ref.x[j] - 20.0) <= 0.05
    assert path_errors(ref, 20.0, -0.7, 0.0)[0] < 0  # right of the path: negative


def test_pure_pursuit_steers_back():
    ref = path_reference(lambda s: 0.0, 50.0, 5.0)
    obs = {"x": 10.0, "y": -1.0, "yaw": 0.0, "vx": 4.0}
    s, a = pure_pursuit(ref, obs, int(10.0 / 0.1), wheelbase=3.4, max_steer=0.5)
    assert 0 < s <= 1 and a > 0  # to the left, and speed up toward 5 m/s


def test_axle_forces_inverts_the_bicycle_model():
    rng = np.random.default_rng(0)
    m, I_z, a, b = 2500.0, 4500.0, 1.6, 1.7
    n = 500
    vx, vy, r, delta = 4 + rng.random(n), 0.3 * rng.standard_normal(n), 0.3 * rng.standard_normal(n), 0.2 * rng.standard_normal(n)
    af = np.arctan((vy + a * r) / vx) - delta
    ar = np.arctan((vy - b * r) / vx)
    fyf, fyr = -9e4 * af, -1e5 * ar
    ay = (fyf * np.cos(delta) + fyr) / m
    rdot = (a * fyf * np.cos(delta) - b * fyr) / I_z
    f = axle_forces({"vx": vx, "vy": vy, "r": r, "delta": delta, "ay": ay, "rdot": rdot}, m, I_z, a, b)
    assert np.allclose(f["fyf"], fyf) and np.allclose(f["fyr"], fyr) and np.allclose(f["alpha_f"], af)


def test_linear_with_peak():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vehicle"))
    pytest.importorskip("casadi")
    from chrono_g1 import linear_with_peak

    al = np.linspace(-0.05, 0.05, 101)
    tyre = linear_with_peak(al, -8e4 * al, 1e4)
    assert abs(tyre.B * tyre.C * tyre.D + 8e4) < 1e-6 and tyre.D == -1e4 and tyre.E == 0
    assert abs(tyre.force(np.array([0.5]))[0]) <= 1e4 + 1e-6  # saturates at the given peak


@pytest.mark.skipif(not os.environ.get("SUPERMIMIC_CHRONO_PYTHON"), reason="needs a PyChrono interpreter")
def test_chrono_pool_runs_and_turns_left():
    from supermimic.vehicles.chrono_env import ChronoPool

    with ChronoPool(2) as pool:
        obs = pool.reset([{"terrain": "rigid", "v": 4.0, "settle": 1.5}, {"terrain": "scm", "soil": "soft", "v": 4.0, "settle": 1.5}])
        info = pool.info()[0]
        # a + b come from the actual spindle positions (3.30 m at rest); GetWheelbase() is the design value (3.378 m)
        assert 2000 < info["mass"] < 3000 and abs(info["a"] + info["b"] - info["wheelbase"]) < 0.1
        for _ in range(40):
            obs = pool.step([0.5, 0.5], [0.0, 0.0])
        for o in obs:
            assert o["ok"] and o["vx"] > 2.0 and o["delta"] > 0.15 and o["r"] > 0.1  # positive command: left turn
            assert math.isfinite(o["ay"]) and abs(o["t"] - 2.3) < 1e-6  # settle 1.5 s + 40 control steps
