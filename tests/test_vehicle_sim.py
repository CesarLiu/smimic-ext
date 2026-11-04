"""Vehicle simulator (E0): tyres and full responses against CommonRoad, numerical convergence, the
assists (ABS / ESC), the sim context and batch independence."""

from __future__ import annotations

import math

import pytest
import torch

from supermimic.vehicles import tire
from supermimic.vehicles.dynamics import (
    AY,
    DELTA,
    VX,
    VY,
    AssistCfg,
    Context,
    VehicleDynamics,
    X,
    Y,
)
from supermimic.vehicles.maneuvers import DT, MANEUVERS, Maneuver
from supermimic.vehicles.params import VehicleParams
from supermimic.vehicles.validation import REFERENCE, TOL, compare, run_maneuvers


@pytest.fixture(scope="module")
def reference():
    return torch.load(REFERENCE, weights_only=False)


def test_tyre_equations_match_commonroad(reference):
    g = reference["tyre_grid"]
    fx, fy = tire.forces(g[:, 0], g[:, 1], g[:, 2], g[:, 3], VehicleParams().tire)
    torch.testing.assert_close(fx, g[:, 4], rtol=1e-9, atol=1e-6)
    torch.testing.assert_close(fy, g[:, 5], rtol=1e-9, atol=1e-6)


def test_default_parameters_are_commonroad_vehicle_2():
    pytest.importorskip("vehiclemodels")
    from vehiclemodels.parameters_vehicle2 import parameters_vehicle2

    from supermimic.vehicles.params import from_commonroad

    assert from_commonroad(parameters_vehicle2()) == VehicleParams()


@pytest.fixture(scope="module")
def comparison():
    return compare(substeps=10, dtype=torch.float32)


@pytest.mark.parametrize("name", [m.name for m in MANEUVERS])
def test_matches_the_commonroad_multibody_model(name, comparison):
    """E0 gate: every metric of every manoeuvre within 5 % (float32, 10 substeps, as on the GPU)."""
    ((_man, mt, _),) = [row for row in comparison if row[0].name == name]
    bad = {k: round(v, 4) for k, v in mt.items() if v > TOL}
    assert not bad, f"{name}: {bad}"


def test_substeps_have_converged():
    near_limit = [m for m in MANEUVERS if m.name in ("step_v20_d0.060", "sine_v20_a0.070", "brake_v25_a-6")]
    runs = {k: run_maneuvers(VehicleDynamics(substeps=k, assist=AssistCfg(abs=False, esc=False)), near_limit, torch.float64) for k in (10, 40)}
    for m in near_limit:
        a, b = runs[10][m.name], runs[40][m.name]
        col = 1 if m.kind == "brake" else 3  # vx or r
        err = float(torch.sqrt(((a[:, col] - b[:, col]) ** 2).mean()) / torch.sqrt((b[:, col] ** 2).mean()))
        assert err < 0.01, (m.name, err)


def _brake(abs_on: bool, mu: float = 0.5):
    """Rows: straight braking, braking while steering for 0.3 s. -> distance, max slip above 3 m/s, lateral offset."""
    d = VehicleDynamics(assist=AssistCfg(abs=abs_on, esc=False))
    s, fl, ctx = d.initial_state(torch.full((2,), 25.0)), d.new_flags(2), Context.nominal(2)
    ctx.mu[:] = mu
    max_slip = torch.zeros(2)
    for i in range(400):
        sr = torch.tensor([0.0, 0.4 if i * DT < 0.3 else 0.0])
        s, fl = d.step(s, fl, sr, torch.full((2,), -10.0), ctx)
        fast = s[:, VX] > 3.0
        max_slip = torch.where(fast, torch.maximum(max_slip, d.tyre_state(s, ctx)["kappa"].amax(-1)), max_slip)
        if bool((s[:, VX] < 0.5).all()):
            break
    return float(s[0, X]), float(max_slip[0]), float(s[1, Y])


def test_abs_bounds_slip_shortens_braking_and_keeps_steering():
    d_off, slip_off, y_off = _brake(False)
    d_on, slip_on, y_on = _brake(True)
    assert slip_off > 0.9 and slip_on < 0.3  # locked wheels without ABS; released near the slip peak with it
    assert d_on < 0.8 * d_off
    assert abs(y_off) < 1.0 and y_on > 5.0  # locked front wheels do not steer


def _sine(esc: bool, amp=0.07, mu=0.6):
    d = VehicleDynamics(assist=AssistCfg(abs=True, esc=esc))
    man = Maneuver("limit_sine", 20.0, 4.0, "sine", delta=amp)
    s, fl, ctx = d.initial_state(torch.tensor([20.0])), d.new_flags(1), Context.nominal(1)
    ctx.mu[:] = mu
    beta_max, active = 0.0, 0
    for i in range(man.steps):
        sr, ac = man.inputs(i * DT, float(s[0, VX]), float(s[0, DELTA]))
        s, fl = d.step(s, fl, torch.tensor([sr]), torch.tensor([ac]), ctx)
        beta_max = max(beta_max, abs(math.atan2(float(s[0, VY]), float(s[0, VX]))))
        active += int(fl[0, 4] > 0.5)
    return beta_max, active


def test_esc_limits_sideslip_beyond_the_grip_limit():
    b_off, n_off = _sine(False)
    b_on, n_on = _sine(True)
    assert n_off == 0 and n_on > 0
    assert b_on < 0.6 * b_off


def test_friction_scale_limits_lateral_acceleration():
    """Rows: road friction 1.0 and 0.5, same large steering input; the peak lateral acceleration (from the
    tyre forces) scales with the friction."""
    man = Maneuver("big_steer", 20.0, 4.0, "steady", delta=0.08)
    d = VehicleDynamics(assist=AssistCfg(abs=False, esc=False))
    s, fl, ctx = d.initial_state(torch.full((2,), 20.0)), d.new_flags(2), Context.nominal(2)
    ctx.mu[:] = torch.tensor([1.0, 0.5])
    peak = torch.zeros(2)
    for i in range(man.steps):
        u = [man.inputs(i * DT, float(s[k, VX]), float(s[k, DELTA])) for k in range(2)]
        s, fl = d.step(s, fl, torch.tensor([x[0] for x in u]), torch.tensor([x[1] for x in u]), ctx)
        peak = torch.maximum(peak, s[:, AY].abs())
    assert 0.4 < float(peak[1] / peak[0]) < 0.6
    assert float(peak[1]) < 0.5 * 1.2 * 9.81


def test_rows_are_independent_of_the_batch():
    """Each environment row evolves as if simulated alone (no cross-talk through the batch)."""
    d = VehicleDynamics()
    n = 6
    torch.manual_seed(0)
    s = d.initial_state(10 + 15 * torch.rand(n))
    ctx = Context(0.5 + 0.6 * torch.rand(n), 0.85 + 0.3 * torch.rand(n), 0.9 + 0.25 * torch.rand(n), 0.2 * torch.rand(n) - 0.1)
    sr, ac = 0.4 * torch.rand(n) - 0.2, 8 * torch.rand(n) - 4
    fl = d.new_flags(n)
    batch = (s, fl)
    for _ in range(30):
        batch = d.step(*batch, sr, ac, ctx)
    for k in range(n):
        one = (s[k:k + 1], fl[k:k + 1])
        for _ in range(30):
            one = d.step(*one, sr[k:k + 1], ac[k:k + 1], ctx.index(slice(k, k + 1)))
        torch.testing.assert_close(one[0][0], batch[0][k], rtol=1e-5, atol=1e-5)
