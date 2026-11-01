"""Worst-case directions and model margins (``--kind adv``) on toy objectives and the stand-in environment."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from supermimic.eval.adversarial import predicted_margin, worst_case_direction
from supermimic.eval.margins import calibration_stats, compare_directions, run_branch_pass
from supermimic.eval.perturb import component_mask, margins_from_outcomes
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

SPEC = StateSpec(StateSpecCfg(), "cpu")
# Error-vector index -> state index for the velocity blocks (dv, dw, dqd).
VEL_ERR_TO_STATE = list(zip(range(6, 12), range(36, 42))) + list(zip(range(41, 70), range(42, 71)))


def _state(n=3):
    s = torch.zeros(n, 71)
    s[:, 3] = 1.0
    s[:, 2] = 0.75
    return s


def test_direction_is_the_normalised_negative_gradient_in_the_mask():
    torch.manual_seed(0)
    w = torch.randn(71)
    objective = lambda st: st @ w  # noqa: E731  linear: the gradient w.r.t. delta is w * err_scale
    mask = component_mask("vel")
    d, gn = worst_case_direction(objective, _state(), mask, SPEC.err_scale)
    expected = torch.zeros(70)
    for e, s in VEL_ERR_TO_STATE:
        expected[e] = -w[s] * SPEC.err_scale[e]
    torch.testing.assert_close(gn, expected.norm().expand(3))
    torch.testing.assert_close(d, (expected / expected.norm()).expand(3, -1))
    assert (d * (1 - mask)).abs().max() == 0


def test_flat_objective_gives_zero_gradient():
    d, gn = worst_case_direction(lambda st: st.sum(-1) * 0.0, _state(1), component_mask("vel"), SPEC.err_scale)
    assert gn.item() == 0.0 and d.abs().max() == 0


def test_predicted_margin_is_the_first_failing_scale():
    s0 = _state(2)

    def fails(st):  # fails once the normalised deviation from the nominal state exceeds 0.75 / 3.0
        dev = SPEC.error_vector(st, s0[:1].expand_as(st)).norm(dim=-1)
        return dev > torch.tensor([0.75, 3.0]).repeat_interleave(st.shape[0] // 2)

    direction = torch.zeros(2, 70)
    direction[:, 41] = 1.0
    pm = predicted_margin(fails, s0, direction, [0.5, 1.0, 2.0, 4.0], SPEC.err_scale)
    assert pm.tolist() == [1.0, 4.0]
    pm_none = predicted_margin(lambda st: torch.zeros(st.shape[0], dtype=torch.bool), s0, direction, [0.5, 1.0], SPEC.err_scale)
    assert all(math.isinf(v) for v in pm_none.tolist())


def test_adv_branches_use_the_cached_direction():
    # The stand-in of test_margins: a branch fails once its deviation from the start state exceeds 0.75.
    from test_margins import THRESHOLD, BiasedModel, StandInEnv, StandInIO
    from types import SimpleNamespace

    times, scales = [8, 33], [0.0, 0.5, 1.0, 2.0]
    specs = [(ti, 0, si) for ti in range(len(times)) for si in range(len(scales))]
    env = StandInEnv(len(specs) + 1)
    cmd = SimpleNamespace(forced_frame=torch.zeros(env.num_envs, dtype=torch.long), library=SimpleNamespace(clip_start=torch.tensor([0])))
    calls = []

    def adv_fn(step, hist_s, hist_a):
        calls.append(step)
        assert len(hist_s) == step + 1 and len(hist_a) == step  # states up to step, actions before it
        d = torch.zeros(70)
        d[41] = 1.0
        return d

    adv_dirs = {}
    policy = lambda o, g: torch.zeros(o.shape[0], 29)  # noqa: E731
    for _ in range(2):  # a second pass (as for clips with more specs than envs) reuses the directions
        res, *_ = run_branch_pass(env, cmd, StandInIO(env), policy, BiasedModel(0.0), torch.zeros(33), 0, times, specs, "adv", scales, None,
                                  component_mask("vel"), SPEC.err_scale, None, 20, 1, 8, adv_fn, adv_dirs)
    assert calls == times
    failed = np.array([f for _, f, _ in res]).reshape(len(times), 1, len(scales))
    offsets = np.array([o for _, _, o in res]).reshape(failed.shape)
    assert (margins_from_outcomes(failed, scales)["margin"] == 1.0).all() and THRESHOLD < 1.0
    assert (offsets[failed] == 0).all() and (offsets[~failed] == -1).all()  # the stand-in fails at the first step


def test_calibration_stats():
    inf = math.inf
    rows = [
        {"model_margin": 2.0, "sim_margin_model_horizon": 4.0},  # within 2x
        {"model_margin": 1.0, "sim_margin_model_horizon": 8.0},  # 8x apart
        {"model_margin": inf, "sim_margin_model_horizon": 4.0},  # missed
        {"model_margin": 4.0, "sim_margin_model_horizon": inf},  # false alarm
        {"model_margin": inf, "sim_margin_model_horizon": inf},  # both fine
        {"margin_min": None},  # no adv data
    ]
    st = calibration_stats(rows)
    assert st["times"] == 5
    assert st["agree"] == pytest.approx(2 / 5) and st["missed"] == pytest.approx(1 / 5) and st["false_alarm"] == pytest.approx(1 / 5)
    assert st["median_log2_model_over_sim"] == pytest.approx(-2.0)  # median of -1 and -3


def test_compare_directions():
    inf = math.inf
    random = {("a", 8): 16.0, ("a", 33): 8.0, ("b", 8): inf, ("b", 33): 4.0, ("c", 8): 2.0}
    adv = {("a", 8): 4.0, ("a", 33): 8.0, ("b", 8): 16.0, ("b", 33): None}
    st = compare_directions(random, adv)
    # keys a8 (log2 4/16 = -2), a33 (0), b8 (16 vs cap 32: -1); b33 has no adv margin, c8 no adv row.
    assert st["times"] == 3
    assert st["median_log2_adv_over_random"] == pytest.approx(-1.0)
    assert st["frac_adv_smaller"] == pytest.approx(2 / 3) and st["frac_adv_at_most_half"] == pytest.approx(2 / 3)
    assert compare_directions({}, {}) == {"times": 0}
