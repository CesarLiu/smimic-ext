"""Branching orchestration of the robustness-margin runs, on a stand-in environment with a known margin.

The stand-in state never moves on its own; a branch fails once its normalised deviation from the start
state exceeds ``THRESHOLD``. With unit perturbation directions the margin is therefore the smallest grid
scale above THRESHOLD, at every branch time.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.eval.margins import branch_times, margin_stats, run_branch_pass
from supermimic.eval.perturb import component_mask, margins_from_outcomes, random_directions
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

THRESHOLD = 0.75
SPEC = StateSpec(StateSpecCfg(), "cpu")


class StandInEnv:
    def __init__(self, n: int):
        self.num_envs, self.device = n, "cpu"
        self.s0 = torch.zeros(71)
        self.s0[3] = 1.0  # identity quaternion
        self.s0[2] = 0.75
        self.state = self.s0.expand(n, 71).clone()

    def _obs(self):
        return {"policy": torch.zeros(self.num_envs, 930), "goal": torch.zeros(self.num_envs, 680)}

    def reset(self):
        self.state = self.s0.expand(self.num_envs, 71).clone()
        return self._obs(), {}

    def step(self, act):
        dev = SPEC.error_vector(self.state, self.s0.expand_as(self.state)).norm(dim=-1)
        term = dev > THRESHOLD
        self.state[term] = self.s0  # auto-reset, as the real env does
        return self._obs(), torch.zeros(self.num_envs), term, torch.zeros_like(term), {}


class StandInIO:
    def __init__(self, env):
        self.env = env

    def read(self):
        return self.env.state.clone()

    def write(self, ids, state):
        self.env.state[ids] = state


class BiasedModel:
    """World model that adds a constant velocity error each step (its k-step error is k * bias)."""

    def __init__(self, bias_norm_per_step: float):
        self.spec = SPEC
        self.bias = torch.zeros(71)
        self.bias[36] = bias_norm_per_step * SPEC.err_scale[6]  # root x velocity, normalised by its scale

    def step(self, s, a, xi):
        return s + self.bias


def _run(kind, scales, num_dirs=4, times=(8, 33), window=20, bias=0.0, n_env=None):
    times = list(times)
    n_dirs = 1 if kind == "wm" else num_dirs
    specs = [(ti, di, si) for ti in range(len(times)) for di in range(n_dirs) for si in range(len(scales))]
    env = StandInEnv(n_env or len(specs) + 1)
    cmd = SimpleNamespace(forced_frame=torch.zeros(env.num_envs, dtype=torch.long), library=SimpleNamespace(clip_start=torch.tensor([0])))
    mask = component_mask("vel")
    dirs = random_directions(num_dirs, 70, mask, seed=0) if kind == "state" else (random_directions(num_dirs, 29, None, 0) if kind == "action" else None)
    policy = lambda o, g: torch.zeros(o.shape[0], 29)  # noqa: E731
    res, err_norm, drift, nominal_fail = run_branch_pass(
        env, cmd, StandInIO(env), policy, BiasedModel(bias), torch.zeros(33), 0, times, specs, kind, scales, dirs, mask, SPEC.err_scale, None, window, 1, 8
    )
    failed = np.zeros((len(times), n_dirs, len(scales)), dtype=bool)
    valid = np.ones_like(failed)
    for (ti, di, si), (v, f) in zip(specs, res):
        valid[ti, di, si], failed[ti, di, si] = v, f
    return failed, valid, err_norm, drift, nominal_fail


def test_state_margin_is_the_first_scale_above_threshold():
    scales = [0.0, 0.5, 1.0, 2.0]
    failed, valid, err_norm, drift, nominal_fail = _run("state", scales)
    assert valid.all() and nominal_fail == -1 and drift == 0.0
    assert not failed[..., 0].any()  # scale-0 control never fails
    m = margins_from_outcomes(failed, scales)
    assert (m["margin"] == 1.0).all() and not m["nonmonotonic"].any()
    assert set(err_norm) == {8, 33} and all(v == 0.0 for v in err_norm.values())  # exact model: zero error


def test_wm_margin_scales_the_models_own_error():
    scales = [0.0, 0.5, 1.0, 2.0, 4.0]
    # k = 8 steps of 0.0625 -> the model's 8-step error has norm 0.5; lambda * 0.5 > 0.75 first at lambda = 2.
    failed, valid, err_norm, *_ = _run("wm", scales, bias=0.0625)
    assert valid.all()
    assert all(math.isclose(v, 0.5, rel_tol=1e-5) for v in err_norm.values())
    m = margins_from_outcomes(failed, scales)
    assert (m["margin"] == 2.0).all()


def test_action_kind_and_idle_envs_leave_the_state_alone():
    # Actions do nothing in the stand-in, so no action branch can fail; extra idle envs act as controls.
    failed, valid, _, drift, _ = _run("action", [0.0, 1.0, 8.0], n_env=64)
    assert valid.all() and not failed.any() and drift == 0.0


def test_branch_times_and_stats():
    assert branch_times(300, 8, 25, 100, 40) == list(range(8, 199, 25))
    assert branch_times(100, 8, 25, 100, 40) == []
    assert len(branch_times(10_000, 8, 25, 100, 40)) == 40
    rows = [
        {"margin_min": 1.0, "model_error_exceeds_margin": False},
        {"margin_min": math.inf, "model_error_exceeds_margin": False},
        {"margin_min": 0.25, "model_error_exceeds_margin": True},
        {"margin_min": None, "model_error_exceeds_margin": None},
    ]
    st = margin_stats(rows)
    assert st["branch_times"] == 4 and st["median_margin_min"] == pytest.approx(1.0)
    assert st["frac_times_with_failure_in_grid"] == pytest.approx(2 / 3) and st["frac_model_error_exceeds_margin"] == pytest.approx(1 / 3)
