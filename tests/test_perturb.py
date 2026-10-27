"""Perturbation helpers: exact inverse of StateSpec.error_vector, masks, margin extraction."""

from __future__ import annotations

import math

import numpy as np
import torch

from supermimic.eval.perturb import apply_state_delta, component_mask, margins_from_outcomes, random_directions
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg


def _states(n: int = 16, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(n, 71, generator=g, dtype=torch.float64) * 0.3
    s[:, 3:7] = torch.nn.functional.normalize(s[:, 3:7] + torch.tensor([2.0, 0, 0, 0], dtype=torch.float64), dim=-1)
    return s


def test_apply_state_delta_inverts_error_vector():
    spec = StateSpec(StateSpecCfg(), "cpu", torch.float64)
    s = _states()
    delta = torch.randn(s.shape[0], 70, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    s2 = apply_state_delta(s, delta, spec.err_scale)
    torch.testing.assert_close(spec.error_vector(s2, s), delta, atol=1e-9, rtol=1e-9)


def test_zero_delta_is_identity_and_limits_clamp():
    spec = StateSpec(StateSpecCfg(), "cpu", torch.float64)
    s = _states()
    torch.testing.assert_close(apply_state_delta(s, torch.zeros(s.shape[0], 70, dtype=torch.float64), spec.err_scale), s)
    limits = torch.stack([torch.full((29,), -0.1, dtype=torch.float64), torch.full((29,), 0.1, dtype=torch.float64)], -1)
    big = torch.zeros(s.shape[0], 70, dtype=torch.float64)
    big[:, 12:41] = 50.0
    out = apply_state_delta(s, big, spec.err_scale, limits)
    assert out[:, 7:36].max() <= 0.1 + 1e-12


def test_masks_partition_the_error_vector():
    pos, vel, full = component_mask("pos"), component_mask("vel"), component_mask("all")
    assert torch.equal(pos + vel, full) and full.sum() == 70 and (pos * vel).sum() == 0
    d = random_directions(32, 70, vel, seed=3)
    torch.testing.assert_close(d.norm(dim=-1), torch.ones(32))
    assert (d * pos).abs().max() == 0


def test_margins_from_outcomes():
    scales = [0.0, 0.5, 1.0, 2.0]
    failed = np.array([
        [False, False, True, True],  # margin 1.0
        [False, False, False, False],  # never fails -> inf
        [False, True, False, True],  # margin 0.5, non-monotonic
        [True, True, True, True],  # fails even unperturbed
    ])
    out = margins_from_outcomes(failed, scales)
    assert out["margin"].tolist() == [1.0, math.inf, 0.5, 0.0]
    assert out["nonmonotonic"].tolist() == [False, False, True, False]
