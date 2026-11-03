"""Counterfactual state / action perturbations for robustness margins (scripts/eval_margins.py).

Perturbations live in the world model's normalised error space (``StateSpec.error_vector``): for the
G1, a 70-d vector [dp(3), rotvec(3), dv(3), dw(3), dq(29), dqd(29)] divided by the nominal scales
``err_scale``; other robots define their own through ``StateSpecBase``.
A state perturbation of norm r is therefore directly comparable with a world-model prediction error of
norm r, which is the comparison the margins are for.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from supermimic.world_model.spec_base import StateSpecBase
from supermimic.world_model.state_spec import ERR_DIM, ERR_POS, ERR_VEL, g1_apply_error

# Kept for existing imports; the groups are defined by the G1 state spec.
_POS = ERR_POS  # dp, rotvec, dq
_VEL = ERR_VEL  # dv, dw, dqd


def component_mask(components: str, device=None, spec=None) -> torch.Tensor:
    """(err_dim,) float mask: ``vel`` (root and joint velocities), ``pos`` (root pose and joint angles) or ``all``.

    ``spec`` (a ``StateSpecBase``) selects another robot's error groups; the default is the G1 layout.
    """
    if spec is not None:
        return spec.component_mask(components, device)
    if components not in ("pos", "vel", "all"):
        raise ValueError(f"components must be pos, vel or all, got {components!r}")
    m = torch.zeros(ERR_DIM, device=device)
    if components in ("pos", "all"):
        m[_POS] = 1.0
    if components in ("vel", "all"):
        m[_VEL] = 1.0
    return m


def apply_state_delta(s: torch.Tensor, delta: torch.Tensor, err_scale: torch.Tensor, joint_limits: torch.Tensor | None = None) -> torch.Tensor:
    """State s' with ``error_vector(s', s) == delta`` for the G1 layout (see ``g1_apply_error``).

    Robot-agnostic code should call ``spec.apply_error(s, delta, limits)`` instead.
    """
    return g1_apply_error(s, delta, err_scale, joint_limits)


def perturb_state(s: torch.Tensor, delta: torch.Tensor, spec, limits: torch.Tensor | None = None) -> torch.Tensor:
    """State s' with ``error_vector(s', s) == delta`` (normalised error units).

    ``spec`` is the robot's ``StateSpecBase``; the G1 ``err_scale`` tensor is still accepted (the
    previous call style). ``limits`` clamps what the spec clamps (G1: joint angles).
    """
    if isinstance(spec, StateSpecBase):
        return spec.apply_error(s, delta, limits)
    return g1_apply_error(s, delta, spec, limits)


def random_directions(n: int, dim: int, mask: torch.Tensor | None = None, seed: int = 0) -> torch.Tensor:
    """(n, dim) unit vectors, uniform on the sphere of the masked subspace (CPU, reproducible)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, dim, generator=g)
    if mask is not None:
        x = x * mask.cpu()
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def margins_from_outcomes(failed: np.ndarray, scales) -> dict[str, np.ndarray]:
    """Robustness margin per (time, direction) from failure outcomes on a grid of perturbation sizes.

    failed: (..., S) bool, the outcome for each of the S ``scales`` (ascending). Returns
      * ``margin``: smallest scale that failed (inf if none did);
      * ``nonmonotonic``: some scale above the margin survived (contact non-smoothness, or noise; such
        margins are upper bounds of the true boundary only in a loose sense, so they are counted).
    """
    scales = np.asarray(scales, dtype=float)
    assert np.all(np.diff(scales) > 0), "scales must be strictly ascending"
    failed = np.asarray(failed, dtype=bool)
    any_fail = failed.any(-1)
    first = np.where(any_fail, failed.argmax(-1), len(scales) - 1)
    margin = np.where(any_fail, scales[first], math.inf)
    above = np.arange(len(scales)) > first[..., None]
    nonmono = any_fail & (above & ~failed).any(-1)
    return {"margin": margin, "nonmonotonic": nonmono}
