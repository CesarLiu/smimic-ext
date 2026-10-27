"""Counterfactual state / action perturbations for robustness margins (scripts/eval_margins.py).

Perturbations live in the world model's normalised error space (``StateSpec.error_vector``): a 70-d
vector [dp(3), rotvec(3), dv(3), dw(3), dq(29), dqd(29)] divided by the nominal scales ``err_scale``.
A state perturbation of norm r is therefore directly comparable with a world-model prediction error of
norm r, which is the comparison the margins are for.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from supermimic.kinematics.rotations import quat_from_rotvec, quat_mul, quat_normalize

ERR_DIM = 70
# Index groups of the 70-d error vector.
_POS = list(range(0, 6)) + list(range(12, 41))  # dp, rotvec, dq
_VEL = list(range(6, 12)) + list(range(41, 70))  # dv, dw, dqd


def component_mask(components: str, device=None) -> torch.Tensor:
    """(70,) float mask: ``vel`` (root and joint velocities), ``pos`` (root pose and joint angles) or ``all``."""
    m = torch.zeros(ERR_DIM, device=device)
    if components in ("pos", "all"):
        m[_POS] = 1.0
    if components in ("vel", "all"):
        m[_VEL] = 1.0
    if components not in ("pos", "vel", "all"):
        raise ValueError(f"components must be pos, vel or all, got {components!r}")
    return m


def apply_state_delta(s: torch.Tensor, delta: torch.Tensor, err_scale: torch.Tensor, joint_limits: torch.Tensor | None = None) -> torch.Tensor:
    """State s' with ``error_vector(s', s) == delta`` (the inverse of ``StateSpec.error_vector``).

    s (N, 71) = [root_pos, root_quat wxyz, joint_pos, root_lin_vel_w, root_ang_vel_b, joint_vel];
    delta (N, 70) normalised; err_scale (70,). The rotation is applied in the body frame
    (q' = q ⊗ exp(r)), matching ``error_vector``'s log(q^-1 ⊗ q'). Joint angles are clamped to
    ``joint_limits`` (29, 2) if given, so the round trip is exact only inside the limits.
    """
    d = delta * err_scale
    p, q, qj, v, w, qd = s[..., 0:3], s[..., 3:7], s[..., 7:36], s[..., 36:39], s[..., 39:42], s[..., 42:71]
    q_new = quat_normalize(quat_mul(quat_normalize(q), quat_from_rotvec(d[..., 3:6])))
    qj_new = qj + d[..., 12:41]
    if joint_limits is not None:
        qj_new = torch.minimum(torch.maximum(qj_new, joint_limits[:, 0]), joint_limits[:, 1])
    return torch.cat([p + d[..., 0:3], q_new, qj_new, v + d[..., 6:9], w + d[..., 9:12], qd + d[..., 41:70]], dim=-1)


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
