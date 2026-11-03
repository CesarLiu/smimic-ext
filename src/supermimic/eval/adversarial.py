"""Worst-case perturbation directions from the differentiable world model (``--kind adv`` margins).

SuperMimic's policy is trained on gradients of the imagined return through the world model, so the same
model can say which small state change hurts the policy most: the steepest-descent direction of the
imagined return with respect to a perturbation of the start state. Whether the simulator agrees (the
margin along that direction is smaller than along random ones) and where the model expects the policy to
fail (its own predicted margin) are the questions ``--kind adv`` answers.

Both functions take the imagined rollout as callables on batches of start states, so they carry no
simulator imports and can be tested with toy objectives:
  * ``objective(states) -> (n,)``: imagined return, differentiable w.r.t. ``states``;
  * ``fails(states) -> (n,) bool``: whether the imagined rollout from each state fails.
"""

from __future__ import annotations

import math

import torch

from supermimic.eval.perturb import perturb_state


def worst_case_direction(objective, s: torch.Tensor, mask: torch.Tensor, spec) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit direction (normalised error space, inside ``mask``) of steepest decrease of ``objective`` at s.

    s (n, state_dim); ``spec`` is the robot's ``StateSpecBase`` (or the G1 ``err_scale`` tensor). Returns
    (directions (n, err_dim), gradient norms (n,)). The first-order direction at zero perturbation; a zero
    gradient (flat objective) gives a zero direction, which callers treat as invalid.
    """
    with torch.enable_grad():
        delta = torch.zeros(s.shape[0], mask.numel(), device=s.device, dtype=s.dtype, requires_grad=True)
        J = objective(perturb_state(s.detach(), delta * mask, spec))
        (g,) = torch.autograd.grad(J.sum(), delta)
    g = -torch.nan_to_num(g) * mask
    norm = g.norm(dim=-1)
    return g / norm.clamp_min(1e-12)[:, None], norm


@torch.no_grad()
def predicted_margin(fails, s: torch.Tensor, direction: torch.Tensor, scales, spec, joint_limits: torch.Tensor | None = None) -> torch.Tensor:
    """The model's own margin: smallest of ``scales`` (ascending) at which ``fails`` fires along ``direction``.

    s (n, state_dim), direction (n, err_dim); ``spec`` as in ``worst_case_direction``. All n * len(scales)
    start states go through ``fails`` in one batch. Returns (n,) floats, inf where no scale fails.
    """
    n, k = s.shape[0], len(scales)
    sc = torch.as_tensor(scales, device=s.device, dtype=s.dtype)
    starts = perturb_state(s.repeat_interleave(k, 0), direction.repeat_interleave(k, 0) * sc.repeat(n)[:, None], spec, joint_limits)
    f = fails(starts).reshape(n, k)
    first = torch.where(f.any(-1), f.float().argmax(-1), torch.full((n,), k - 1, device=s.device))
    return torch.where(f.any(-1), sc[first], torch.full((n,), math.inf, device=s.device, dtype=s.dtype))
