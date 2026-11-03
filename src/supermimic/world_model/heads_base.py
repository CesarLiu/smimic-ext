"""Robot-agnostic interface of the world model's analytic heads.

The analytic heads are the parameter-free part of the world model: differentiable functions of the
(predicted) state that give the policy observation, the reward, the termination features and the
critic's privileged features. Together with the state spec (``spec_base.py``) they are everything
the world model and the imagination need to know about one robot and its task.

Contract (``tests/test_heads_interface.py`` checks it for every implementation):

* ``observation(s, prev_action, xi)``: noise-free policy observation (…, obs_dim);
* ``obs_noise_half()``: (obs_dim,) half-widths of the uniform sensor noise re-added in imagination;
* ``flatten_obs_history(hist)``: (H, N, obs_dim) oldest→newest -> (N, H * obs_dim) policy input;
* ``evaluate(s_next, action, prev_action, xi, ref)``: everything one imagined step needs from the
  heads, computed together so shared work (e.g. forward kinematics) runs once: the reward terms
  (a dict with ``total`` and every name in ``reward_term_names``), the next noise-free
  observation, and an optional auxiliary tensor for the goal side (``None`` if unused);
* ``state_termination_features(s, ref)``: (…, term_feat_dim) inputs of the learned termination head;
* ``termination_feature_scale(wm_cfg)``: per-feature scale of those inputs (typically 1/threshold);
* ``critic_features(s, ref)``: (…, critic_feat_dim) privileged critic inputs;
* ``max_step_reward``: the largest per-step reward (normalises the reported return).

``ref`` is the task's reference for one target step. It must be a dataclass whose fields are tensors
(or ``None``): compiled code passes it as a flat tuple of tensors (``ref_tensors`` /
``ref_from_tensors``) so the compiled graphs see only tensors.

The G1 motion-tracking implementation is ``supermimic.world_model.heads.AnalyticHeads``.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import NamedTuple

import torch


class HeadsStep(NamedTuple):
    reward_terms: dict[str, torch.Tensor]  # includes "total"
    obs: torch.Tensor  # next noise-free observation (n, obs_dim)
    aux: torch.Tensor | None  # goal-side auxiliary output (G1: anchor body pose), or None


class AnalyticHeadsBase(ABC):
    obs_dim: int
    reward_term_names: tuple[str, ...]
    term_feat_dim: int
    critic_feat_dim: int

    @property
    @abstractmethod
    def max_step_reward(self) -> float:
        """Largest attainable per-step reward."""

    @abstractmethod
    def observation(self, s: torch.Tensor, prev_action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Noise-free observation (…, obs_dim) of state ``s`` after ``prev_action``."""

    @abstractmethod
    def obs_noise_half(self) -> torch.Tensor:
        """(obs_dim,) half-widths of the uniform observation noise (CPU tensor)."""

    @abstractmethod
    def flatten_obs_history(self, hist: torch.Tensor) -> torch.Tensor:
        """(H, N, obs_dim) oldest→newest -> (N, H * obs_dim) in the policy's input layout."""

    @abstractmethod
    def evaluate(self, s_next: torch.Tensor, action: torch.Tensor, prev_action: torch.Tensor, xi: torch.Tensor, ref) -> HeadsStep:
        """Reward r_{t+1} of reaching ``s_next`` under ``ref``, the next observation and the aux output."""

    @abstractmethod
    def state_termination_features(self, s: torch.Tensor, ref) -> torch.Tensor:
        """(…, term_feat_dim) features the learned termination head classifies."""

    @abstractmethod
    def termination_feature_scale(self, wm_cfg) -> tuple[float, ...]:
        """(term_feat_dim,) scale applied to the termination features; may read ``wm_cfg``."""

    @abstractmethod
    def critic_features(self, s: torch.Tensor, ref) -> torch.Tensor:
        """(…, critic_feat_dim) privileged critic inputs."""


def ref_tensors(ref) -> tuple:
    """A reference dataclass as the flat tuple of its fields (tensors or None), in field order."""
    return tuple(getattr(ref, f.name) for f in dataclasses.fields(ref))


def ref_from_tensors(cls, tensors) -> object:
    """Inverse of ``ref_tensors``."""
    return cls(*tensors)
