"""Robot-agnostic interface of the world model's state representation.

A state spec defines everything the world model, the imagination rollout and the robustness tools
need to know about one robot's simulator state, without knowing what the numbers mean:

* ``canonical(s)``: network input features (invariances such as heading are removed here);
* ``integrate(s, delta)``: next state from the dynamics network's raw output;
* ``error_vector(s_pred, s_true)``: normalised error, the training loss and the reported metric;
* ``apply_error(s, e)``: the inverse of ``error_vector`` (perturbations in error units);
* ``normalize_xi(xi)``: sim-context normalisation;
* ``error_groups()``: named index groups of the error vector (at least ``pos`` and ``vel``).

The G1 humanoid implementation is ``supermimic.world_model.state_spec.StateSpec``. A new robot
adds a config dataclass deriving from ``StateSpecCfgBase`` whose ``build`` returns its spec; the
world model calls ``make_state_spec(cfg, device)`` and sizes its dynamics network from the spec's
dimensions, so no other world-model code changes. ``tests/test_state_spec_interface.py`` lists the
contract every implementation must satisfy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class StateSpecCfgBase(ABC):
    """Config of one state spec. Concrete configs are dataclasses (tyro CLI, ``asdict`` checkpoints)."""

    dt: float

    @abstractmethod
    def build(self, device, dtype=torch.float32) -> StateSpecBase:
        """Instantiate the state spec this config describes."""


class StateSpecBase(ABC):
    """State representation of one robot (see the module docstring for the contract)."""

    dt: float
    state_dim: int  # simulator state s
    feat_dim: int  # canonical(s), the dynamics network's state input
    delta_dim: int  # dynamics network output consumed by integrate
    err_dim: int  # error_vector
    action_dim: int  # policy action a
    xi_dim: int  # sim context xi

    @abstractmethod
    def canonical(self, s: torch.Tensor) -> torch.Tensor:
        """(…, state_dim) -> (…, feat_dim) scaled, invariant network input features."""

    @abstractmethod
    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        """(…, xi_dim) -> (…, xi_dim)."""

    @abstractmethod
    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """(…, state_dim), (…, delta_dim) -> (…, state_dim). ``delta`` is the raw network output."""

    @abstractmethod
    def error_vector(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """(…, state_dim) x2 -> (…, err_dim), each component divided by its nominal scale."""

    @abstractmethod
    def apply_error(self, s: torch.Tensor, err: torch.Tensor, limits: torch.Tensor | None = None) -> torch.Tensor:
        """State s' with ``error_vector(s', s) == err`` (exact unless ``limits`` clamps it)."""

    @abstractmethod
    def error_groups(self) -> dict[str, list[int]]:
        """Named index groups of the error vector; ``pos`` and ``vel`` must partition it."""

    # ------------------------------------------------------------------ shared helpers

    def normalized_sq_error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Mean over dims of the squared normalised error (the reported 'prediction error')."""
        e = self.error_vector(s_pred, s_true)
        return (e * e).mean(-1)

    def component_mask(self, components: str, device=None) -> torch.Tensor:
        """(err_dim,) float mask over ``pos``, ``vel`` or ``all`` error components."""
        if components not in ("pos", "vel", "all"):
            raise ValueError(f"components must be pos, vel or all, got {components!r}")
        groups = self.error_groups()
        m = torch.zeros(self.err_dim, device=device)
        for name in ("pos", "vel"):
            if components in (name, "all"):
                m[groups[name]] = 1.0
        return m


def make_state_spec(cfg: StateSpecCfgBase, device, dtype=torch.float32) -> StateSpecBase:
    """Build the state spec a config describes (the world model's single entry point)."""
    if not isinstance(cfg, StateSpecCfgBase):
        raise TypeError(f"{type(cfg).__name__} is not a StateSpecCfgBase; derive the config from it and implement build()")
    return cfg.build(device, dtype)
