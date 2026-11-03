"""Robot-agnostic interface of the task on the learner side.

The task says what the policy is asked to do at each step of task time: where the target is
(the reference the analytic heads score the state against), what the policy is told about it (the
goal features), and how task time advances. Task time is a global step index (``idx``): for motion
tracking, a frame of the concatenated motion library; for a vehicle, a point of a concatenated set
of reference trajectories. Replay-buffer starts carry it (``StartBatch.frame``).

Contract (``tests/test_task_interface.py`` checks it for every implementation):

* ``targets(idx, horizon)``: (horizon+1, n) target steps idx, idx+1, … clamped at the end of each
  sequence;
* ``reference(idx)``: the reference at the (flat) steps ``idx``, a dataclass of tensors batched over
  ``idx`` that the heads' ``evaluate`` / ``state_termination_features`` / ``critic_features`` accept;
* ``termination_reference(idx)``: the part of the reference the termination features need (world-model
  fitting asks for it every window; defaults to ``reference``);
* ``goal(idx, ref, aux, noise)``: (n, goal_dim) policy goal. ``aux`` is the heads' auxiliary output at
  the current state (``AnalyticHeadsBase.aux`` / ``HeadsStep.aux``), requested only if ``needs_aux``;
  ``noise`` asks for the goal-side observation noise, if the task has any.

The reset distribution lives on the simulator side (``tasks.reset_sampler``). The G1 motion-tracking
implementation is ``supermimic.tasks.motion_tracking.MotionTrackingTask``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class TaskBase(ABC):
    goal_dim: int
    needs_aux: bool = False

    @abstractmethod
    def targets(self, idx: torch.Tensor, horizon: int) -> torch.Tensor:
        """(horizon+1, n) target steps from ``idx`` (n,), clamped at the end of each sequence."""

    @abstractmethod
    def reference(self, idx: torch.Tensor):
        """Reference dataclass batched over the flat steps ``idx``."""

    def termination_reference(self, idx: torch.Tensor):
        """The reference fields the termination features read (others may be None)."""
        return self.reference(idx)

    @abstractmethod
    def goal(self, idx: torch.Tensor, ref, aux: torch.Tensor | None, noise: bool) -> torch.Tensor:
        """(n, goal_dim) goal features for target steps ``idx`` with reference ``ref``."""
