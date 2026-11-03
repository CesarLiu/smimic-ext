"""Motion tracking (the paper's default interface) as a ``TaskBase``: references and goal windows from
a ``MotionLibrary``, optionally with the anchor-relative goal term (``--env.anchor-obs``)."""

from __future__ import annotations

import torch

from supermimic.anchor_obs import ANCHOR_OBS_SLICES, anchor_obs_dim, anchor_obs_mode
from supermimic.tasks.base import TaskBase
from supermimic.world_model.heads import ReferenceFrame, anchor_relative_obs

GOAL_FRAME_DIM = 68  # per-frame goal features (motion_library.goal_features)
# Uniform half-widths of the anchor-relative goal noise: position (3), rot6d (6).
ANCHOR_NOISE_HALF = torch.tensor([0.25] * 3 + [0.05] * 6)


def reference_at(lib, frame: torch.Tensor) -> ReferenceFrame:
    r = lib.ref(frame)
    b = lib.bodies(frame)
    return ReferenceFrame(b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w, r.root_pos_w, r.root_quat_w)


class MotionTrackingTask(TaskBase):
    def __init__(self, lib, future_frames: int = 10, anchor_obs: bool | str = "none", anchor_index: int | None = None, goal_term_scale=(1.0, 5.0, 1.0, 2.0, 1.0, 1.0)):
        self.lib = lib
        self.goal_term_scale = tuple(goal_term_scale)  # PolicyCfg.goal_term_scale defaults
        self.future_frames = future_frames
        self.mode = anchor_obs_mode(anchor_obs)
        self.goal_dim = future_frames * GOAL_FRAME_DIM + anchor_obs_dim(self.mode)
        self.needs_aux = self.mode != "none"
        if self.needs_aux and anchor_index is None:
            raise ValueError("anchor goals need the anchor body's index among the tracked bodies (heads.anchor_index)")
        self.anchor_index = anchor_index
        self._slice = ANCHOR_OBS_SLICES[self.mode]

    def targets(self, idx: torch.Tensor, horizon: int) -> torch.Tensor:
        last = self.lib.last_frame_of(idx)
        steps = torch.arange(horizon + 1, device=idx.device)[:, None]
        return torch.minimum(idx[None, :] + steps, last[None, :])

    def reference(self, idx: torch.Tensor) -> ReferenceFrame:
        return reference_at(self.lib, idx)

    def termination_reference(self, idx: torch.Tensor) -> ReferenceFrame:
        # Only the root pose: the G1 termination features are height and tilt of the root.
        r = self.lib.ref(idx)
        return ReferenceFrame(None, None, None, None, r.root_pos_w, r.root_quat_w)

    def goal_scale(self) -> torch.Tensor:
        from supermimic.policy.nets import g1_goal_scale

        return g1_goal_scale(self.goal_term_scale, self.goal_dim)

    def goal(self, idx: torch.Tensor, ref: ReferenceFrame, aux: torch.Tensor | None, noise: bool) -> torch.Tensor:
        g = self.lib.goal_window(idx, self.future_frames)
        if self.mode == "none":
            return g
        ai = self.anchor_index
        rel = anchor_relative_obs(aux[..., 0:3], aux[..., 3:7], ref.body_pos_w[..., ai, :], ref.body_quat_w[..., ai, :])[..., self._slice]
        if noise:
            rel = rel + (torch.rand_like(rel) * 2 - 1) * ANCHOR_NOISE_HALF.to(rel.device)[self._slice]
        return torch.cat([g, rel], dim=-1)


def as_task(task_or_lib, heads=None, future_frames: int = 10, anchor_obs: bool | str = "none") -> TaskBase:
    """A ``TaskBase`` as is; a motion library wrapped as motion tracking (the previous call style)."""
    if isinstance(task_or_lib, TaskBase):
        return task_or_lib
    mode = anchor_obs_mode(anchor_obs)
    ai = heads.anchor_index if (mode != "none" and heads is not None) else None
    return MotionTrackingTask(task_or_lib, future_frames, mode, ai)
