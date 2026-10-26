"""Tracking metrics that do not depend on global drift."""

from __future__ import annotations

import torch

from supermimic.kinematics.rotations import quat_apply, quat_inv, quat_mul, yaw_quat


def anchor_aligned_body_error(
    robot_body_pos: torch.Tensor,
    robot_anchor_pos: torch.Tensor,
    robot_anchor_quat: torch.Tensor,
    ref_body_pos: torch.Tensor,
    ref_anchor_pos: torch.Tensor,
    ref_anchor_quat: torch.Tensor,
) -> torch.Tensor:
    """Mean tracked-body position error [m] after moving the reference onto the robot anchor in xy and yaw.

    Same alignment as the body-position reward (``MultiClipMotionCommand.update_relative_body_poses``):
    the reference keeps its own anchor height, so height errors count, while global xy / yaw drift does
    not. Shapes: bodies (N, B, 3), anchor pos (N, 3), anchor quat (N, 4) wxyz -> (N,).
    """
    n, nb, _ = ref_body_pos.shape
    delta_pos = robot_anchor_pos.clone()
    delta_pos[:, 2] = ref_anchor_pos[:, 2]
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(ref_anchor_quat)))
    rel = (ref_body_pos - ref_anchor_pos[:, None, :]).reshape(-1, 3)
    q = delta_ori[:, None, :].expand(n, nb, 4).reshape(-1, 4)
    ref_aligned = delta_pos[:, None, :] + quat_apply(q, rel).reshape(n, nb, 3)
    return (ref_aligned - robot_body_pos).norm(dim=-1).mean(-1)
