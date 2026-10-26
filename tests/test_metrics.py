"""Anchor-aligned body error: invariant to global xy / yaw drift, sensitive to height and local errors."""

from __future__ import annotations

import math

import torch

from supermimic.eval.metrics import anchor_aligned_body_error


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], dim=-1)


def _rot_z(p: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    x, y, z = p.unbind(-1)
    return torch.stack([c * x - s * y, s * x + c * y, z], dim=-1)


def _reference(n: int = 6, nb: int = 14, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    body = torch.randn(n, nb, 3, generator=g, dtype=torch.float64) * 0.4
    body[..., 2] += 0.8
    ai = 7
    # A tilted anchor orientation: only its yaw may be aligned away.
    axis = torch.nn.functional.normalize(torch.randn(n, 3, generator=g, dtype=torch.float64), dim=-1)
    ang = torch.rand(n, generator=g, dtype=torch.float64) * 0.6
    quat = torch.cat([torch.cos(ang / 2)[:, None], torch.sin(ang / 2)[:, None] * axis], dim=-1)
    return body, ai, quat


def _err(robot_body, robot_quat, ref_body, ref_quat, ai):
    return anchor_aligned_body_error(robot_body, robot_body[:, ai], robot_quat, ref_body, ref_body[:, ai], ref_quat)


def test_zero_for_identical_poses():
    body, ai, quat = _reference()
    assert _err(body, quat, body, quat, ai).abs().max() < 1e-12


def test_invariant_to_xy_and_yaw_drift():
    body, ai, quat = _reference()
    yaw = torch.linspace(-math.pi, math.pi, body.shape[0], dtype=torch.float64)
    shift = torch.tensor([3.0, -7.5, 0.0], dtype=torch.float64)
    robot = _rot_z(body.reshape(-1, 3), yaw.repeat_interleave(body.shape[1])).reshape(body.shape) + shift
    from supermimic.kinematics.rotations import quat_mul

    robot_quat = quat_mul(_yaw_quat(yaw), quat)
    assert _err(robot, robot_quat, body, quat, ai).abs().max() < 1e-9


def test_counts_height_error():
    body, ai, quat = _reference()
    robot = body + torch.tensor([0.0, 0.0, -0.05], dtype=torch.float64)
    torch.testing.assert_close(_err(robot, quat, body, quat, ai), torch.full((body.shape[0],), 0.05, dtype=torch.float64))


def test_counts_local_body_error():
    body, ai, quat = _reference()
    robot = body.clone()
    robot[:, 0, 0] += 0.14  # one non-anchor body off by 14 cm
    torch.testing.assert_close(_err(robot, quat, body, quat, ai), torch.full((body.shape[0],), 0.14 / body.shape[1], dtype=torch.float64))
