"""G1 motion-tracking analytic heads of the world model (plan D4): observation, reward, termination.

``AnalyticHeads`` implements ``AnalyticHeadsBase`` (``heads_base.py``, the robot-agnostic contract).
Everything here is a pure differentiable function of the 71-d simulator state
``s = [qpos(36) | qvel(35)]`` (root position expressed relative to the env origin,
see ``StateIO``), the previous clean action, the sim context ``xi`` and the
reference frame, so the same code serves the simulator-side tests and the
imagined rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.envs.mdp.rewards import TrackingRewardCfg, tracking_reward_terms
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import projected_gravity, quat_apply_inverse
from supermimic.world_model.heads_base import AnalyticHeadsBase, HeadsStep

OBS_DIM = 93
# Term layout of the ``policy`` observation group (order matters: term-major history).
OBS_TERMS = (("joint_pos", 29), ("joint_vel", 29), ("base_ang_vel", 3), ("projected_gravity", 3), ("actions", 29))
# Paper Table A.1 sensor noise (uniform half-widths) in the 93-d observation order.
OBS_NOISE_HALF = torch.tensor([0.01] * 29 + [0.5] * 29 + [0.2] * 3 + [0.05] * 3 + [0.0] * 29)
# Reward terms reported per imagined step (order of the diagnostics vector).
REWARD_TERM_NAMES = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "action_rate_l2",
    "joint_limit",
)


def split_state(s: torch.Tensor):
    """-> root_pos (…,3), root_quat (…,4), joint_pos (…,29), root_lin_vel_w (…,3), root_ang_vel_b (…,3), joint_vel (…,29)."""
    return s[..., 0:3], s[..., 3:7], s[..., 7:36], s[..., 36:39], s[..., 39:42], s[..., 42:71]


def analytic_observation(
    s: torch.Tensor, prev_action: torch.Tensor, xi: torch.Tensor, default_joint_pos: torch.Tensor
) -> torch.Tensor:
    """Noise-free 93-d proprioceptive observation (paper Table A.1) from state + context.

    xi[..., 1:30] is the encoder bias: observed joint positions are ``q + bias - q0``.
    """
    _, root_quat, q, _, ang_b, qd = split_state(s)
    bias = xi[..., 1:30]
    return torch.cat([q + bias - default_joint_pos, qd, ang_b, projected_gravity(root_quat), prev_action], dim=-1)


def anchor_relative_obs(robot_anchor_pos: torch.Tensor, robot_anchor_quat: torch.Tensor, ref_anchor_pos: torch.Tensor, ref_anchor_quat: torch.Tensor) -> torch.Tensor:
    """Reference anchor pose expressed in the robot anchor frame: [pos (3), rot6d (6)] (= mjlab's
    motion_anchor_pos_b / motion_anchor_ori_b, plan D5 alternative)."""
    from supermimic.kinematics.rotations import matrix_from_quat, quat_inv, quat_mul

    q10 = quat_inv(robot_anchor_quat)
    pos = quat_apply_inverse(robot_anchor_quat, ref_anchor_pos - robot_anchor_pos)
    q12 = quat_mul(q10, ref_anchor_quat)
    mat = matrix_from_quat(q12)
    return torch.cat([pos, mat[..., :2].reshape(*q12.shape[:-1], 6)], dim=-1)


ANCHOR_OBS_DIM = 9  # full [pos, rot6d]; the "ori" mode keeps the last 6 (supermimic/anchor_obs.py)


def flatten_history_term_major(hist: torch.Tensor) -> torch.Tensor:
    """(H, N, 93) oldest→newest  ->  (N, 930) in mjlab's term-major layout [A_t0..A_tH-1, B_t0..]."""
    H, N, _ = hist.shape
    chunks = []
    start = 0
    for _, d in OBS_TERMS:
        chunks.append(hist[:, :, start : start + d].permute(1, 0, 2).reshape(N, H * d))
        start += d
    return torch.cat(chunks, dim=-1)


@dataclass
class ReferenceFrame:
    """Reference quantities of one target frame for a batch (N, ·)."""

    body_pos_w: torch.Tensor  # (N, B, 3)
    body_quat_w: torch.Tensor  # (N, B, 4)
    body_lin_vel_w: torch.Tensor  # (N, B, 3)
    body_ang_vel_w: torch.Tensor  # (N, B, 3)
    root_pos_w: torch.Tensor  # (N, 3)
    root_quat_w: torch.Tensor  # (N, 4)


class AnalyticHeads(AnalyticHeadsBase):
    """Reward / termination features computed from the predicted state through differentiable FK."""

    obs_dim = OBS_DIM
    reward_term_names = REWARD_TERM_NAMES
    term_feat_dim = 2  # [height error, tilt error]
    critic_feat_dim = 4  # root position error (3) + yaw error (1), see root_error_features

    def __init__(
        self,
        fk: G1Kinematics,
        tracked_body_names: tuple[str, ...],
        anchor_body_name: str,
        root_body_name: str,
        soft_limits: torch.Tensor,
        default_joint_pos: torch.Tensor,
        reward_cfg: TrackingRewardCfg | None = None,
    ):
        self.fk = fk
        self.tracked_idx = fk.body_indices(tracked_body_names)
        self.anchor_index = tracked_body_names.index(anchor_body_name)
        self.root_fk_index = fk.body_index(root_body_name)
        self.soft_limits = soft_limits
        self.default_joint_pos = default_joint_pos
        self.reward_cfg = reward_cfg or TrackingRewardCfg()

    # ------------------------------------------------------------------ AnalyticHeadsBase

    @property
    def max_step_reward(self) -> float:
        return self.reward_cfg.max_step_reward

    def observation(self, s: torch.Tensor, prev_action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        return analytic_observation(s, prev_action, xi, self.default_joint_pos)

    def obs_noise_half(self) -> torch.Tensor:
        return OBS_NOISE_HALF

    def flatten_obs_history(self, hist: torch.Tensor) -> torch.Tensor:
        return flatten_history_term_major(hist)

    def evaluate(self, s_next, action, prev_action, xi, ref: ReferenceFrame) -> HeadsStep:
        """FK once; reward terms, next observation and the anchor body pose (n, 7) for anchor goals."""
        bodies = self.robot_bodies(s_next)
        terms = self.reward(s_next, action, prev_action, ref, bodies)
        obs = self.observation(s_next, action, xi)
        ai = self.anchor_index
        anchor_pose = torch.cat([bodies["pos"][..., ai, :], bodies["quat"][..., ai, :]], dim=-1)
        return HeadsStep(terms, obs, anchor_pose)

    def state_termination_features(self, s: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        # The pelvis is the free-joint body, so its pose is the state's root pose: no FK needed.
        return self.termination_features(s[..., 0:3], s[..., 3:7], ref)

    def termination_feature_scale(self, wm_cfg) -> tuple[float, ...]:
        return (1.0 / wm_cfg.height_threshold, 1.0 / wm_cfg.tilt_threshold)

    def critic_features(self, s: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        return self.root_error_features(s, ref)

    def anchor_pose(self, s: torch.Tensor) -> torch.Tensor:
        """(n, 7) world pose of the anchor body (G1 anchor goals; the same quantity as ``evaluate``'s aux)."""
        b = self.robot_bodies(s)
        ai = self.anchor_index
        return torch.cat([b["pos"][..., ai, :], b["quat"][..., ai, :]], dim=-1)

    # ------------------------------------------------------------------ G1 helpers

    def robot_bodies(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.fk(s[..., :36], s[..., 36:])
        idx = self.tracked_idx
        return {
            "pos": out["pos"][..., idx, :],
            "quat": out["quat"][..., idx, :],
            "lin_vel": out["lin_vel"][..., idx, :],
            "ang_vel": out["ang_vel"][..., idx, :],
            "root_pos": out["pos"][..., self.root_fk_index, :],
            "root_quat": out["quat"][..., self.root_fk_index, :],
        }

    def reward(self, s_next: torch.Tensor, action: torch.Tensor, prev_action: torch.Tensor, ref: ReferenceFrame, bodies=None):
        """Reward r_{t+1} of reaching s_next (= s_{t+1}) under target frame g_t. Returns dict with 'total'."""
        b = bodies if bodies is not None else self.robot_bodies(s_next)
        return tracking_reward_terms(
            robot_body_pos_w=b["pos"],
            robot_body_quat_w=b["quat"],
            robot_body_lin_vel_w=b["lin_vel"],
            robot_body_ang_vel_w=b["ang_vel"],
            ref_body_pos_w=ref.body_pos_w,
            ref_body_quat_w=ref.body_quat_w,
            ref_body_lin_vel_w=ref.body_lin_vel_w,
            ref_body_ang_vel_w=ref.body_ang_vel_w,
            anchor_index=self.anchor_index,
            action=action,
            prev_action=prev_action,
            joint_pos=s_next[..., 7:36],
            soft_limits=self.soft_limits,
            cfg=self.reward_cfg,
        )

    @staticmethod
    def termination_features(root_pos: torch.Tensor, root_quat: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        """[height error, tilt error] (N, 2) — the quantities the paper thresholds (0.25 m, 0.8)."""
        g = torch.tensor([0.0, 0.0, -1.0], dtype=root_quat.dtype, device=root_quat.device).expand(root_quat.shape[:-1] + (3,))
        tilt = quat_apply_inverse(ref.root_quat_w, g)[..., 2] - quat_apply_inverse(root_quat, g)[..., 2]
        height = ref.root_pos_w[..., 2] - root_pos[..., 2]
        return torch.stack([height, tilt], dim=-1)

    def terminated(self, s: torch.Tensor, ref: ReferenceFrame, height_thr: float = 0.25, tilt_thr: float = 0.8) -> torch.Tensor:
        f = self.termination_features(s[..., 0:3], s[..., 3:7], ref)  # pelvis == free-joint root
        return (f[..., 0].abs() > height_thr) | (f[..., 1].abs() > tilt_thr)

    @staticmethod
    def root_error_features(s: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        """Critic-only privileged drift features (plan D10): root position error in the reference
        heading frame (3) and yaw error (1). (N, 4)"""
        from supermimic.kinematics.rotations import heading_yaw, quat_apply_inverse, wrap_to_pi, yaw_quat

        dp = quat_apply_inverse(yaw_quat(ref.root_quat_w), s[..., 0:3] - ref.root_pos_w)
        dyaw = wrap_to_pi(heading_yaw(s[..., 3:7]) - heading_yaw(ref.root_quat_w))
        return torch.cat([dp, dyaw[..., None]], dim=-1)
