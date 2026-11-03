"""G1 implementation of the world model's state representation (plan D1–D3).

Implements ``StateSpecBase`` (``spec_base.py``, the robot-agnostic contract); other robots add
their own spec next to this one.

Simulator state  s = [root_pos(3), root_quat wxyz(4), joint_pos(29), root_lin_vel_w(3),
                      root_ang_vel_b(3), joint_vel(29)]            (71, MuJoCo layout, origin-free)

Canonical features (network input, heading-invariant, 71):
    [root_z(1), rot6d(heading^-1 ⊗ root_quat)(6), root_lin_vel in heading frame(3),
     root_ang_vel_b(3), joint_pos(29), joint_vel(29)]

Network output ``delta`` (70) and the integration step (dt = 0.02 s):
    dv_h(3), dw_b(3), dqd(29)            velocity increments, heading / body frame
    dp_h(3), dth_b(3), dq(29)            position corrections on top of explicit integration
    v'  = v + R_h dv_h               w' = w + dw_b          qd' = qd + dqd
    p'  = p + dt v' + R_h dp_h       q' = q ⊗ exp(dt w' + dth_b)    qj' = qj + dt qd' + dq
With zero output the model extrapolates at constant velocity (a physical prior); the
last layer of the dynamics MLP is zero-initialised so training starts from it.

Loss / error normalisation (D3): per-quantity nominal scales sigma applied to the
70-d error vector [dp(3), rotvec(3), dv(3), dw(3), dq(29), dqd(29)].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from supermimic.kinematics.rotations import (
    heading_quat,
    quat_apply,
    quat_apply_inverse,
    quat_canonical,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)
from supermimic.world_model.spec_base import StateSpecBase, StateSpecCfgBase

STATE_DIM = 71
FEAT_DIM = 71
DELTA_DIM = 70
NJ = 29
XI_DIM = 1 + NJ + 3  # friction, joint encoder bias, torso CoM offset
ERR_DIM = 70

# Index groups of the 70-d error vector [dp(3), rotvec(3), dv(3), dw(3), dq(29), dqd(29)].
ERR_POS = list(range(6)) + list(range(12, 12 + NJ))  # dp, rotvec, dq
ERR_VEL = list(range(6, 12)) + list(range(12 + NJ, ERR_DIM))  # dv, dw, dqd


@dataclass
class StateSpecCfg(StateSpecCfgBase):
    dt: float = 0.02
    # Nominal error scales (D3) for [root pos m, root rotvec rad, root lin vel m/s, root ang vel rad/s, joint pos rad, joint vel rad/s].
    err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    # Output scales of the dynamics head for [dv_h, dw_b, dqd, dp_h, dth_b, dq].
    out_scale: tuple[float, ...] = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
    # Input feature scales for [root_z, rot6d, lin_vel_h, ang_vel_b, joint_pos, joint_vel] (divide by these).
    in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 1.0, 5.0)
    explicit_integration: bool = True
    # xi normalisation: (offset, scale) for friction, encoder bias, com offset.
    xi_offset: tuple[float, ...] = field(default_factory=lambda: (0.75,) + (0.0,) * 29 + (0.0,) * 3)
    xi_scale: tuple[float, ...] = field(default_factory=lambda: (0.45,) + (0.01,) * 29 + (0.05,) * 3)

    def build(self, device, dtype=torch.float32) -> StateSpec:
        return StateSpec(self, device, dtype)


class StateSpec(StateSpecBase):
    """Unitree G1 (29 joints, free-floating root)."""

    state_dim = STATE_DIM
    feat_dim = FEAT_DIM
    delta_dim = DELTA_DIM
    err_dim = ERR_DIM
    action_dim = NJ

    def __init__(self, cfg: StateSpecCfg, device, dtype=torch.float32):
        self.cfg = cfg
        self.dt = cfg.dt
        rep = lambda scales, sizes: torch.cat([torch.full((n,), float(s)) for s, n in zip(scales, sizes)])  # noqa: E731
        self.err_scale = rep(cfg.err_scale, (3, 3, 3, 3, NJ, NJ)).to(device=device, dtype=dtype)
        self.out_scale = rep(cfg.out_scale, (3, 3, NJ, 3, 3, NJ)).to(device=device, dtype=dtype)
        self.in_scale = rep(cfg.in_scale, (1, 6, 3, 3, NJ, NJ)).to(device=device, dtype=dtype)
        self.xi_offset = torch.tensor(cfg.xi_offset, device=device, dtype=dtype)
        self.xi_scale = torch.tensor(cfg.xi_scale, device=device, dtype=dtype)
        self.xi_dim = len(cfg.xi_offset)
        if len(cfg.xi_scale) != self.xi_dim:
            raise ValueError(f"xi_offset has {self.xi_dim} entries but xi_scale has {len(cfg.xi_scale)}")

    # ------------------------------------------------------------------ split / features

    @staticmethod
    def split(s: torch.Tensor):
        return s[..., 0:3], s[..., 3:7], s[..., 7:36], s[..., 36:39], s[..., 39:42], s[..., 42:71]

    def canonical(self, s: torch.Tensor) -> torch.Tensor:
        """Heading-invariant, scaled network input features (…, 71)."""
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        q_local = quat_canonical(quat_mul(quat_apply_inverse_quat(hq), q))
        v_h = quat_apply_inverse(hq, v)
        feat = torch.cat([p[..., 2:3], rot6d_from_quat(q_local), v_h, w, qj, qd], dim=-1)
        return feat / self.in_scale

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.xi_offset) / self.xi_scale

    # ------------------------------------------------------------------ integration

    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Apply the (unscaled) network output ``delta`` (…, 70) to state ``s`` (…, 71)."""
        d = delta * self.out_scale
        dv_h, dw_b, dqd, dp_h, dth_b, dq = d[..., 0:3], d[..., 3:6], d[..., 6:35], d[..., 35:38], d[..., 38:41], d[..., 41:70]
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        v_new = v + quat_apply(hq, dv_h)
        w_new = w + dw_b
        qd_new = qd + dqd
        dt = self.dt if self.cfg.explicit_integration else 0.0
        p_new = p + dt * v_new + quat_apply(hq, dp_h)
        q_new = quat_normalize(quat_mul(q, quat_from_rotvec(dt * w_new + dth_b)))
        qj_new = qj + dt * qd_new + dq
        return torch.cat([p_new, q_new, qj_new, v_new, w_new, qd_new], dim=-1)

    # ------------------------------------------------------------------ errors

    def error_vector(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Normalised 70-d error [dp, rotvec, dv, dw, dq, dqd] / sigma."""
        p1, q1, qj1, v1, w1, qd1 = self.split(s_pred)
        p2, q2, qj2, v2, w2, qd2 = self.split(s_true)
        q1 = quat_normalize(q1)
        q2 = quat_normalize(q2)
        rot_err = rotvec_from_quat(quat_mul(quat_apply_inverse_quat(q2), q1))
        err = torch.cat([p1 - p2, rot_err, v1 - v2, w1 - w2, qj1 - qj2, qd1 - qd2], dim=-1)
        return err / self.err_scale

    def apply_error(self, s: torch.Tensor, err: torch.Tensor, limits: torch.Tensor | None = None) -> torch.Tensor:
        """Inverse of ``error_vector``; ``limits`` (29, 2) clamps the joint angles."""
        return g1_apply_error(s, err, self.err_scale, limits)

    def error_groups(self) -> dict[str, list[int]]:
        return {"pos": ERR_POS, "vel": ERR_VEL}


def g1_apply_error(s: torch.Tensor, err: torch.Tensor, err_scale: torch.Tensor, joint_limits: torch.Tensor | None = None) -> torch.Tensor:
    """State s' with ``error_vector(s', s) == err`` (the inverse of ``StateSpec.error_vector``).

    s (N, 71) = [root_pos, root_quat wxyz, joint_pos, root_lin_vel_w, root_ang_vel_b, joint_vel];
    err (N, 70) normalised; err_scale (70,). The rotation is applied in the body frame
    (q' = q ⊗ exp(r)), matching ``error_vector``'s log(q^-1 ⊗ q'). Joint angles are clamped to
    ``joint_limits`` (29, 2) if given, so the round trip is exact only inside the limits.
    """
    d = err * err_scale
    p, q, qj, v, w, qd = StateSpec.split(s)
    q_new = quat_normalize(quat_mul(quat_normalize(q), quat_from_rotvec(d[..., 3:6])))
    qj_new = qj + d[..., 12:41]
    if joint_limits is not None:
        qj_new = torch.minimum(torch.maximum(qj_new, joint_limits[:, 0]), joint_limits[:, 1])
    return torch.cat([p + d[..., 0:3], q_new, qj_new, v + d[..., 6:9], w + d[..., 9:12], qd + d[..., 41:70]], dim=-1)


def quat_apply_inverse_quat(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a unit quaternion (inverse)."""
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)
