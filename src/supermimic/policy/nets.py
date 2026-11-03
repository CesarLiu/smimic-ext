"""Policy (goal encoder + shared decoder) and critic (paper Table A.4).

The networks' input/output sizes and fixed input scales come from a ``NetIO``:

* ``NetIO.g1(pcfg)``: the G1 motion-tracking layout, with the per-term scales of ``PolicyCfg`` (the
  default when no ``io`` is given, so existing configs and checkpoints are unchanged);
* ``NetIO.from_interfaces(spec, heads, task, history)``: any robot, from the state spec (action,
  canonical-state and sim-context sizes), the analytic heads (observation size, history layout and
  ``obs_scale``, critic features) and the task (goal size and ``goal_scale``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from supermimic.world_model.dynamics import MLP
from supermimic.world_model.heads import OBS_DIM, OBS_TERMS

GOAL_FRAME_SIZES = (29, 29, 3, 3, 3, 1)  # per-frame goal features: joint_pos, joint_vel, grav, ang_vel_b, lin_vel_h, height
GOAL_DIM = 680
LATENT_DIM = 32
ACT_DIM = 29


@dataclass
class PolicyCfg:
    encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    latent_dim: int = LATENT_DIM
    history: int = 10
    # G1: fixed per-term observation scales (divide): joint_pos, joint_vel, base_ang_vel, projected_gravity, actions.
    obs_term_scale: tuple[float, ...] = (1.0, 5.0, 2.0, 1.0, 1.0)
    # G1: fixed goal feature scales: joint_pos 29, joint_vel 29, grav 3, ang_vel_b 3, lin_vel_h 3, height 1.
    goal_term_scale: tuple[float, ...] = (1.0, 5.0, 1.0, 2.0, 1.0, 1.0)
    decoder_last_gain: float = 0.01
    goal_dim: int = GOAL_DIM  # G1: 680 (+6 with anchor_obs="ori", +9 with "pos_ori")


@dataclass
class CriticCfg:
    hidden: tuple[int, ...] = (256, 128)


def _obs_scale_vector(cfg: PolicyCfg) -> torch.Tensor:
    parts = []
    for (_name, d), s in zip(OBS_TERMS, cfg.obs_term_scale):
        parts.append(torch.full((cfg.history * d,), 1.0 / s))
    return torch.cat(parts)  # term-major layout matches the flattened history


def _goal_scale_vector(cfg: PolicyCfg, goal_dim: int) -> torch.Tensor:
    return g1_goal_scale(cfg.goal_term_scale, goal_dim)


def g1_goal_scale(goal_term_scale, goal_dim: int) -> torch.Tensor:
    """Per-frame G1 goal scales repeated over the goal frames; trailing extras (anchor term) unscaled."""
    per_frame = torch.cat([torch.full((n,), 1.0 / s) for n, s in zip(GOAL_FRAME_SIZES, goal_term_scale)])
    k, extra = divmod(goal_dim, sum(GOAL_FRAME_SIZES))
    return torch.cat([per_frame.repeat(k), torch.ones(extra)])


@dataclass
class NetIO:
    """Sizes and fixed input scales (multiplied in) of the policy and critic."""

    obs_dim: int  # flattened observation history
    goal_dim: int
    action_dim: int
    feat_dim: int  # canonical state (critic)
    xi_dim: int  # sim context (critic)
    critic_feat_dim: int  # privileged critic features
    obs_scale: torch.Tensor  # (obs_dim,)
    goal_scale: torch.Tensor  # (goal_dim,)

    @classmethod
    def g1(cls, pcfg: PolicyCfg) -> NetIO:
        return cls(
            obs_dim=pcfg.history * OBS_DIM, goal_dim=pcfg.goal_dim, action_dim=ACT_DIM, feat_dim=71, xi_dim=33, critic_feat_dim=4,
            obs_scale=_obs_scale_vector(pcfg), goal_scale=_goal_scale_vector(pcfg, pcfg.goal_dim),
        )

    @classmethod
    def from_interfaces(cls, spec, heads, task, history: int) -> NetIO:
        per_step = heads.obs_scale().reshape(1, 1, heads.obs_dim).expand(history, 1, heads.obs_dim)
        return cls(
            obs_dim=history * heads.obs_dim, goal_dim=task.goal_dim, action_dim=spec.action_dim, feat_dim=spec.feat_dim,
            xi_dim=spec.xi_dim, critic_feat_dim=heads.critic_feat_dim,
            obs_scale=heads.flatten_obs_history(per_step)[0].clone(), goal_scale=task.goal_scale().clone(),
        )


class MotionEncoder(nn.Module):
    """Goal encoder (the motion interface's; the same network serves any task's goal)."""

    def __init__(self, cfg: PolicyCfg, goal_dim: int | None = None, goal_scale: torch.Tensor | None = None):
        super().__init__()
        goal_dim = goal_dim or cfg.goal_dim
        self.mlp = MLP(goal_dim, cfg.encoder_hidden, cfg.latent_dim)
        self.register_buffer("scale", goal_scale.clone() if goal_scale is not None else _goal_scale_vector(cfg, goal_dim))

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(goal * self.scale))


class Decoder(nn.Module):
    def __init__(self, cfg: PolicyCfg, obs_dim: int | None = None, action_dim: int = ACT_DIM, obs_scale: torch.Tensor | None = None):
        super().__init__()
        obs_dim = obs_dim or cfg.history * OBS_DIM
        self.mlp = MLP(cfg.latent_dim + obs_dim, cfg.decoder_hidden, action_dim, last_gain=cfg.decoder_last_gain)
        self.register_buffer("scale", obs_scale.clone() if obs_scale is not None else _obs_scale_vector(cfg))

    def forward(self, z: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([z, obs_hist * self.scale], dim=-1))


class Policy(nn.Module):
    """a = dec(enc_motion(g), o_hist). Extra goal interfaces can be registered later (plan Appendix B).

    ``io`` defaults to the G1 layout (``NetIO.g1(cfg)``)."""

    def __init__(self, cfg: PolicyCfg, io: NetIO | None = None):
        super().__init__()
        io = io or NetIO.g1(cfg)
        self.cfg = cfg
        self.encoders = nn.ModuleDict({"motion": MotionEncoder(cfg, io.goal_dim, io.goal_scale)})
        self.decoder = Decoder(cfg, io.obs_dim, io.action_dim, io.obs_scale)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, interface: str = "motion") -> torch.Tensor:
        return self.decoder(self.encoders[interface](goal), obs_hist)


class Critic(nn.Module):
    """V(canonical state, goal, xi, privileged features) -> scalar (one per interface); G1: 71, 680, 33, 4.

    Sizes come from ``io`` when given, otherwise from the arguments (G1 defaults)."""

    def __init__(self, cfg: CriticCfg, feat_dim: int = 71, goal_dim: int | None = None, xi_dim: int = 33, pcfg: PolicyCfg | None = None, e_root_dim: int = 4, io: NetIO | None = None):
        super().__init__()
        pcfg = pcfg or PolicyCfg()
        if io is not None:
            feat_dim, goal_dim, xi_dim, e_root_dim = io.feat_dim, io.goal_dim, io.xi_dim, io.critic_feat_dim
            goal_scale = io.goal_scale.clone()
        else:
            goal_dim = goal_dim or pcfg.goal_dim
            goal_scale = _goal_scale_vector(pcfg, goal_dim)
        self.mlp = MLP(feat_dim + goal_dim + xi_dim + e_root_dim, cfg.hidden, 1)
        self.register_buffer("goal_scale", goal_scale)

    def forward(self, feat: torch.Tensor, goal: torch.Tensor, xi_norm: torch.Tensor, e_root: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([feat, goal * self.goal_scale, xi_norm, e_root], dim=-1)).squeeze(-1)
