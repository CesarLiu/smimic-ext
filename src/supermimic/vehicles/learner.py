"""The vehicle behind the learner's three interfaces (state spec, analytic heads, task), so the SuperMimic
world model, imagination and policy / critic training run on ``VehicleTrackingEnv`` data unchanged.

* ``VehicleStateSpec``: the environment's logged state (simulator state, actuator state, pending commands,
  ABS / ESC flags). Network input drops the pose (the dynamics are invariant to position and heading);
  the network predicts increments, the pose is integrated explicitly from the predicted body velocities.
* ``VehicleHeads``: observation, reward, termination and critic features from ``tracking.py`` and the
  environment's observation, so imagination scores a state exactly as the environment does. The
  auxiliary output is the pose, which the goal preview needs.
* ``VehicleTask``: references and the goal preview (``env.preview_goal``) from a ``ReferenceLibrary``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.tasks.base import TaskBase
from supermimic.vehicles import tracking
from supermimic.vehicles.dynamics import (
    AX,
    AY,
    DELTA,
    PITCH,
    PITCH_RATE,
    ROLL,
    ROLL_RATE,
    VX,
    VY,
    YAW,
    R,
    W,
    X,
    Y,
)
from supermimic.vehicles.env import (
    ACTION_DIM,
    XI_DIM,
    ReferenceLibrary,
    VehicleEnvCfg,
    actuator_update,
    preview_goal,
)
from supermimic.world_model.heads_base import AnalyticHeadsBase, HeadsStep
from supermimic.world_model.spec_base import StateSpecBase, StateSpecCfgBase

# Logged state layout (VehicleEnvCfg.state_dim with max_delay = 3): sim state 0..16, a_act 17,
# pending commands 18..23, ABS flags 24..27, ESC flag 28.
A_ACT = 17


@dataclass
class VehicleStateSpecCfg(StateSpecCfgBase):
    dt: float = 0.02
    max_delay: int = 3
    explicit_integration: bool = True
    # actuators (``env.actuator_update``), exact in the world model; defaults = VehicleEnvCfg / VehicleParams
    steer_scale: float = 0.2
    steer_tau: float = 0.05
    accel_scale: float = 6.0
    accel_tau: float = 0.15
    action_clip: float = 3.0
    steer_rate_max: float = 0.4
    steer_max: float = 1.066

    @classmethod
    def from_env(cls, env_cfg: VehicleEnvCfg, params) -> VehicleStateSpecCfg:
        return cls(dt=env_cfg.dt, max_delay=env_cfg.max_delay, steer_scale=env_cfg.steer_scale, steer_tau=env_cfg.steer_tau,
                   accel_scale=env_cfg.accel_scale, accel_tau=env_cfg.accel_tau, action_clip=env_cfg.action_clip, steer_rate_max=params.steer_rate_max, steer_max=params.steer_max)

    def build(self, device, dtype=torch.float32) -> VehicleStateSpec:
        return VehicleStateSpec(self, device, dtype)


class VehicleStateSpec(StateSpecBase):
    def __init__(self, cfg: VehicleStateSpecCfg, device, dtype=torch.float32):
        self.cfg, self.dt = cfg, cfg.dt
        self.state_dim = 17 + 1 + cfg.max_delay * ACTION_DIM + 5
        self.feat_dim = self.state_dim - 3
        self.delta_dim = self.state_dim
        self.err_dim = self.state_dim
        self.action_dim = ACTION_DIM
        self.xi_dim = XI_DIM
        n_cmd = cfg.max_delay * ACTION_DIM
        # nominal scales: error (state units per normalised unit), network output, input features
        per = {"x": 0.05, "y": 0.05, "yaw": 0.01, "vx": 0.1, "vy": 0.05, "r": 0.02, "delta": 0.005, "roll": 0.005, "roll_rate": 0.05,
               "pitch": 0.002, "pitch_rate": 0.02, "w": 2.0, "a": 0.3, "a_act": 0.05, "cmd": 0.05, "flag": 0.5}
        err = [per["x"], per["y"], per["yaw"], per["vx"], per["vy"], per["r"], per["delta"], per["roll"], per["roll_rate"], per["pitch"], per["pitch_rate"],
               *[per["w"]] * 4, per["a"], per["a"], per["a_act"], *[per["cmd"]] * n_cmd, *[per["flag"]] * 5]
        self.err_scale = torch.tensor(err, device=device, dtype=dtype)
        self.out_scale = self.err_scale.clone()
        inp = [10.0, 1.0, 0.5, 0.1, 0.05, 0.5, 0.02, 0.2, *[30.0] * 4, 5.0, 5.0, 3.0, *[1.0] * n_cmd, *[1.0] * 5]
        self.in_scale = torch.tensor(inp, device=device, dtype=dtype)
        self.xi_offset = torch.tensor([0.85, 1.0, 1.0, 0.0, 1.5], device=device, dtype=dtype)
        self.xi_scale = torch.tensor([0.25, 0.15, 0.125, 0.1, 1.5], device=device, dtype=dtype)

    def canonical(self, s):
        return s[..., 3:] / self.in_scale

    def normalize_xi(self, xi):
        return (xi - self.xi_offset) / self.xi_scale

    def integrate(self, s, delta):
        d = delta * self.out_scale
        out = s + d
        if self.cfg.explicit_integration:  # pose from the new body velocities, corrections on top
            yaw = s[..., YAW]
            vx, vy = out[..., VX], out[..., VY]
            c, sn = torch.cos(yaw), torch.sin(yaw)
            out = out.clone()
            out[..., X] = s[..., X] + self.dt * (vx * c - vy * sn) + d[..., X]
            out[..., Y] = s[..., Y] + self.dt * (vx * sn + vy * c) + d[..., Y]
            out[..., YAW] = s[..., YAW] + self.dt * out[..., R] + d[..., YAW]
            out[..., ROLL] = s[..., ROLL] + self.dt * out[..., ROLL_RATE] + d[..., ROLL]
            out[..., PITCH] = s[..., PITCH] + self.dt * out[..., PITCH_RATE] + d[..., PITCH]
        return out

    def step_known(self, s, s_next, action, xi):
        """Steering angle, longitudinal response and pending actions: exact (``env.actuator_update``)."""
        c = self.cfg
        env_like = VehicleEnvCfg(dt=c.dt, max_delay=c.max_delay, steer_scale=c.steer_scale, steer_tau=c.steer_tau, accel_scale=c.accel_scale, accel_tau=c.accel_tau, action_clip=c.action_clip)
        pending = s[..., A_ACT + 1:A_ACT + 1 + c.max_delay * ACTION_DIM].reshape(*s.shape[:-1], c.max_delay, ACTION_DIM)
        _, a_new, pend_new, delta_next = actuator_update(env_like, action, pending, xi[..., 4].long(), s[..., DELTA], s[..., A_ACT], c.steer_rate_max, c.steer_max)
        out = s_next.clone()
        out[..., DELTA] = delta_next
        out[..., A_ACT] = a_new
        out[..., A_ACT + 1:A_ACT + 1 + c.max_delay * ACTION_DIM] = pend_new.reshape(*s.shape[:-1], -1)
        return out

    def error_vector(self, s_pred, s_true):
        e = s_pred - s_true
        e = torch.cat([e[..., :YAW], tracking.wrap(e[..., YAW:YAW + 1]), e[..., YAW + 1:]], -1)
        return e / self.err_scale

    def apply_error(self, s, err, limits=None):
        return s + err * self.err_scale

    def error_groups(self):
        vel = [VX, VY, R, ROLL_RATE, PITCH_RATE, *range(W.start, W.stop), AX, AY]
        return {"pos": [i for i in range(self.err_dim) if i not in vel], "vel": vel}


class VehicleHeads(AnalyticHeadsBase):
    obs_dim = 12
    reward_term_names = tracking.TERM_NAMES
    term_feat_dim = 5
    critic_feat_dim = 5

    def __init__(self, env_cfg: VehicleEnvCfg, wheel_radius: float):
        self.cfg = env_cfg.tracking
        self.R_w = wheel_radius
        self._noise = torch.tensor([0.1, 0.01, 0.1, 0.1, 0.002, *[0.05] * 4, 0.05, 0.0, 0.0])

    @property
    def max_step_reward(self) -> float:
        return self.cfg.max_step_reward

    def observation(self, s, prev_action, xi):
        """``VehicleTrackingEnv.observation``, from the logged state."""
        return torch.cat([s[..., VX:VX + 1], s[..., R:R + 1], s[..., AX:AY + 1], s[..., DELTA:DELTA + 1], s[..., W] * self.R_w, s[..., A_ACT:A_ACT + 1], prev_action], -1)

    def obs_noise_half(self):
        return self._noise

    def flatten_obs_history(self, hist):
        H, N, D = hist.shape
        return hist.permute(1, 0, 2).reshape(N, H * D)

    def evaluate(self, s_next, action, prev_action, xi, ref) -> HeadsStep:
        terms = tracking.reward_terms(s_next, action, prev_action, ref, self.cfg)
        return HeadsStep(terms, self.observation(s_next, action, xi), self.aux(s_next))

    def aux(self, s):
        return s[..., X:YAW + 1]

    def state_termination_features(self, s, ref):
        return tracking.termination_features(s, ref)

    def termination_feature_scale(self, wm_cfg):
        c = self.cfg
        return (1 / c.lat_max, 1 / c.lon_max, 1 / c.yaw_max, 1 / c.beta_max, 1 / c.min_speed)

    def critic_features(self, s, ref):
        """Tracking errors over their termination thresholds, clipped (bounded even off the data)."""
        e, c = tracking.errors(s, ref), self.cfg
        f = torch.stack([e["lat"] / c.lat_max, e["lon"] / c.lon_max, e["yaw"] / c.yaw_max, e["v"] / 5.0, e["beta"] / c.beta_max], -1)
        return f.clamp(-5.0, 5.0)


class VehicleTask(TaskBase):
    needs_aux = True

    def __init__(self, lib: ReferenceLibrary, preview):
        self.lib, self.preview = lib, tuple(preview)
        self.goal_dim = 5 * len(self.preview)
        self._scale = torch.tensor([0.1, 0.5, 2.0, 0.1, 5.0]).repeat(len(self.preview))

    def targets(self, idx, horizon):
        steps = torch.arange(horizon + 1, device=idx.device)[:, None]
        return torch.minimum(idx[None, :] + steps, self.lib.layout.last_of(idx)[None, :])

    def reference(self, idx):
        return self.lib.ref(idx)

    def goal_scale(self):
        return self._scale

    def goal(self, idx, ref, aux, noise):
        return preview_goal(self.lib, idx, aux, self.preview)
