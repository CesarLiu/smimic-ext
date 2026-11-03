"""A minimal planar vehicle (state spec + analytic heads) used to check that the world model and the
imagined step run on a robot other than the G1 through the two interfaces alone.

State s = [x, y, psi, vx_body, vy_body, yaw_rate]; action a = [steer, accel]; sim context xi = [mu].
The task is tracking a reference pose and speed (``VehicleRef``). Not a vehicle model: the point is
the shapes and the gradient paths, not the physics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.world_model.heads_base import AnalyticHeadsBase, HeadsStep
from supermimic.world_model.spec_base import StateSpecBase, StateSpecCfgBase


@dataclass
class PlanarVehicleSpecCfg(StateSpecCfgBase):
    dt: float = 0.02
    out_scale: float = 0.5
    err_scale: tuple[float, ...] = (0.1, 0.1, 0.05, 0.5, 0.5, 0.2)

    def build(self, device, dtype=torch.float32) -> PlanarVehicleSpec:
        return PlanarVehicleSpec(self, device, dtype)


class PlanarVehicleSpec(StateSpecBase):
    state_dim, feat_dim, delta_dim, err_dim, action_dim, xi_dim = 6, 3, 3, 6, 2, 1

    def __init__(self, cfg: PlanarVehicleSpecCfg, device, dtype=torch.float32):
        self.cfg, self.dt = cfg, cfg.dt
        self.err_scale = torch.tensor(cfg.err_scale, device=device, dtype=dtype)

    def canonical(self, s):
        return s[..., 3:6]  # pose-invariant: body-frame velocities only

    def normalize_xi(self, xi):
        return (xi - 0.75) / 0.45

    def integrate(self, s, delta):
        v = s[..., 3:6] + self.cfg.out_scale * delta
        psi = s[..., 2] + self.dt * v[..., 2]
        c, sn = torch.cos(psi), torch.sin(psi)
        x = s[..., 0] + self.dt * (c * v[..., 0] - sn * v[..., 1])
        y = s[..., 1] + self.dt * (sn * v[..., 0] + c * v[..., 1])
        return torch.cat([torch.stack([x, y, psi], -1), v], -1)

    def error_vector(self, s_pred, s_true):
        d = s_pred - s_true
        dpsi = torch.atan2(torch.sin(d[..., 2]), torch.cos(d[..., 2]))
        return torch.cat([d[..., 0:2], dpsi[..., None], d[..., 3:6]], -1) / self.err_scale

    def apply_error(self, s, err, limits=None):
        return s + err * self.err_scale

    def error_groups(self):
        return {"pos": [0, 1, 2], "vel": [3, 4, 5]}


@dataclass
class VehicleRef:
    pos: torch.Tensor  # (n, 2)
    yaw: torch.Tensor  # (n,)
    speed: torch.Tensor  # (n,)


def _wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


class PlanarVehicleHeads(AnalyticHeadsBase):
    obs_dim = 5  # vx, vy, yaw rate, previous action (2)
    reward_term_names = ("lateral", "heading", "speed")
    term_feat_dim = 2  # lateral error, heading error
    critic_feat_dim = 2

    def __init__(self, dt: float = 0.02, lat_thr: float = 2.0, yaw_thr: float = 0.5):
        self.dt, self.lat_thr, self.yaw_thr = dt, lat_thr, yaw_thr

    @property
    def max_step_reward(self) -> float:
        return 3.0 * self.dt

    def observation(self, s, prev_action, xi):
        return torch.cat([s[..., 3:6], prev_action], -1)

    def obs_noise_half(self):
        return torch.tensor([0.05, 0.05, 0.02, 0.0, 0.0])

    def flatten_obs_history(self, hist):
        H, N, D = hist.shape
        return hist.permute(1, 0, 2).reshape(N, H * D)

    def _errors(self, s, ref: VehicleRef):
        d = s[..., 0:2] - ref.pos
        c, sn = torch.cos(ref.yaw), torch.sin(ref.yaw)
        e_lat = -sn * d[..., 0] + c * d[..., 1]
        e_yaw = _wrap(s[..., 2] - ref.yaw)
        return e_lat, e_yaw

    def evaluate(self, s_next, action, prev_action, xi, ref: VehicleRef) -> HeadsStep:
        e_lat, e_yaw = self._errors(s_next, ref)
        speed = s_next[..., 3:5].norm(dim=-1)
        terms = {
            "lateral": torch.exp(-(e_lat**2) / 0.5**2) * self.dt,
            "heading": torch.exp(-(e_yaw**2) / 0.1**2) * self.dt,
            "speed": torch.exp(-((speed - ref.speed) ** 2)) * self.dt,
        }
        terms["total"] = terms["lateral"] + terms["heading"] + terms["speed"]
        return HeadsStep(terms, self.observation(s_next, action, xi), None)

    def state_termination_features(self, s, ref):
        return torch.stack(self._errors(s, ref), -1)

    def termination_feature_scale(self, wm_cfg):
        return (1.0 / self.lat_thr, 1.0 / self.yaw_thr)

    def critic_features(self, s, ref):
        return torch.stack(self._errors(s, ref), -1)


def vehicle_states(n, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 6, generator=g, dtype=dtype)


def vehicle_ref(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return VehicleRef(torch.randn(n, 2, generator=g), torch.randn(n, generator=g), 5.0 + torch.rand(n, generator=g))
