"""Reference-tracking errors, reward and termination for the vehicle (shared by the environment and,
later, the world model's analytic heads, so both score a state identically).

Everything is relative to the reference (VEHICLE_PROPOSAL.md §2): a drifting reference with a large
sideslip angle is tracked, not terminated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.vehicles.dynamics import VX, VY, YAW, X, Y


@dataclass
class VehicleRef:
    """Reference at a batch of target steps, each (n,)."""

    x: torch.Tensor
    y: torch.Tensor
    yaw: torch.Tensor
    v: torch.Tensor  # speed
    beta: torch.Tensor  # sideslip angle
    r: torch.Tensor  # yaw rate


@dataclass
class TrackingCfg:
    sigma_lat: float = 0.5
    sigma_lon: float = 1.0
    sigma_yaw: float = 0.1
    sigma_v: float = 1.0
    sigma_beta: float = 0.1
    w_lat: float = 1.0
    w_lon: float = 0.5
    w_yaw: float = 1.0
    w_v: float = 1.0
    w_beta: float = 0.5
    w_action_rate: float = -0.1
    dt: float = 0.02
    # terminations (relative to the reference)
    lat_max: float = 2.0
    lon_max: float = 4.0
    yaw_max: float = 0.6
    beta_max: float = 0.35
    min_speed: float = 1.0

    @property
    def max_step_reward(self) -> float:
        return (self.w_lat + self.w_lon + self.w_yaw + self.w_v + self.w_beta) * self.dt


TERM_NAMES = ("lateral", "longitudinal", "heading", "speed", "sideslip", "action_rate")


def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def errors(s: torch.Tensor, ref: VehicleRef) -> dict[str, torch.Tensor]:
    """Errors of the state against the reference, in the reference's heading frame."""
    dx, dy = s[:, X] - ref.x, s[:, Y] - ref.y
    c, sn = torch.cos(ref.yaw), torch.sin(ref.yaw)
    vx, vy = s[:, VX], s[:, VY]
    v = torch.sqrt(vx**2 + vy**2 + 1e-12)
    return {
        "lon": c * dx + sn * dy,
        "lat": -sn * dx + c * dy,
        "yaw": wrap(s[:, YAW] - ref.yaw),
        "v": v - ref.v,
        "beta": wrap(torch.atan2(vy, vx.clamp_min(0.1)) - ref.beta),
        "speed": v,
    }


def reward_terms(s_next: torch.Tensor, action: torch.Tensor, prev_action: torch.Tensor, ref: VehicleRef, cfg: TrackingCfg) -> dict[str, torch.Tensor]:
    e = errors(s_next, ref)
    k = lambda x, sig: torch.exp(-((x / sig) ** 2))
    t = {
        "lateral": cfg.w_lat * k(e["lat"], cfg.sigma_lat),
        "longitudinal": cfg.w_lon * k(e["lon"], cfg.sigma_lon),
        "heading": cfg.w_yaw * k(e["yaw"], cfg.sigma_yaw),
        "speed": cfg.w_v * k(e["v"], cfg.sigma_v),
        "sideslip": cfg.w_beta * k(e["beta"], cfg.sigma_beta),
        "action_rate": cfg.w_action_rate * ((action - prev_action) ** 2).sum(-1),
    }
    t = {name: v * cfg.dt for name, v in t.items()}
    t["total"] = sum(t[n] for n in TERM_NAMES)
    return t


def termination_features(s: torch.Tensor, ref: VehicleRef) -> torch.Tensor:
    """(n, 5): lateral, longitudinal, heading and sideslip errors, and the speed (for the stall test)."""
    e = errors(s, ref)
    return torch.stack([e["lat"], e["lon"], e["yaw"], e["beta"], e["speed"]], -1)


def terminated(s: torch.Tensor, ref: VehicleRef, cfg: TrackingCfg) -> torch.Tensor:
    f = termination_features(s, ref)
    return (f[:, 0].abs() > cfg.lat_max) | (f[:, 1].abs() > cfg.lon_max) | (f[:, 2].abs() > cfg.yaw_max) | (f[:, 3].abs() > cfg.beta_max) | (f[:, 4] < cfg.min_speed)
