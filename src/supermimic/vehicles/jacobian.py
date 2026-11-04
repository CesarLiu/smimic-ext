"""E1: does the vehicle world model's Jacobian agree with the simulator's, up to and beyond the grip limit?

Policy gradients in SuperMimic flow through the world model's Jacobians. Near tyre saturation the true
sensitivities change sharply (more steering stops producing more lateral acceleration), so this is where
a learned model is most likely to mislead the policy (VEHICLE_PROPOSAL.md §8, first risk).

What is compared, per evaluated state, in the world model's normalised error units:

* one step: the Jacobian of the state *change* ``s' - s`` w.r.t. the dynamic state components and the
  command acting in this step (the action itself without delay; with a delay of d steps, the pending
  action from d steps ago, which is part of the state), restricted to the components the network predicts (``LEARNED``; pose integration, actuator
  states and pending commands are exact and would inflate agreement);
* k steps (``k = 8``): the gradient of the predicted state after k steps w.r.t. the first action, along the
  recorded actions (what BPTT through an imagined rollout uses);
* key sensitivities: lateral acceleration, yaw rate and lateral velocity w.r.t. the steering action,
  longitudinal speed w.r.t. the acceleration action (sign agreement and relative error).

Simulator side: central finite differences in float64 through the same actuator + dynamics pipeline as the
environment (``logged_step``); each state is differenced with two step sizes and states where they
disagree (non-smooth points: wheel-speed clamps, steering limits, ABS / ESC switching) are excluded and
counted. World-model side: autograd.

States are grouped by friction utilisation ``u = sqrt(ax^2 + ay^2) / (mu g)``.
"""

from __future__ import annotations

import math

import torch

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
    Context,
    G,
    R,
    VehicleDynamics,
    W,
)
from supermimic.vehicles.env import ACTION_DIM, VehicleEnvCfg, actuator_update

A_ACT = 17
WHEELS = list(range(W.start, W.stop))
LEARNED = [VX, VY, R, ROLL, ROLL_RATE, PITCH, PITCH_RATE, *WHEELS, AX, AY]  # rows the network predicts
STATE_COLS = [VX, VY, R, DELTA, ROLL, ROLL_RATE, PITCH, PITCH_RATE, *WHEELS, AX, AY, A_ACT]  # inputs that matter
KEY = {"ay/steer": (AY, "steer"), "r/steer": (R, "steer"), "vy/steer": (VY, "steer"), "vx/accel": (VX, "accel")}
KEY_MIN = 1e-3  # a key sensitivity is scored only where the simulator's is at least this (normalised units)
GROUPS = {"velocity": [VX, VY, R], "attitude": [ROLL, ROLL_RATE, PITCH, PITCH_RATE], "wheels": WHEELS, "acceleration": [AX, AY]}
BINS = (0.0, 0.3, 0.6, 0.8, 0.9, 1.0, math.inf)


def logged_step(dyn: VehicleDynamics, cfg: VehicleEnvCfg, s: torch.Tensor, a: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """One environment control step on the logged state (``VehicleTrackingEnv.state`` layout)."""
    d = cfg.max_delay
    n = s.shape[0]
    sim, a_act = s[:, :A_ACT], s[:, A_ACT]
    pending = s[:, A_ACT + 1:A_ACT + 1 + d * ACTION_DIM].reshape(n, d, ACTION_DIM)
    flags = s[:, A_ACT + 1 + d * ACTION_DIM:]
    ctx = Context(xi[:, 0], xi[:, 1], xi[:, 2], xi[:, 3])
    p = dyn.p
    steer_rate, a_new, pend_new, _ = actuator_update(cfg, a, pending, xi[:, 4].long(), sim[:, DELTA], a_act, p.steer_rate_max, p.steer_max)
    sim_new, flags_new = dyn.step(sim, flags, steer_rate, a_new, ctx)
    return torch.cat([sim_new, a_new[:, None], pend_new.reshape(n, -1), flags_new], -1)


def utilisation(s: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(s[:, AX] ** 2 + s[:, AY] ** 2) / (xi[:, 0] * G)


def _with_delta(J: torch.Tensor) -> torch.Tensor:
    """J (n, D, cols) of s' -> J of s' - s: subtract the identity on the state columns."""
    J = J.clone()
    for k, c in enumerate(STATE_COLS):
        J[:, c, k] -= 1.0
    return J


def acting_command_slot(xi: torch.Tensor) -> torch.Tensor:
    """(n,) state index of the first component of the command acting now, or -1 where it is the action."""
    d = xi[:, 4].long()
    return torch.where(d > 0, A_ACT + 1 + ACTION_DIM * (d - 1), torch.full_like(d, -1))


@torch.no_grad()
def sim_jacobian(step, s, a, xi, err_scale, h: float = 1e-2) -> torch.Tensor:
    """(n, D, len(STATE_COLS) + 2) Jacobian of ``s' - s`` by central differences, in normalised units: state
    inputs perturbed by ``h`` normalised units, the acting command by ``h`` raw units, outputs over
    ``err_scale``."""
    n, D = s.shape
    m = len(STATE_COLS) + ACTION_DIM
    S = s[None].repeat(2 * m, 1, 1)
    A = a[None].repeat(2 * m, 1, 1)
    for k, c in enumerate(STATE_COLS):
        S[2 * k, :, c] += h * err_scale[c]
        S[2 * k + 1, :, c] -= h * err_scale[c]
    slot = acting_command_slot(xi)
    now, rows = slot < 0, torch.arange(n)
    for j in range(ACTION_DIM):
        k = len(STATE_COLS) + j
        for sign, layer in ((1.0, 2 * k), (-1.0, 2 * k + 1)):
            A[layer, now, j] += sign * h
            S[layer, rows[~now], slot[~now] + j] += sign * h
    out = step(S.reshape(-1, D), A.reshape(-1, ACTION_DIM), xi.repeat(2 * m, 1)).reshape(2 * m, n, D)
    J = (out[0::2] - out[1::2]).permute(1, 2, 0) / (2 * h) / err_scale[None, :, None]
    return _with_delta(J)


def wm_jacobian(wm, s, a, xi, err_scale) -> torch.Tensor:
    """The world model's Jacobian of ``s' - s``, in the same units as ``sim_jacobian``."""
    s = s.detach().clone().requires_grad_(True)
    a = a.detach().clone().requires_grad_(True)
    out = wm.step(s, a, xi) / err_scale
    n, D = s.shape
    slot = acting_command_slot(xi)
    now, rows = slot < 0, torch.arange(n)
    J = torch.zeros(n, D, len(STATE_COLS) + ACTION_DIM, dtype=s.dtype)
    for i in LEARNED:
        gs, ga = torch.autograd.grad(out[:, i].sum(), (s, a), retain_graph=True)
        J[:, i, : len(STATE_COLS)] = gs[:, STATE_COLS] * err_scale[STATE_COLS]
        for j in range(ACTION_DIM):
            J[now, i, len(STATE_COLS) + j] = ga[now, j]
            J[rows[~now], i, len(STATE_COLS) + j] = gs[rows[~now], slot[~now] + j]
    return _with_delta(J.detach())


@torch.no_grad()
def sim_kstep_action_grad(step, s, actions, xi, err_scale, h: float = 1e-2) -> torch.Tensor:
    """(n, D, 2): d s_k / d a_0 along the recorded ``actions`` (k, n, 2), central differences."""
    k, n, _ = actions.shape
    S = s[None].repeat(2 * ACTION_DIM, 1, 1).reshape(-1, s.shape[1])
    X = xi.repeat(2 * ACTION_DIM, 1)
    for t in range(k):
        A = actions[t][None].repeat(2 * ACTION_DIM, 1, 1)
        if t == 0:
            for j in range(ACTION_DIM):
                A[2 * j, :, j] += h
                A[2 * j + 1, :, j] -= h
        S = step(S, A.reshape(-1, ACTION_DIM), X)
    out = S.reshape(2 * ACTION_DIM, n, -1)
    return ((out[0::2] - out[1::2]).permute(1, 2, 0) / (2 * h)) / err_scale[None, :, None]


def wm_kstep_action_grad(wm, s, actions, xi, err_scale) -> torch.Tensor:
    a0 = actions[0].detach().clone().requires_grad_(True)
    x = wm.step(s.detach(), a0, xi)
    for t in range(1, actions.shape[0]):
        x = wm.step(x, actions[t].detach(), xi)
    out = x / err_scale
    J = torch.zeros(s.shape[0], s.shape[1], ACTION_DIM, dtype=s.dtype)
    for i in LEARNED:
        (g,) = torch.autograd.grad(out[:, i].sum(), a0, retain_graph=True)
        J[:, i] = g
    return J.detach()


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)


def compare(J_wm: torch.Tensor, J_sim: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-sample metrics on the learned rows: cosine and magnitude ratio of the whole Jacobian and of its
    action block, cosines per output group, and the key sensitivities (NaN where the simulator's value is
    below ``KEY_MIN``: e.g. while the steering servo runs at its rate limit, this step's lateral response
    does not depend on the steering command at all, and a sign comparison means nothing)."""
    rows = LEARNED
    ncol = len(STATE_COLS)
    a_w, a_s = J_wm[:, rows], J_sim[:, rows]
    out = {
        "cos": _cos(a_w, a_s),
        "mag_ratio": a_w.reshape(len(a_w), -1).norm(dim=-1) / a_s.reshape(len(a_s), -1).norm(dim=-1).clamp_min(1e-12),
        "cos_action": _cos(a_w[:, :, ncol:], a_s[:, :, ncol:]),
    }
    for g, grows in GROUPS.items():
        out[f"cos_{g}"] = _cos(J_wm[:, grows], J_sim[:, grows])
    for name, (row, which) in KEY.items():
        col = ncol + (0 if which == "steer" else 1)
        w, s = J_wm[:, row, col], J_sim[:, row, col]
        live = s.abs() >= KEY_MIN
        nan = torch.full_like(s, float("nan"))
        out[f"{name}_sign_ok"] = torch.where(live, ((w * s) > 0).to(s.dtype), nan)
        out[f"{name}_rel_err"] = torch.where(live, (w - s).abs() / s.abs().clamp_min(1e-12), nan)
        out[f"{name}_sim"] = s
        out[f"{name}_wm"] = w
    return out


SAT_BIN = 4  # first bin of the saturation range, u > 0.9
GATE_KSTEP_COS = 0.8
GATE_SAT_ERROR = 0.2


def e1_gate(err_rows: list[dict], sat_kstep_cos: float) -> tuple[bool, list[str]]:
    """E1 gate as revised 2026-11-04 (VEHICLE_PROPOSAL.md §3.9): near saturation (u > 0.9, bins pooled)
    the median cosine of the k-step action gradient exceeds 0.8 and the mean k-step prediction error is
    below 0.2. The one-step Jacobian is reported as a diagnostic only.

    ``err_rows``: per-bin {"windows", "nse_8step"} in bin order."""
    sat = [r for r in err_rows[SAT_BIN:] if r.get("nse_8step") is not None and r.get("windows", 0) > 0]
    n = sum(r["windows"] for r in sat)
    sat_err = sum(r["nse_8step"] * r["windows"] for r in sat) / n if n else float("nan")
    fails = []
    if not (math.isfinite(sat_kstep_cos) and sat_kstep_cos > GATE_KSTEP_COS):
        fails.append(f"k-step action-gradient cosine near saturation {sat_kstep_cos:.3f} <= {GATE_KSTEP_COS}")
    if not (math.isfinite(sat_err) and sat_err < GATE_SAT_ERROR):
        fails.append(f"k-step error near saturation {sat_err:.3f} >= {GATE_SAT_ERROR}")
    return not fails, fails


def saturation_error(err_rows: list[dict]) -> float:
    sat = [r for r in err_rows[SAT_BIN:] if r.get("nse_8step") is not None and r.get("windows", 0) > 0]
    n = sum(r["windows"] for r in sat)
    return sum(r["nse_8step"] * r["windows"] for r in sat) / n if n else float("nan")


def bin_index(u: torch.Tensor) -> torch.Tensor:
    edges = torch.tensor(BINS[1:-1], dtype=u.dtype)
    return torch.bucketize(u, edges, right=True)


def bin_label(i: int) -> str:
    lo, hi = BINS[i], BINS[i + 1]
    return f"{lo:.1f}-{hi:.1f}" if math.isfinite(hi) else f">{lo:.1f}"
