"""Margin tools through the state-spec interface: for the G1, passing the spec gives exactly what the
previous ``err_scale`` call style gave; for the planar vehicle (``toy_vehicle.py``), branching runs,
model-error margins and the model's worst-case direction work with its own state and error sizes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from test_margins import SPEC, BiasedModel, StandInEnv, StandInIO
from toy_vehicle import PlanarVehicleHeads, PlanarVehicleSpecCfg, VehicleTrackingTask

from supermimic.data.replay_buffer import StartBatch
from supermimic.eval.adversarial import predicted_margin, worst_case_direction
from supermimic.eval.margins import run_branch_pass
from supermimic.eval.perturb import (
    component_mask,
    margins_from_outcomes,
    perturb_state,
    random_directions,
)
from supermimic.imagination.rollout import imagine
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

THRESHOLD = 0.75


def _cmd(n):
    return SimpleNamespace(forced_frame=torch.zeros(n, dtype=torch.long), library=SimpleNamespace(clip_start=torch.tensor([0])))


# ---------------------------------------------------------------------- G1: spec == err_scale


def test_g1_perturbation_and_directions_are_unchanged_by_passing_the_spec():
    g = torch.Generator().manual_seed(0)
    s = torch.zeros(4, 71)
    s[:, 3], s[:, 2] = 1.0, 0.75
    s[:, 7:] = 0.1 * torch.randn(4, 64, generator=g)
    d = torch.randn(4, 70, generator=g)
    lim = torch.stack([-0.2 * torch.ones(29), 0.2 * torch.ones(29)], -1)
    assert torch.equal(perturb_state(s, d, SPEC, lim), perturb_state(s, d, SPEC.err_scale, lim))
    w = torch.randn(71, generator=g)
    mask = component_mask("all", spec=SPEC)
    a = worst_case_direction(lambda st: torch.tanh(st) @ w, s, mask, SPEC)
    b = worst_case_direction(lambda st: torch.tanh(st) @ w, s, mask, SPEC.err_scale)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    fails = lambda st: SPEC.error_vector(st, s.repeat_interleave(3, 0)).norm(dim=-1) > 1.5
    assert torch.equal(predicted_margin(fails, s, a[0], [0.5, 1.0, 2.0], SPEC, lim), predicted_margin(fails, s, a[0], [0.5, 1.0, 2.0], SPEC.err_scale, lim))


def test_g1_branch_pass_is_unchanged_by_passing_the_spec():
    times, scales = [8, 33], [0.0, 0.5, 1.0, 2.0]
    mask = component_mask("vel")
    dirs = random_directions(3, 70, mask, seed=0)
    specs = [(ti, di, si) for ti in range(2) for di in range(3) for si in range(4)]
    out = []
    for spec_arg in (SPEC, SPEC.err_scale):
        env = StandInEnv(len(specs) + 1, jitter=1e-5)
        res = run_branch_pass(env, _cmd(env.num_envs), StandInIO(env), lambda o, g: torch.zeros(o.shape[0], 29), BiasedModel(0.01), torch.zeros(33), 0,
                              times, specs, "state", scales, dirs, mask, spec_arg, None, 20, 1, 8)
        out.append(res)
    assert out[0] == out[1]


# ---------------------------------------------------------------------- vehicle


VEH = PlanarVehicleSpecCfg().build("cpu")


class VehicleStandInEnv:
    """A vehicle state that never moves on its own; a branch fails once its normalised deviation from the
    start state exceeds THRESHOLD (so the margin along a unit direction is the first scale above it)."""

    def __init__(self, n):
        self.num_envs, self.device = n, "cpu"
        self.s0 = torch.tensor([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
        self.state = self.s0.expand(n, 6).clone()

    def _obs(self):
        return {"policy": torch.zeros(self.num_envs, 15), "goal": torch.zeros(self.num_envs, 12)}

    def reset(self):
        self.state = self.s0.expand(self.num_envs, 6).clone()
        return self._obs(), {}

    def step(self, act):
        term = VEH.error_vector(self.state, self.s0.expand_as(self.state)).norm(dim=-1) > THRESHOLD
        self.state[term] = self.s0
        return self._obs(), torch.zeros(self.num_envs), term, torch.zeros_like(term), {}


class VehicleBiasedModel:
    """Adds a constant normalised yaw-rate error each step: its k-step error norm is k * bias."""

    def __init__(self, bias):
        self.spec = VEH
        self.bias = torch.zeros(6)
        self.bias[5] = bias

    def step(self, s, a, xi):
        return VEH.apply_error(s, self.bias.expand_as(s))


def _vehicle_run(kind, scales, bias=0.0):
    times, n_dirs = [8, 20], (1 if kind == "wm" else 3)
    specs = [(ti, di, si) for ti in range(len(times)) for di in range(n_dirs) for si in range(len(scales))]
    env = VehicleStandInEnv(len(specs) + 1)
    mask = component_mask("vel", spec=VEH)
    dirs = random_directions(n_dirs, VEH.err_dim, mask, seed=1) if kind == "state" else (random_directions(n_dirs, VEH.action_dim, None, 1) if kind == "action" else None)
    res, err_norm, _drift, nominal_fail = run_branch_pass(env, _cmd(env.num_envs), StandInIO(env), lambda o, g: torch.zeros(o.shape[0], 2), VehicleBiasedModel(bias), torch.zeros(1), 0,
                                                         times, specs, kind, scales, dirs, mask, VEH, None, 10, 1, 8)
    failed = np.zeros((len(times), n_dirs, len(scales)), dtype=bool)
    for (ti, di, si), (v, f, _) in zip(specs, res):
        assert v
        failed[ti, di, si] = f
    return failed, err_norm, nominal_fail


def test_vehicle_state_margins():
    scales = [0.0, 0.5, 1.0, 2.0]
    failed, err_norm, nominal_fail = _vehicle_run("state", scales)
    assert nominal_fail == -1 and not failed[..., 0].any()
    assert (margins_from_outcomes(failed, scales)["margin"] == 1.0).all()
    assert set(err_norm) == {8, 20} and all(v == 0.0 for v in err_norm.values())


def test_vehicle_model_error_margins():
    scales = [0.0, 0.5, 1.0, 2.0, 4.0]
    failed, err_norm, _ = _vehicle_run("wm", scales, bias=0.0625)  # 8-step error norm 0.5
    assert all(abs(v - 0.5) < 1e-5 for v in err_norm.values())
    assert (margins_from_outcomes(failed, scales)["margin"] == 2.0).all()


def test_vehicle_action_branches_leave_the_state_alone():
    failed, _, _ = _vehicle_run("action", [0.0, 1.0, 8.0])
    assert not failed.any()


def test_vehicle_worst_case_direction_from_an_imagined_return():
    torch.manual_seed(0)
    heads, task = PlanarVehicleHeads(), VehicleTrackingTask()
    wm = WorldModel(WorldModelCfg(hidden=(32, 32), state_spec=PlanarVehicleSpecCfg()), heads, "cpu")
    with torch.no_grad():
        wm.dynamics.mlp.net[-1].weight.normal_(0.0, 0.1)
    wm.requires_grad_(False)
    lin = torch.nn.Linear(3 * heads.obs_dim + task.goal_dim, 2)
    lin.requires_grad_(False)
    s0 = torch.cat([task.pos[10], task.yaw[10:11], torch.tensor([5.0, 0.0, 0.0])])[None]

    def rollout(st):
        n = st.shape[0]
        start = StartBatch(state=st, xi=torch.full((n, 1), 0.8), frame=torch.full((n,), 10), prev_action=torch.zeros(n, 2),
                           hist_states=st[None].expand(3, -1, -1), hist_actions=torch.zeros(3, n, 2))
        return imagine(wm, lambda o, g: lin(torch.cat([o, g], -1)), task, start, 12, obs_noise=False)

    def objective(st):
        ro = rollout(st)
        return (ro.survival[:-1] * ro.rewards).sum(0)

    mask = component_mask("pos", spec=VEH)
    d, gn = worst_case_direction(objective, s0, mask, VEH)
    assert gn.item() > 0 and abs(d.norm().item() - 1.0) < 1e-5 and (d * (1 - mask)).abs().max() == 0
    eps = 1e-2  # first order: a small step along the direction lowers the imagined return
    assert objective(perturb_state(s0, eps * d, VEH)).item() < objective(s0).item()
    pm = predicted_margin(lambda st: rollout(st).survival[-1] < 0.5, s0, d, [0.5, 2.0, 8.0, 32.0], VEH)
    assert pm.shape == (1,)
