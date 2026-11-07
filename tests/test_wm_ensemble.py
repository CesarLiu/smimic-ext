"""World-model ensemble and the uncertainty penalty in imagination (VEHICLE_PROPOSAL.md §10.6): a single model is
unchanged, the ensemble steps with the members' mean, its disagreement is zero for identical members and positive
otherwise, every member is trained, and the penalty lowers the imagined reward with a gradient into the action."""

from __future__ import annotations

import copy

import torch

from supermimic.imagination.compiled import ImagStep
from supermimic.vehicles.env import ReferenceLibrary, VehicleEnvCfg, VehicleTrackingEnv
from supermimic.vehicles.learner import VehicleHeads, VehicleStateSpecCfg, VehicleTask
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def _setup(ensemble, chassis=True, n=16, steps=40):
    torch.manual_seed(0)
    cfg = VehicleEnvCfg(num_envs=n, chassis=chassis, replay_capacity=60)
    lib = ReferenceLibrary.generate(VehicleEnvCfg(), num=2, steps=120, seed=0)
    env = VehicleTrackingEnv(cfg, lib, seed=0)
    for _ in range(steps):
        env.step(torch.randn(n, cfg.action_dim))
    heads = VehicleHeads(cfg, env.dyn.p.R_w)
    wm = WorldModel(WorldModelCfg(hidden=(32, 32), state_spec=VehicleStateSpecCfg.from_env(cfg, env.dyn.p), ensemble=ensemble), heads, "cpu")
    return cfg, lib, env, heads, wm


def test_single_model_is_unchanged():
    _c, _l, _e, _h, wm = _setup(1)
    assert wm.ensemble == 1 and not any(k.startswith("ensemble_extra") for k in wm.state_dict())
    s, a, xi = _e.state(), torch.randn(16, 5), _e.xi
    s1, u = wm.step_with_uncertainty(s, a, xi)
    torch.testing.assert_close(s1, wm.step(s, a, xi))
    assert float(u.abs().max()) == 0.0


def _randomise(wm):
    with torch.no_grad():
        for k in range(wm.ensemble):
            last = wm.member(k).mlp.net[-1]
            last.weight.normal_(0, 0.05)
            last.bias.normal_(0, 0.05)


def test_ensemble_mean_and_disagreement():
    _c, _l, env, _h, wm = _setup(3)
    _randomise(wm)
    s, a, xi = env.state(), torch.randn(16, 5), env.xi
    s_mean, u = wm.step_with_uncertainty(s, a, xi)
    torch.testing.assert_close(s_mean, wm.step(s, a, xi))
    assert float(u.min()) > 0.0
    # the mean of the members' increments, through the spec's integration
    feat, xin = wm.spec.canonical(s), wm.spec.normalize_xi(xi)
    d = torch.stack([wm.member(k)(feat, a, xin) for k in range(3)]).mean(0)
    torch.testing.assert_close(s_mean, wm.spec.step_known(s, wm.spec.integrate(s, d), a, xi))
    # identical members do not disagree
    with torch.no_grad():
        for k in (1, 2):
            wm.member(k).load_state_dict(copy.deepcopy(wm.member(0).state_dict()))
    assert float(wm.step_with_uncertainty(s, a, xi)[1].abs().max()) < 1e-10


def test_every_member_is_trained():
    _c, lib, env, heads, wm = _setup(3)
    before = [copy.deepcopy(wm.member(k).state_dict()) for k in range(3)]
    tr = WorldModelTrainer(wm, VehicleTask(lib, env.cfg.preview), WMTrainerCfg(window=4, batches_per_update=2, batch_size=8, compile=None))
    m = tr.update(env.buffer)
    assert all(v == v for v in m.values())
    for k in range(3):
        after = wm.member(k).state_dict()
        assert any(not torch.equal(before[k][n], after[n]) for n in after)


def test_uncertainty_penalty_lowers_the_reward_and_has_a_gradient():
    _c, lib, env, heads, wm = _setup(3)
    _randomise(wm)
    s, xi = env.state(), env.xi
    ref = lib.ref(env.frame)
    a = torch.randn(16, 5, requires_grad=True)
    plain = ImagStep(wm)
    pen = ImagStep(wm, uncertainty_penalty=2.0)
    r0 = plain(s, a, env.prev_action, xi, ref)[1]
    r1 = pen(s, a, env.prev_action, xi, ref)[1]
    u = pen.last_disagreement
    torch.testing.assert_close(r1, r0 - 2.0 * heads.max_step_reward * u)
    g = torch.autograd.grad(u.sum(), a)[0]
    assert float(g.abs().sum()) > 0.0
