"""AnalyticHeads interface: G1 regression of the whole heads → world model → imagination → update chain
against outputs recorded before the refactor, the contract every heads implementation must meet, and
a planar vehicle (``toy_vehicle.py``) that runs the world-model loss and the imagined step through the
two interfaces alone."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from supermimic.imagination.compiled import ImagStep
from supermimic.world_model.heads_base import AnalyticHeadsBase, HeadsStep, ref_from_tensors, ref_tensors
from supermimic.world_model.world_model import WorldModel, WorldModelCfg
from toy_vehicle import PlanarVehicleHeads, PlanarVehicleSpecCfg, vehicle_ref, vehicle_states

GOLDEN = Path(__file__).parent / "data" / "g1_heads_golden.pt"


@pytest.fixture(scope="module")
def g1():
    import g1_fixtures as fx

    return fx


def _assert_tree_close(actual, expected, path=""):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), f"{path}: keys {sorted(set(actual) ^ set(expected))}"
        for k in expected:
            _assert_tree_close(actual[k], expected[k], f"{path}/{k}")
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_tree_close(a, e, f"{path}[{i}]")
    else:
        # Exact on the machine that recorded it; the tolerance absorbs BLAS differences elsewhere.
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6, msg=lambda m: f"{path}: {m}")


def test_g1_chain_matches_recorded_outputs(g1):
    golden = torch.load(GOLDEN, weights_only=True)
    _assert_tree_close(g1.run_all(), golden)


# ---------------------------------------------------------------------- contract


def _g1_case(fx):
    heads = fx.build_heads()
    q0 = heads.default_joint_pos
    return heads, fx.states(fx.N, 1, q0), fx.reference(fx.N, 5), fx.start_batch(q0).xi, 29


def _vehicle_case():
    n = 8
    return PlanarVehicleHeads(), vehicle_states(n), vehicle_ref(n), 0.75 + 0.1 * torch.randn(n, 1), 2


@pytest.mark.parametrize("name", ["g1", "planar_vehicle"])
def test_contract(name, g1):
    heads, s, ref, xi, act_dim = _g1_case(g1) if name == "g1" else _vehicle_case()
    assert isinstance(heads, AnalyticHeadsBase)
    n = s.shape[0]
    a = 0.3 * torch.randn(n, act_dim)

    obs = heads.observation(s, a, xi)
    assert obs.shape == (n, heads.obs_dim)
    assert heads.obs_noise_half().shape == (heads.obs_dim,)
    hist = torch.stack([obs] * 3)
    assert heads.flatten_obs_history(hist).shape == (n, 3 * heads.obs_dim)

    out = heads.evaluate(s, a, torch.zeros_like(a), xi, ref)
    assert isinstance(out, HeadsStep)
    assert set(heads.reward_term_names) | {"total"} <= set(out.reward_terms)
    assert out.reward_terms["total"].shape == (n,)
    torch.testing.assert_close(out.obs, heads.observation(s, a, xi))
    assert float(out.reward_terms["total"].max()) <= heads.max_step_reward + 1e-6

    assert heads.state_termination_features(s, ref).shape == (n, heads.term_feat_dim)
    assert len(heads.termination_feature_scale(WorldModelCfg())) == heads.term_feat_dim
    assert heads.critic_features(s, ref).shape == (n, heads.critic_feat_dim)

    # The reference survives the flatten/rebuild round trip compiled code uses.
    back = ref_from_tensors(type(ref), ref_tensors(ref))
    assert all(x is y for x, y in zip(ref_tensors(back), ref_tensors(ref)))


# ---------------------------------------------------------------------- a second robot end to end


def _vehicle_world_model():
    torch.manual_seed(0)
    heads = PlanarVehicleHeads()
    wm = WorldModel(WorldModelCfg(hidden=(32, 32), state_spec=PlanarVehicleSpecCfg()), heads, "cpu")
    with torch.no_grad():
        wm.dynamics.mlp.net[-1].weight.normal_(0.0, 0.1)
    return wm, heads


def test_vehicle_world_model_loss_trains_dynamics_and_termination():
    wm, heads = _vehicle_world_model()
    assert wm.term_head.mlp.net[0].in_features == 2 * heads.term_feat_dim
    k, n = 3, 8
    states = torch.stack([vehicle_states(n, seed=j) for j in range(k + 1)])
    a = torch.randn(k, n, 2)
    xi = torch.rand(n, 1)
    done = torch.zeros(k, n)
    done[-1, :2] = 1.0
    refs = [vehicle_ref(n, seed=10 + j) for j in range(k)]
    loss, metrics = wm.window_loss(states, a, xi, done, refs)
    loss.backward()
    assert torch.isfinite(loss)
    assert wm.dynamics.mlp.net[0].weight.grad.abs().sum() > 0
    assert wm.term_head.mlp.net[0].weight.grad.abs().sum() > 0
    assert {"state_loss", "term_bce", "nse_kstep"} <= set(metrics)


def test_vehicle_imagined_rollout_backpropagates_into_the_policy():
    wm, heads = _vehicle_world_model()
    wm.requires_grad_(False)
    n, hist_len, horizon = 8, 3, 5
    policy = torch.nn.Linear(hist_len * heads.obs_dim + 3, 2)  # obs history + reference speed/yaw/lateral
    step = ImagStep(wm)
    s = vehicle_states(n)
    xi = torch.rand(n, 1)
    a_prev = torch.zeros(n, 2)
    hist = [heads.observation(s, a_prev, xi)] * hist_len
    c = torch.ones(n)
    J = torch.zeros(n)
    for t in range(horizon):
        ref = vehicle_ref(n, seed=t)
        goal = torch.cat([ref.speed[:, None], heads.critic_features(s, ref)], -1)
        a = policy(torch.cat([heads.flatten_obs_history(torch.stack(hist)), goal], -1))
        s, r, term_vec, logits, obs, aux = step(s, a, a_prev, xi, ref)
        assert aux is None and term_vec.shape == (len(heads.reward_term_names), n)
        c = c * (1 - torch.sigmoid(logits))
        J = J + 0.99**t * c * r
        hist = hist[1:] + [obs]
        a_prev = a
    J.mean().backward()
    assert policy.weight.grad is not None and policy.weight.grad.abs().sum() > 0
