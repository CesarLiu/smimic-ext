"""StateSpec interface: G1 regression against recorded outputs, the contract every spec must meet,
and a second (planar vehicle, ``toy_vehicle.py``) spec that drives the world model through the
interface alone."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from toy_vehicle import PlanarVehicleSpecCfg, vehicle_states

from supermimic.eval.perturb import apply_state_delta, component_mask
from supermimic.world_model.spec_base import StateSpecBase, make_state_spec
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

GOLDEN = Path(__file__).parent / "data" / "g1_state_spec_golden.pt"


def _g1_states(n, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(n, 71, generator=g, dtype=dtype)
    s[:, 3:7] = s[:, 3:7] / s[:, 3:7].norm(dim=-1, keepdim=True)
    return s


SPECS = {
    "g1": (StateSpecCfg(), _g1_states),
    "planar_vehicle": (PlanarVehicleSpecCfg(), vehicle_states),
}


# ---------------------------------------------------------------------- contract


@pytest.mark.parametrize("name", list(SPECS))
def test_contract(name):
    cfg, states = SPECS[name]
    spec = make_state_spec(cfg, "cpu", torch.float64)
    assert isinstance(spec, StateSpecBase)
    n = 32
    s = states(n, torch.float64)
    g = torch.Generator().manual_seed(1)
    delta = torch.randn(n, spec.delta_dim, generator=g, dtype=torch.float64)
    xi = torch.rand(n, spec.xi_dim, generator=g, dtype=torch.float64)

    assert spec.canonical(s).shape == (n, spec.feat_dim)
    assert spec.normalize_xi(xi).shape == (n, spec.xi_dim)
    assert spec.integrate(s, delta).shape == (n, spec.state_dim)
    assert spec.error_vector(s, s).shape == (n, spec.err_dim)
    assert spec.error_vector(s, s).abs().max() < 1e-12
    assert spec.normalized_sq_error(s, s).shape == (n,)

    # apply_error inverts error_vector.
    err = 0.1 * torch.randn(n, spec.err_dim, generator=g, dtype=torch.float64)
    torch.testing.assert_close(spec.error_vector(spec.apply_error(s, err), s), err, atol=1e-9, rtol=1e-9)

    # pos and vel partition the error vector; the masks follow them.
    groups = spec.error_groups()
    assert sorted(groups["pos"] + groups["vel"]) == list(range(spec.err_dim))
    m_all = spec.component_mask("all")
    assert torch.equal(m_all, torch.ones(spec.err_dim))
    assert torch.equal(spec.component_mask("pos") + spec.component_mask("vel"), m_all)
    with pytest.raises(ValueError):
        spec.component_mask("joints")

    # integrate is differentiable in the network output (BPTT goes through it).
    d = delta.clone().requires_grad_(True)
    spec.normalized_sq_error(spec.integrate(s, d), s).sum().backward()
    assert d.grad is not None and d.grad.abs().sum() > 0


def test_make_state_spec_rejects_other_configs():
    with pytest.raises(TypeError):
        make_state_spec(object(), "cpu")


@pytest.mark.parametrize("name", list(SPECS))
def test_world_model_sizes_itself_from_the_spec(name):
    cfg, states = SPECS[name]
    torch.manual_seed(0)
    wm = WorldModel(WorldModelCfg(hidden=(32, 32), state_spec=cfg), heads=None, device="cpu")
    sp = wm.spec
    assert wm.dynamics.mlp.net[0].in_features == sp.feat_dim + sp.action_dim + sp.xi_dim
    assert wm.dynamics.mlp.net[-1].out_features == sp.delta_dim

    k, n = 4, 8
    s0 = states(n, torch.float32)
    a = torch.randn(k, n, sp.action_dim, requires_grad=True)
    xi = torch.rand(n, sp.xi_dim)
    with torch.no_grad():  # zero-initialised last layer: make the dynamics act on the action
        wm.dynamics.mlp.net[-1].weight.normal_(0.0, 0.1)
    s = s0
    for j in range(k):
        s = wm.step(s, a[j], xi)
    assert s.shape == (n, sp.state_dim)
    s.sum().backward()
    assert a.grad.abs().sum() > 0

    logged = torch.stack([s0] + [states(n, torch.float32, seed=j + 1) for j in range(k)])
    assert wm.rollout_error(logged, a.detach(), xi).shape == (k,)


# ---------------------------------------------------------------------- G1 regression


@pytest.fixture(scope="module")
def golden():
    return torch.load(GOLDEN, weights_only=True)


def test_g1_spec_matches_recorded_outputs(golden):
    x, ref = golden["inputs"], golden["spec64"]
    for explicit in (True, False):
        spec = StateSpec(StateSpecCfg(explicit_integration=explicit), "cpu", torch.float64)
        torch.testing.assert_close(spec.integrate(x["s"], x["delta"]), ref[f"integrate_{explicit}"], atol=0, rtol=0)
    spec = StateSpec(StateSpecCfg(), "cpu", torch.float64)
    torch.testing.assert_close(spec.canonical(x["s"]), ref["canonical"], atol=0, rtol=0)
    torch.testing.assert_close(spec.normalize_xi(x["xi"]), ref["normalize_xi"], atol=0, rtol=0)
    torch.testing.assert_close(spec.error_vector(x["s"], x["s_true"]), ref["error_vector"], atol=0, rtol=0)
    torch.testing.assert_close(spec.normalized_sq_error(x["s"], x["s_true"]), ref["normalized_sq_error"], atol=0, rtol=0)
    # The old free function and the new method give the recorded perturbations.
    torch.testing.assert_close(apply_state_delta(x["s"], x["err"], spec.err_scale), ref["apply_state_delta"], atol=0, rtol=0)
    torch.testing.assert_close(spec.apply_error(x["s"], x["err"]), ref["apply_state_delta"], atol=0, rtol=0)
    torch.testing.assert_close(spec.apply_error(x["s"], x["err"], x["limits"]), ref["apply_state_delta_limits"], atol=0, rtol=0)


def test_g1_world_model_matches_recorded_step_and_checkpoint_layout(golden):
    x, ref = golden["inputs"], golden["wm"]
    wm = WorldModel(WorldModelCfg(hidden=(64, 64)), heads=None, device="cpu")
    sd = wm.state_dict()
    # Same parameter names and shapes as before the refactor: old checkpoints still load.
    assert {k: tuple(v.shape) for k, v in sd.items()} == {k: tuple(v.shape) for k, v in ref["state_dict"].items()}
    wm.load_state_dict(ref["state_dict"])
    with torch.no_grad():
        nxt = wm.step(x["s"].float(), x["action"].float(), x["xi"].float())
    # float32 matmuls: exact on the recording machine, a small tolerance for other BLAS builds.
    torch.testing.assert_close(nxt, ref["step"], atol=1e-6, rtol=1e-5)


def test_g1_masks_and_config_unchanged():
    spec = StateSpec(StateSpecCfg(), "cpu")
    for c in ("pos", "vel", "all"):
        assert torch.equal(component_mask(c), spec.component_mask(c))
        assert torch.equal(component_mask(c, spec=spec), spec.component_mask(c))
    # Checkpoints store asdict(StateSpecCfg) and rebuild it with StateSpecCfg(**d) (eval/benchmark.py).
    d = asdict(StateSpecCfg())
    assert set(d) == {"dt", "err_scale", "out_scale", "in_scale", "explicit_integration", "xi_offset", "xi_scale"}
    assert StateSpecCfg(**d) == StateSpecCfg()
    assert (spec.state_dim, spec.feat_dim, spec.delta_dim, spec.err_dim, spec.action_dim, spec.xi_dim) == (71, 71, 70, 70, 29, 33)
    assert not math.isnan(float(spec.normalized_sq_error(_g1_states(4, torch.float32), _g1_states(4, torch.float32, 1)).mean()))


def test_g1_config_still_parses_from_the_cli():
    tyro = pytest.importorskip("tyro")
    cfg = tyro.cli(WorldModelCfg, args=["--state-spec.dt", "0.01", "--state-spec.no-explicit-integration"])
    assert isinstance(cfg.state_spec, StateSpecCfg)
    assert cfg.state_spec.dt == 0.01 and cfg.state_spec.explicit_integration is False
