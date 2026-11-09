"""``xi_gradient_check.SimAsWM``: the simulator behind the world model's interface runs through ``imagine`` with the
same rewards and policy calls, its return responds to the context, and the world model's dJ/dxi is finite."""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vehicle"))


def test_sim_as_wm_closed_loop():
    pytest.importorskip("casadi")  # compare_ppo imports the NMPC baseline
    import compare_ppo
    from limit_study import env_cfg, libraries
    from xi_gradient_check import SimAsWM, closed_loop_return, collect_starts

    from supermimic.vehicles import jacobian as J
    from supermimic.vehicles.dynamics import VehicleDynamics
    from supermimic.vehicles.env import ReferenceLibrary, VehicleTrackingEnv
    from supermimic.vehicles.learner import VehicleStateSpecCfg
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    torch.manual_seed(0)
    run = types.SimpleNamespace(**{**vars(compare_ppo.parse(["--refs-per-level", "2", "--ref-steps", "200"])), "device": "cpu"})
    cfg = env_cfg(run, 8, 80)
    lib = ReferenceLibrary.concat(libraries(run, cfg, 2, 0))
    env = VehicleTrackingEnv(cfg, lib, seed=0)
    heads, task, *_r, policy, _critic = compare_ppo.networks(run, env, lib)
    wm = WorldModel(WorldModelCfg(hidden=(32, 32), state_spec=VehicleStateSpecCfg.from_env(cfg, env.dyn.p)), heads, "cpu")
    start = collect_starts(env, heads, policy, run, 6, 30)
    sim = SimAsWM(wm, VehicleDynamics(env.dyn.p, cfg.dt, cfg.substeps, cfg.assist), cfg)
    s1 = sim.step(start.state, torch.zeros(6, 2), start.xi)
    assert torch.equal(s1, J.logged_step(sim.dyn, cfg, start.state, torch.zeros(6, 2), start.xi))
    with torch.no_grad():
        J0, ro = closed_loop_return(sim, policy, task, start, 5)
        xi = start.xi.clone()
        xi[:, 2] *= 1.3  # heavier car
        J1, _ = closed_loop_return(sim, policy, task, replace(start, xi=xi), 5)
    assert J0.shape == (6,) and ro.rewards.shape == (5, 6) and torch.isfinite(J0).all()
    assert not torch.allclose(J0, J1)  # the true return depends on the context
    xi = start.xi.clone().requires_grad_(True)
    Jw, _ = closed_loop_return(wm, policy, task, replace(start, xi=xi), 5)
    (g,) = torch.autograd.grad(Jw.sum(), xi)
    assert g.shape == start.xi.shape and torch.isfinite(g).all()
