"""Vehicle end to end: the learner's view (``VehicleStateSpec``, ``VehicleHeads``, ``VehicleTask``) agrees
with the environment, and the full demo loop (collection -> replay -> world model -> imagination ->
policy / critic -> evaluation) runs. Whether the policy improves is the demo's own check
(``scripts/vehicle/demo_e2e.py``, minutes on a CPU); here it runs at a size that only exercises the code."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

from supermimic.vehicles import tracking
from supermimic.vehicles.dynamics import DELTA
from supermimic.vehicles.env import ReferenceLibrary, VehicleEnvCfg, VehicleTrackingEnv
from supermimic.vehicles.learner import A_ACT, VehicleHeads, VehicleStateSpecCfg, VehicleTask

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def setup():
    cfg = VehicleEnvCfg(num_envs=48, replay_capacity=200, episode_steps=80)
    lib = ReferenceLibrary.generate(cfg, num=4, steps=200, seed=0)
    env = VehicleTrackingEnv(cfg, lib, seed=0)
    heads = VehicleHeads(cfg, env.dyn.p.R_w)
    return cfg, lib, env, heads, VehicleTask(lib, cfg.preview), VehicleStateSpecCfg.from_env(cfg, env.dyn.p).build("cpu")


def test_learner_scores_transitions_as_the_environment(setup):
    """For every transition that did not end an episode: the heads' reward and termination features, the
    observation and the goal equal the environment's; the spec's known parts (steering angle, longitudinal
    response, pending actions) equal the simulator's."""
    cfg, _lib, env, heads, task, spec = setup
    torch.manual_seed(0)
    checked = 0
    for _ in range(60):
        s0, xi, frame, prev_a = env.state(), env.xi.clone(), env.frame.clone(), env.prev_action.clone()
        a = 0.8 * torch.randn(cfg.num_envs, 2)
        obs, r, _term, _, info = env.step(a)
        live = ~info["done"]
        if not live.any():
            continue
        s1 = env.state()[live]
        ref = task.reference(frame[live])
        out = heads.evaluate(s1, a[live], prev_a[live], xi[live], ref)
        torch.testing.assert_close(out.reward_terms["total"], r[live], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(heads.state_termination_features(s1, ref), tracking.termination_features(s1, ref))
        torch.testing.assert_close(out.obs, obs["policy"][live])
        torch.testing.assert_close(task.goal(env.frame[live], None, heads.aux(s1), False), obs["goal"][live], rtol=1e-5, atol=1e-5)
        known = spec.step_known(s0[live], s0[live].clone(), a[live], xi[live])
        idx = [DELTA, *range(A_ACT, A_ACT + 1 + 2 * cfg.max_delay)]
        torch.testing.assert_close(known[:, idx], s1[:, idx], rtol=1e-5, atol=1e-6)
        checked += int(live.sum())
    assert checked > 1000


def test_spec_contract_and_pose_invariance(setup):
    cfg, _lib, env, _heads, _task, spec = setup
    s = env.state()
    assert spec.state_dim == cfg.state_dim == s.shape[1]
    moved = s.clone()
    moved[:, 0:3] += torch.tensor([5.0, -3.0, 1.0])
    torch.testing.assert_close(spec.canonical(moved), spec.canonical(s))  # the network never sees the pose
    s64 = s.double()  # positions reach hundreds of metres: float32 cannot resolve small errors there
    err = 0.1 * torch.randn_like(s64)
    torch.testing.assert_close(spec.error_vector(spec.apply_error(s64, err), s64), err, rtol=1e-6, atol=1e-8)
    assert sorted(spec.error_groups()["pos"] + spec.error_groups()["vel"]) == list(range(spec.err_dim))


def test_demo_runs_end_to_end(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts" / "vehicle"))
    import demo_e2e

    out = tmp_path / "report.json"
    code = demo_e2e.main(["--iters", "50", "--warmup", "10", "--policy-warmup", "20", "--num-envs", "32", "--refs", "3", "--eval-refs", "2",
                          "--ref-steps", "120", "--episode-steps", "40", "--batch", "32", "--horizon", "6", "--log-every", "25", "--out", str(out)])
    rep = json.loads(out.read_text())
    assert code in (0, 1) and rep["passed"] == (code == 0)  # far too short to have to pass; it must run
    assert math.isfinite(rep["wm_8step_error"]) and math.isfinite(rep["after"]["normalized_return"])
    last = rep["log"][-1]
    assert all(math.isfinite(last[k]) for k in ("wm_nse_8step", "imag_return", "critic_loss"))


def test_limit_study_runs(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts" / "vehicle"))
    import limit_study

    out = tmp_path / "limit.json"
    code = limit_study.main(["--models", "small", "--mu-levels", "0.6", "1.0", "--refs-per-level", "2", "--eval-refs-per-level", "2", "--ref-steps", "120",
                             "--iters", "40", "--warmup", "10", "--policy-warmup", "20", "--num-envs", "24", "--episode-steps", "40", "--batch", "24",
                             "--horizon", "5", "--log-every", "20", "--out", str(out)])
    rep = json.loads(out.read_text())
    assert code == 0
    after = rep["results"]["small"]["after"]
    assert set(after) == {"mu_0.6", "mu_1.0"}
    assert all(0.0 <= lv["near_limit_survived_share"] <= 1.0 for lv in after.values())
