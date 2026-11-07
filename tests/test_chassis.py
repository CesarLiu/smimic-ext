"""Chassis actuators (VEHICLE_PROPOSAL.md §10, C1): rear-wheel steering, torque-vectoring yaw moment and the
front / rear roll-stiffness split act in the physically expected direction; the environment, the world model's
exact actuator update and the observation agree; the plain vehicle is unchanged."""

from __future__ import annotations

import torch

from supermimic.vehicles.dynamics import DELTA, R, VY, AssistCfg, Context, VehicleDynamics
from supermimic.vehicles.env import ReferenceLibrary, VehicleEnvCfg, VehicleTrackingEnv
from supermimic.vehicles.learner import A_ACT, VehicleHeads, VehicleStateSpecCfg


def _steady(chassis, delta=0.03, v=20.0, steps=150):
    """Yaw rate and sideslip velocity after ``steps`` control steps at a fixed front-wheel angle (no assists)."""
    dyn = VehicleDynamics(assist=AssistCfg(abs=False, esc=False))
    n = chassis.shape[0]
    s = dyn.initial_state(torch.full((n,), v), delta=torch.full((n,), delta))
    flags, ctx = dyn.new_flags(n), Context.nominal(n)
    z = torch.zeros(n)
    for _ in range(steps):
        s, flags = dyn.step(s, flags, z, z, ctx, chassis)
    return s[:, R], s[:, VY]


def test_neutral_chassis_matches_the_plain_vehicle():
    dyn = VehicleDynamics(assist=AssistCfg(abs=False, esc=False))
    s0 = dyn.initial_state(torch.full((3,), 18.0), delta=torch.tensor([0.0, 0.02, -0.04]))
    flags, ctx, z = dyn.new_flags(3), Context.nominal(3), torch.zeros(3)
    a, fa = s0.clone(), flags.clone()
    b, fb = s0.clone(), flags.clone()
    for _ in range(50):
        a, fa = dyn.step(a, fa, z, z, ctx)
        b, fb = dyn.step(b, fb, z, z, ctx, torch.zeros(3, 3))
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_rear_steering():
    """Counter-phase rear steering (rear wheels opposite to the front) raises the yaw rate, in-phase lowers it."""
    r, _ = _steady(torch.tensor([[0.0, 0.0, 0.0], [-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]]))
    assert r[1] > r[0] * 1.2 and r[2] < r[0] * 0.8


def test_torque_vectoring_yaw_moment():
    """A positive yaw moment turns the car more to the left; a negative one less; nothing in a straight line but yaw."""
    r, _ = _steady(torch.tensor([[0.0, 0.0, 0.0], [0.0, 1500.0, 0.0], [0.0, -1500.0, 0.0]]))
    assert r[1] > r[0] + 0.02 and r[2] < r[0] - 0.02
    r0, _ = _steady(torch.tensor([[0.0, 1500.0, 0.0]]), delta=0.0, steps=25)
    assert r0[0] > 0.01


def test_roll_split_shifts_understeer():
    """More of the roll stiffness at the front moves load transfer to the front axle: more understeer, lower yaw rate;
    less at the front: more yaw. At about 0.7 g the effect is small (about 1 %); closer to the limit, with a steady steer
    and no ESC, the shifted split can lift the inner front wheel and the response is erratic (VEHICLE_PROPOSAL.md §10.5)."""
    r, _ = _steady(torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, -0.2]]), delta=0.08, v=15.0)
    assert r[1] < r[0] < r[2]


def _chassis_env(n=8):
    cfg = VehicleEnvCfg(num_envs=n, chassis=True, replay_capacity=20, domain_randomization=True)
    lib = ReferenceLibrary.generate(VehicleEnvCfg(), num=2, steps=120, seed=0)
    return cfg, VehicleTrackingEnv(cfg, lib, seed=0)


def test_environment_layout():
    cfg, env = _chassis_env()
    assert cfg.action_dim == 5 and env.state().shape == (8, cfg.state_dim) and cfg.state_dim == 17 + 4 + 3 * 5 + 5
    heads = VehicleHeads(cfg, env.dyn.p.R_w)
    obs = env.reset()["policy"]
    assert obs.shape == (8, heads.obs_dim) == (8, 18)
    assert heads.obs_noise_half().shape == (heads.obs_dim,) and heads.obs_scale().shape == (heads.obs_dim,)
    # the heads' observation from the logged state is the environment's
    a = torch.randn(8, 5)
    obs2, *_ = env.step(a)
    live = ~torch.zeros(8, dtype=torch.bool)
    torch.testing.assert_close(heads.observation(env.state(), env.prev_action, env.xi)[live], obs2["policy"][live])
    assert env.buffer is not None


def test_world_model_actuator_update_is_exact():
    """``step_known`` reproduces the environment's actuator states (steering angle, longitudinal response, chassis
    actuators, pending commands) for random actions and delays."""
    torch.manual_seed(0)
    cfg, env = _chassis_env()
    spec = VehicleStateSpecCfg.from_env(cfg, env.dyn.p).build("cpu")
    assert spec.state_dim == cfg.state_dim and spec.action_dim == 5
    for _ in range(10):
        s, xi = env.state().clone(), env.xi.clone()
        a = 2 * torch.randn(8, 5)
        _o, _r, _t, _tr, info = env.step(a)
        keep = ~info["done"]
        pred = spec.step_known(s, torch.zeros_like(s), a, xi)
        cols = [DELTA, *range(A_ACT, cfg.state_dim - 5)]
        torch.testing.assert_close(pred[keep][:, cols], env.state()[keep][:, cols], rtol=1e-5, atol=1e-5)


def test_chassis_actuators_respond_and_saturate():
    cfg, env = _chassis_env(2)
    env.reset()
    a = torch.tensor([[0.0, 0.0, 3.0, 3.0, 3.0], [0.0, 0.0, -3.0, -3.0, -3.0]])
    for _ in range(40):
        env.step(a)
    ch = env.act.chassis
    t3 = float(torch.tanh(torch.tensor(3.0)))
    torch.testing.assert_close(ch[0], torch.tensor([cfg.rear_steer_scale * t3, cfg.yaw_moment_scale * t3, cfg.roll_split_scale * t3]), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ch[1], -ch[0], rtol=1e-3, atol=1e-3)


def test_chassis_demo_end_to_end(tmp_path):
    """Tiny plain and chassis runs through compare_ppo.py, then the demo's comparison (no video)."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vehicle"))
    import chassis_demo
    import compare_ppo

    small = ["--methods", "supermimic", "ppo", "--budget-min", "0.05", "--eval-every-s", "100", "--eval-at-steps", "--mu-levels", "1.0",
             "--refs-per-level", "2", "--eval-refs-per-level", "1", "--ref-steps", "120", "--episode-steps", "40", "--num-envs", "16", "--batch", "16",
             "--warmup", "20", "--policy-warmup", "30", "--horizon", "4", "--ppo-envs", "16", "--ppo-steps", "8", "--no-plots"]
    assert compare_ppo.main(small + ["--out-dir", str(tmp_path / "plain")]) == 0
    assert compare_ppo.main(small + ["--chassis", "--out-dir", str(tmp_path / "chassis")]) == 0
    code = chassis_demo.main(["--plain-from", str(tmp_path / "plain"), "--chassis-from", str(tmp_path / "chassis"), "--elk-dry-kmh", "50",
                              "--elk-wet-kmh", "--no-video", "--out-dir", str(tmp_path / "demo")])
    assert code == 0
    s = json.loads((tmp_path / "demo" / "summary.json").read_text())
    assert set(s["training"]) == {"plain", "chassis"} and set(s["training"]["chassis"]) == {"supermimic", "ppo"}
    case = s["elk"]["mu_1.0_50kmh"]
    assert {"supermimic", "ppo", "supermimic_chassis", "ppo_chassis"} <= set(case)
