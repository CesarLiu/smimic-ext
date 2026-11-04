"""Task interface and the task-agnostic reset sampler: G1 regressions against outputs recorded before
the refactor, the contract every task must meet, and a planar vehicle task (``toy_vehicle.py``) that
runs imagination and the reset sampler end to end through the three interfaces."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from toy_vehicle import PlanarVehicleHeads, PlanarVehicleSpecCfg, VehicleTrackingTask

from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.commands import normalize_return_to_go as from_commands
from supermimic.imagination.rollout import imagine
from supermimic.tasks.base import TaskBase
from supermimic.tasks.motion_tracking import MotionTrackingTask, as_task
from supermimic.tasks.reset_sampler import (
    ResetSamplerCfg,
    ReturnBasedResetSampler,
    SequenceLayout,
    normalize_return_to_go,
)
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

DATA = Path(__file__).parent / "data"


# ---------------------------------------------------------------------- G1 regressions


def test_reset_sampler_matches_recorded_behaviour():
    import sampler_fixtures as sf

    golden = torch.load(DATA / "sampler_golden.pt", weights_only=True)
    out = sf.run_all()
    assert set(out) == set(golden)
    for cfg_name, rec in golden.items():
        for k, v in rec.items():
            if v.is_floating_point():
                torch.testing.assert_close(out[cfg_name][k], v, rtol=1e-5, atol=1e-6, msg=lambda m, c=cfg_name, k=k: f"{c}/{k}: {m}")
            else:
                assert torch.equal(out[cfg_name][k], v), f"{cfg_name}/{k}"


def test_sequence_layout_from_lengths_matches_the_motion_library():
    import sampler_fixtures as sf

    lib = sf.build_library()
    a = SequenceLayout.from_motion_library(lib)
    b = SequenceLayout.from_lengths(sf.CLIP_LENGTHS, sf.BIN)
    for f in ("seq_start", "seq_last", "step_seq", "seq_bin_offset", "bin_start", "bin_end"):
        assert torch.equal(getattr(a, f), getattr(b, f)), f
    idx = torch.arange(lib.num_frames)
    assert torch.equal(b.bin_of(idx), lib.bin_of_frame(idx))
    assert torch.equal(b.last_of(idx), lib.last_frame_of(idx))
    assert from_commands is normalize_return_to_go


@pytest.mark.parametrize("anchor", ["none", "ori"])
def test_explicit_motion_tracking_task_reproduces_the_library_call(anchor):
    """imagine(…, task) and PolicyTrainer(…, task) give the recorded results of the library call style."""
    import g1_fixtures as fx

    from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
    from supermimic.policy.nets import Critic, CriticCfg, Policy

    golden = torch.load(DATA / "g1_heads_golden.pt", weights_only=True)
    torch.set_num_threads(1)
    heads = fx.build_heads()
    q0 = heads.default_joint_pos
    wm = fx.world_model(heads)
    lib = fx.StubLibrary(q0)
    start = fx.start_batch(q0)
    task = MotionTrackingTask(lib, fx.FUTURE, anchor, heads.anchor_index if anchor != "none" else None)
    pcfg = fx.policy_cfg(anchor)
    torch.manual_seed(7)
    pol = Policy(pcfg)
    torch.manual_seed(8)
    ro = imagine(wm, pol, task, start, fx.HORIZON, obs_noise=True)
    rec = golden[f"imagine_{anchor}_True"]
    for k in ("states", "actions", "rewards", "survival", "goals", "e_root"):
        torch.testing.assert_close(getattr(ro, k).detach(), rec[k], rtol=1e-4, atol=1e-6)
    assert torch.equal(ro.frames, rec["frames"])

    torch.manual_seed(9)
    critic = Critic(CriticCfg(hidden=(32,)), pcfg=pcfg)
    tcfg = PolicyTrainerCfg(horizon=fx.HORIZON, batch_size=fx.N, future_frames=fx.FUTURE, compile=None, critic_epochs=2, critic_minibatches=2, anchor_obs=anchor)
    trainer = PolicyTrainer(wm, pol, critic, task, tcfg)
    assert trainer.task is task
    torch.manual_seed(10)
    m = trainer.update(fx.StubBuffer(start))
    for k, v in golden[f"update_{anchor}"]["metrics"].items():
        assert abs(m[k] - float(v)) <= 1e-4 * max(1.0, abs(float(v))), k


def test_library_wrapper_and_misuse():
    import g1_fixtures as fx

    heads = fx.build_heads()
    lib = fx.StubLibrary(heads.default_joint_pos)
    t = as_task(lib, heads, 3, "ori")
    assert isinstance(t, MotionTrackingTask) and t.goal_dim == 3 * 68 + 6 and t.needs_aux
    assert as_task(t) is t
    with pytest.raises(ValueError):
        MotionTrackingTask(lib, 2, "ori")  # anchor goals need the anchor index
    with pytest.raises(ValueError):  # wrapper options next to an explicit task
        imagine(None, None, t, None, 1, anchor_obs="ori")


# ---------------------------------------------------------------------- contract


def _g1_task_case():
    import g1_fixtures as fx

    heads = fx.build_heads()
    lib = fx.StubLibrary(heads.default_joint_pos)
    task = MotionTrackingTask(lib, fx.FUTURE, "ori", heads.anchor_index)
    idx = torch.tensor([0, 5, 31, 40, 63])
    last = lib.last_frame_of(idx)
    return task, heads, idx, last, heads.aux(fx.states(idx.shape[0], 3, heads.default_joint_pos))


def _vehicle_task_case():
    task = VehicleTrackingTask()
    idx = torch.tensor([0, 35, 39, 64, 124])
    return task, PlanarVehicleHeads(), idx, task.layout.last_of(idx), None


@pytest.mark.parametrize("name", ["motion_tracking", "vehicle"])
def test_contract(name):
    task, _heads, idx, last, aux = _g1_task_case() if name == "motion_tracking" else _vehicle_task_case()
    assert isinstance(task, TaskBase)
    n, H = idx.shape[0], 6
    tg = task.targets(idx, H)
    assert tg.shape == (H + 1, n)
    assert torch.equal(tg[0], idx)
    assert bool((tg[1:] >= tg[:-1]).all()) and bool((tg <= last[None]).all())
    ref = task.reference(tg.reshape(-1))
    g = task.goal(idx, task.reference(idx), aux, noise=False)
    assert g.shape == (n, task.goal_dim)
    assert task.goal(idx, task.reference(idx), aux, noise=False).equal(g)  # deterministic without noise
    # The heads accept the task's references.
    s_ref = task.termination_reference(idx)
    assert type(s_ref) is type(ref)


# ---------------------------------------------------------------------- vehicle end to end


def test_vehicle_imagination_through_all_three_interfaces():
    torch.manual_seed(0)
    task, heads = VehicleTrackingTask(), PlanarVehicleHeads()
    wm = WorldModel(WorldModelCfg(hidden=(32, 32), state_spec=PlanarVehicleSpecCfg()), heads, "cpu")
    with torch.no_grad():
        wm.dynamics.mlp.net[-1].weight.normal_(0.0, 0.1)
    wm.requires_grad_(False)
    n, hist_len, horizon = 6, 3, 8
    lin = torch.nn.Linear(hist_len * heads.obs_dim + task.goal_dim, 2)
    start = StartBatch(
        state=torch.randn(n, 6),
        xi=torch.rand(n, 1),
        frame=torch.tensor([0, 10, 38, 45, 64, 120]),
        prev_action=torch.zeros(n, 2),
        hist_states=torch.randn(hist_len, n, 6),
        hist_actions=torch.zeros(hist_len, n, 2),
    )
    ro = imagine(wm, lambda o, g: lin(torch.cat([o, g], -1)), task, start, horizon)
    assert ro.states.shape == (horizon + 1, n, 6) and ro.goals.shape == (horizon + 1, n, task.goal_dim)
    assert ro.e_root.shape == (horizon + 1, n, heads.critic_feat_dim)
    assert set(ro.reward_terms) == set(heads.reward_term_names)
    assert bool((ro.frames <= task.layout.last_of(start.frame)[None]).all())  # clamped at trajectory ends
    (ro.survival[:-1] * ro.rewards).sum().backward()
    assert lin.weight.grad.abs().sum() > 0


def test_reset_sampler_on_the_vehicle_layout_favours_hard_bins():
    task = VehicleTrackingTask()
    lay = task.layout
    torch.manual_seed(0)
    sampler = ReturnBasedResetSampler(lay, ResetSamplerCfg(g_max=1.0, ema_alpha=1.0, max_logged_steps=64), num_envs=4, episode_steps=64, device="cpu")
    # Env 0 fails right after starting in bin 3; env 1 tracks perfectly to the end of trajectory 0.
    starts = torch.tensor([lay.bin_start[3], 0, 0, 0])
    for t in range(40):
        ids = torch.tensor([1]) if t else torch.tensor([0, 1])
        sampler.log_step(ids, torch.full((ids.numel(),), 0.01), starts[ids] + t)
    sampler.finalize(torch.tensor([0, 1]), terminated=torch.tensor([True, False]), seq_end=torch.tensor([False, True]))
    assert float(sampler.bin_return[3]) < float(sampler.bin_return[1])
    f, ent = sampler.sample(4000)
    assert bool((f >= 0).all()) and bool((f <= lay.last_of(f)).all()) and ent is not None
    share = (lay.bin_of(f) == 3).float().mean()
    assert float(share) > 2.0 / lay.num_bins  # the failed bin is drawn well above uniform


def test_reset_sampler_ignores_non_finite_returns():
    """A diverged episode (non-finite rewards) must not poison its bins or the sampling weights."""
    from supermimic.tasks.reset_sampler import ReturnBasedResetSampler, ResetSamplerCfg, SequenceLayout

    lay = SequenceLayout.from_lengths([40, 40], bin_size=10)
    smp = ReturnBasedResetSampler(lay, ResetSamplerCfg(), num_envs=2, episode_steps=50, device="cpu")
    for t in range(12):
        r = torch.tensor([0.02, float("nan") if t == 5 else 0.02])
        smp.log_step(torch.arange(2), r, torch.tensor([t, 40 + t]))
    smp.finalize(torch.arange(2), torch.tensor([True, True]), torch.tensor([False, False]))
    assert bool(torch.isfinite(smp.bin_return).all())
    f, _ = smp.sample(500)
    assert f.shape == (500,)
