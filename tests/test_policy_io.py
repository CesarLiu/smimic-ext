"""Policy / critic I/O (``NetIO``): the G1 layout is unchanged (recorded before the change), the layout
built from the three interfaces agrees with it, and a planar vehicle trains a policy and critic through
``PolicyTrainer`` with networks sized from its interfaces."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from toy_vehicle import PlanarVehicleHeads, PlanarVehicleSpecCfg, VehicleTrackingTask

from supermimic.data.replay_buffer import StartBatch
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.policy.nets import Critic, CriticCfg, NetIO, Policy, PolicyCfg
from supermimic.tasks.motion_tracking import MotionTrackingTask
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

LAYOUT = Path(__file__).parent / "data" / "g1_policy_layout.pt"


@pytest.mark.parametrize("goal_dim", [680, 686])
def test_g1_default_networks_match_the_recorded_layout(goal_dim):
    rec = torch.load(LAYOUT, weights_only=True)[f"goal{goal_dim}"]
    pc = PolicyCfg(goal_dim=goal_dim)
    for name, net in (("policy", Policy(pc)), ("critic", Critic(CriticCfg(), pcfg=pc)), ("policy", Policy(pc, NetIO.g1(pc)))):
        shapes = {k: torch.tensor(v.shape) for k, v in net.state_dict().items()}
        assert shapes.keys() == rec[name]["shapes"].keys()
        for k, v in shapes.items():
            assert torch.equal(v, rec[name]["shapes"][k]), k
        for k, v in net.named_buffers():
            assert torch.equal(v, rec[name]["buffers"][k]), k


@pytest.mark.parametrize("anchor", ["none", "ori"])
def test_g1_layout_from_the_interfaces_matches_the_config_layout(anchor):
    import g1_fixtures as fx

    heads = fx.build_heads()
    lib = fx.StubLibrary(heads.default_joint_pos)
    task = MotionTrackingTask(lib, 10, anchor, heads.anchor_index if anchor != "none" else None)
    spec = StateSpecCfg().build("cpu")
    a = NetIO.from_interfaces(spec, heads, task, history=10)
    b = NetIO.g1(PolicyCfg(goal_dim=task.goal_dim))
    for f in ("obs_dim", "goal_dim", "action_dim", "feat_dim", "xi_dim", "critic_feat_dim"):
        assert getattr(a, f) == getattr(b, f), f
    assert torch.equal(a.obs_scale, b.obs_scale)
    assert torch.equal(a.goal_scale, b.goal_scale)


def test_g1_history_other_than_ten_now_builds():
    pc = PolicyCfg(history=4, encoder_hidden=(16,), decoder_hidden=(16,))
    pol = Policy(pc)
    assert pol(torch.zeros(3, 4 * 93), torch.zeros(3, 680)).shape == (3, 29)


# ---------------------------------------------------------------------- vehicle: one full training update


class _Buffer:
    def __init__(self, start, history):
        self.start, self.history = start, history

    def sample_starts(self, batch_size, history):
        assert history == self.history and batch_size == self.start.state.shape[0]
        return self.start


def test_vehicle_policy_trainer_update():
    torch.manual_seed(0)
    spec_cfg, heads, task = PlanarVehicleSpecCfg(), PlanarVehicleHeads(), VehicleTrackingTask()
    wm = WorldModel(WorldModelCfg(hidden=(32, 32), state_spec=spec_cfg), heads, "cpu")
    with torch.no_grad():
        wm.dynamics.mlp.net[-1].weight.normal_(0.0, 0.1)
    n, history = 16, 3
    io = NetIO.from_interfaces(wm.spec, heads, task, history)
    assert (io.obs_dim, io.goal_dim, io.action_dim, io.feat_dim, io.xi_dim, io.critic_feat_dim) == (15, 12, 2, 3, 1, 2)
    pc = PolicyCfg(encoder_hidden=(32,), decoder_hidden=(32,), latent_dim=8, history=history, decoder_last_gain=0.5)
    policy, critic = Policy(pc, io), Critic(CriticCfg(hidden=(32,)), io=io)
    start = StartBatch(
        state=torch.randn(n, 6) * torch.tensor([1.0, 1.0, 0.3, 1.0, 0.3, 0.3]) + torch.tensor([0.0, 0.0, 0.0, 5.0, 0.0, 0.0]),
        xi=torch.rand(n, 1),
        frame=torch.randint(0, task.layout.num_steps, (n,)),
        prev_action=torch.zeros(n, 2),
        hist_states=torch.randn(history, n, 6),
        hist_actions=torch.zeros(history, n, 2),
    )
    trainer = PolicyTrainer(wm, policy, critic, task, PolicyTrainerCfg(horizon=6, batch_size=n, history=history, compile=None, critic_epochs=2, critic_minibatches=2))
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    m = trainer.update(_Buffer(start, history))
    assert all(torch.isfinite(torch.tensor(v)) for v in m.values())
    assert {f"imag_r/{k}" for k in heads.reward_term_names} <= set(m)
    assert m["policy_grad_norm"] > 0 and "critic_loss" in m
    changed = [k for k, v in policy.state_dict().items() if not torch.equal(v, before[k])]
    assert any(k.startswith("decoder.mlp") for k in changed) and any(k.startswith("encoders.motion.mlp") for k in changed)
