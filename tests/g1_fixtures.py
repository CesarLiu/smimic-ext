"""CPU fixtures for the G1 analytic-heads / imagination regression (no simulator, no motion data).

``run_all()`` drives every public entry point that touches the analytic heads, on fixed inputs:
the reward, observation and termination features, the world model's termination logits and k-step
loss, one imagined step, a whole imagined rollout (with and without anchor observations) and one
policy + critic update. ``tests/data/make_g1_heads_golden.py`` records its output and
``tests/test_heads_interface.py`` compares the current code against the recording.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.imagination.compiled import ImagStep
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.imagination.rollout import imagine
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame, analytic_observation, flatten_history_term_major
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

N = 8  # batch
B = len(TRACKED_BODIES)
F = 64  # frames in the stub library
HIST = 10
FUTURE = 2
HORIZON = 4


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def build_heads():
    model = load_g1_model()
    q0 = torch.tensor(model.qpos0[7:], dtype=torch.float32)
    limits = torch.tensor(model.jnt_range[1:], dtype=torch.float32)
    return AnalyticHeads(G1Kinematics(model, device="cpu"), TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, limits, q0)


def states(n, seed, q0):
    g = _gen(seed)
    s = torch.zeros(n, 71)
    s[:, 0:2] = 0.3 * torch.randn(n, 2, generator=g)
    s[:, 2] = 0.78 + 0.03 * torch.randn(n, generator=g)
    s[:, 3:7] = _unit(torch.tensor([1.0, 0.0, 0.0, 0.0]) + 0.1 * torch.randn(n, 4, generator=g))
    s[:, 7:36] = q0 + 0.1 * torch.randn(n, 29, generator=g)
    s[:, 36:71] = 0.3 * torch.randn(n, 35, generator=g)
    return s


def reference(n, seed):
    g = _gen(seed)
    return ReferenceFrame(
        body_pos_w=torch.tensor([0.0, 0.0, 0.8]) + 0.3 * torch.randn(n, B, 3, generator=g),
        body_quat_w=_unit(torch.tensor([1.0, 0.0, 0.0, 0.0]) + 0.1 * torch.randn(n, B, 4, generator=g)),
        body_lin_vel_w=0.5 * torch.randn(n, B, 3, generator=g),
        body_ang_vel_w=0.5 * torch.randn(n, B, 3, generator=g),
        root_pos_w=torch.tensor([0.0, 0.0, 0.78]) + 0.1 * torch.randn(n, 3, generator=g),
        root_quat_w=_unit(torch.tensor([1.0, 0.0, 0.0, 0.0]) + 0.1 * torch.randn(n, 4, generator=g)),
    )


@dataclass
class _Ref:
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor


@dataclass
class _Bodies:
    pos_w: torch.Tensor
    quat_w: torch.Tensor
    lin_vel_w: torch.Tensor
    ang_vel_w: torch.Tensor


class StubLibrary:
    """The slice of ``MotionLibrary`` the imagination uses, over F random frames in two clips."""

    def __init__(self, q0):
        self.default_joint_pos = q0
        r = reference(F, 11)
        self._root_pos, self._root_quat = r.root_pos_w, r.root_quat_w
        self._bodies = _Bodies(r.body_pos_w, r.body_quat_w, r.body_lin_vel_w, r.body_ang_vel_w)
        self._goal = torch.randn(F, 68, generator=_gen(12))
        self._last = torch.where(torch.arange(F) < F // 2, F // 2 - 1, F - 1)

    def ref(self, frame):
        return _Ref(self._root_pos[frame], self._root_quat[frame])

    def bodies(self, frame):
        b = self._bodies
        return _Bodies(b.pos_w[frame], b.quat_w[frame], b.lin_vel_w[frame], b.ang_vel_w[frame])

    def last_frame_of(self, frame):
        return self._last[frame]

    def goal_window(self, frame, k):
        idx = torch.minimum(frame[:, None] + torch.arange(k)[None], self.last_frame_of(frame)[:, None])
        return self._goal[idx.reshape(-1)].reshape(frame.shape[0], k * 68)


def start_batch(q0):
    g = _gen(21)
    return StartBatch(
        state=states(N, 22, q0),
        xi=torch.cat([0.75 + 0.1 * torch.randn(N, 1, generator=g), 0.01 * torch.randn(N, 32, generator=g)], -1),
        frame=torch.tensor([0, 5, 20, 30, 31, 40, 50, 60]),
        prev_action=0.3 * torch.randn(N, 29, generator=g),
        hist_states=torch.stack([states(N, 100 + i, q0) for i in range(HIST)]),
        hist_actions=0.3 * torch.randn(HIST, N, 29, generator=g),
    )


class StubBuffer:
    def __init__(self, start):
        self.start = start

    def sample_starts(self, batch_size, history):
        assert batch_size == N and history == HIST
        return self.start


def world_model(heads):
    torch.manual_seed(0)
    wm = WorldModel(WorldModelCfg(hidden=(64, 64)), heads, "cpu")
    with torch.no_grad():  # zero-initialised last layer: give the dynamics something to do
        wm.dynamics.mlp.net[-1].weight.normal_(0.0, 0.02)
        wm.dynamics.mlp.net[-1].bias.normal_(0.0, 0.02)
    return wm


def policy_cfg(anchor):
    return PolicyCfg(encoder_hidden=(32,), decoder_hidden=(32,), goal_dim=FUTURE * 68 + {"none": 0, "ori": 6}[anchor])


def run_all() -> dict:
    torch.set_num_threads(1)
    heads = build_heads()
    q0 = heads.default_joint_pos
    out: dict = {}

    # ---- heads on their own
    s = states(N, 1, q0)
    s_next = states(N, 2, q0)
    a = 0.3 * torch.randn(N, 29, generator=_gen(3))
    a_prev = 0.3 * torch.randn(N, 29, generator=_gen(4))
    xi = start_batch(q0).xi
    ref = reference(N, 5)
    out["reward"] = {k: v for k, v in heads.reward(s_next, a, a_prev, ref).items()}
    out["observation"] = analytic_observation(s_next, a, xi, q0)
    out["obs_history_flat"] = flatten_history_term_major(torch.stack([analytic_observation(states(N, 30 + i, q0), a, xi, q0) for i in range(HIST)]))
    out["root_error_features"] = heads.root_error_features(s_next, ref)
    out["termination_features"] = heads.termination_features(s_next[..., 0:3], s_next[..., 3:7], ref)

    # ---- world model: termination logits and the k-step loss
    wm = world_model(heads)
    out["termination_logits"] = wm.termination_logits(s_next, ref)
    k = 3
    win_states = torch.stack([states(N, 40 + j, q0) for j in range(k + 1)])
    win_a = 0.3 * torch.randn(k, N, 29, generator=_gen(50))
    done = (torch.rand(k, N, generator=_gen(51)) < 0.2).float()
    refs = [reference(N, 60 + j) for j in range(k)]
    loss, metrics = wm.window_loss(win_states, win_a, xi, done, refs)
    out["window_loss"] = loss.detach()
    out["window_metrics"] = {kk: v.detach() for kk, v in metrics.items()}

    # ---- one imagined step
    step = ImagStep(wm, compile_mode=None)
    out["imag_step"] = tuple(t.detach() for t in step(s, a, a_prev, xi, ref))

    # ---- whole imagined rollouts and one policy + critic update
    lib = StubLibrary(q0)
    start = start_batch(q0)
    for anchor in ("none", "ori"):
        pcfg = policy_cfg(anchor)
        torch.manual_seed(7)
        pol = Policy(pcfg)
        for noise in (False, True):
            torch.manual_seed(8)
            ro = imagine(wm, pol, lib, start, HORIZON, FUTURE, obs_noise=noise, anchor_obs=anchor)
            out[f"imagine_{anchor}_{noise}"] = {
                "states": ro.states.detach(),
                "actions": ro.actions.detach(),
                "rewards": ro.rewards.detach(),
                "dhat": ro.dhat.detach(),
                "survival": ro.survival.detach(),
                "goals": ro.goals.detach(),
                "e_root": ro.e_root.detach(),
                "frames": ro.frames,
            }
        torch.manual_seed(9)
        critic = Critic(CriticCfg(hidden=(32,)), pcfg=pcfg)
        tcfg = PolicyTrainerCfg(horizon=HORIZON, batch_size=N, future_frames=FUTURE, compile=None, critic_epochs=2, critic_minibatches=2, anchor_obs=anchor)
        trainer = PolicyTrainer(wm, pol, critic, lib, tcfg)
        torch.manual_seed(10)
        m = trainer.update(StubBuffer(start))
        out[f"update_{anchor}"] = {
            "metrics": {kk: torch.tensor(v, dtype=torch.float64) for kk, v in m.items()},
            "policy": {kk: v.detach().clone() for kk, v in pol.state_dict().items()},
            "critic": {kk: v.detach().clone() for kk, v in critic.state_dict().items()},
        }
    return out
