"""Imagined rollouts through the world model (paper §3.3).

Robot-specific parts (observation, its noise and history layout, reward, termination and critic
features) come from ``wm.heads`` (``AnalyticHeadsBase``); references, goals and the advance of task
time come from the task (``TaskBase``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.imagination.compiled import ImagStep
from supermimic.tasks.base import TaskBase
from supermimic.tasks.motion_tracking import ANCHOR_NOISE_HALF  # noqa: F401  (re-exported, moved)
from supermimic.tasks.motion_tracking import as_task
from supermimic.world_model.heads import OBS_NOISE_HALF  # noqa: F401  (re-exported, G1 table)
from supermimic.world_model.heads_base import ref_slice
from supermimic.world_model.world_model import WorldModel


@dataclass
class ImagRollout:
    states: torch.Tensor  # (H+1, n, state_dim)
    actions: torch.Tensor  # (H, n, action_dim)
    rewards: torch.Tensor  # (H, n)      r_{t+1}
    dhat: torch.Tensor  # (H+1, n)    predicted termination prob of s_t (d_0 = 0)
    survival: torch.Tensor  # (H+1, n)    c_t
    goals: torch.Tensor  # (H+1, n, goal_dim)
    e_root: torch.Tensor  # (H+1, n, critic_feat_dim)  privileged critic features
    frames: torch.Tensor  # (H+1, n)    target steps of task time
    reward_terms: dict  # mean of each reward term over the rollout (diagnostics)


def imagine(
    wm: WorldModel,
    policy,
    task: TaskBase,
    start: StartBatch,
    horizon: int,
    future_frames: int = 10,
    obs_noise: bool = True,
    step_fn: ImagStep | None = None,
    anchor_obs: bool | str = False,
) -> ImagRollout:
    """Roll ``policy(obs_history_flat, goal) -> action`` through the frozen world model for ``horizon`` steps.

    ``task`` is a ``TaskBase``; a ``MotionLibrary`` is accepted and wrapped as motion tracking with
    ``future_frames`` and ``anchor_obs`` (which are not used when a task is given).
    """
    if isinstance(task, TaskBase) and (future_frames != 10 or anchor_obs not in (False, "none")):
        raise ValueError("future_frames / anchor_obs configure the motion-library wrapper; set them on the task instead")
    heads = wm.heads
    task = as_task(task, heads, future_frames, anchor_obs)
    device = start.state.device
    n = start.state.shape[0]
    step_fn = step_fn or ImagStep(wm)
    noise_half = heads.obs_noise_half().to(device)

    def add_noise(o):
        if obs_noise:
            return o + (torch.rand_like(o) * 2 - 1) * noise_half
        return o

    # Initial observation history from the logged states/actions (noise re-sampled per frame).
    hist = [add_noise(heads.observation(start.hist_states[i], start.hist_actions[i], start.xi)) for i in range(start.hist_states.shape[0])]
    s = start.state
    frame = start.frame.clone()
    a_prev = start.prev_action
    c = torch.ones(n, device=device)

    states = [s]
    actions, rewards = [], []
    dhats, survivals, frames = [torch.zeros(n, device=device)], [c], [frame]
    # All target steps of the rollout are known up front: gather the references in one batched call.
    frames_all = task.targets(frame, horizon)  # (H+1, n)
    ref_all = task.reference(frames_all.reshape(-1))

    def ref_at(t):
        return ref_slice(ref_all, slice(t * n, (t + 1) * n))

    ref = ref_at(0)
    aux = heads.aux(s) if task.needs_aux else None
    goals = [task.goal(frame, ref, aux, obs_noise)]
    e_roots = [heads.critic_features(s, ref)]
    term_names = heads.reward_term_names
    term_sum = torch.zeros(len(term_names), device=device)

    for t in range(horizon):
        obs_flat = heads.flatten_obs_history(torch.stack(hist, 0))
        a = policy(obs_flat, goals[-1])
        s_next, r, term_vec, logits, obs_clean, aux = step_fn(s, a, a_prev, start.xi, ref)
        d = torch.sigmoid(logits)
        c = c * (1.0 - d)
        term_sum += term_vec.detach().mean(-1)

        frame = frames_all[t + 1]
        ref = ref_at(t + 1)
        hist = hist[1:] + [add_noise(obs_clean)]
        a_prev = a
        s = s_next
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dhats.append(d)
        survivals.append(c)
        goals.append(task.goal(frame, ref, aux, obs_noise))
        e_roots.append(heads.critic_features(s, ref))
        frames.append(frame)

    return ImagRollout(
        states=torch.stack(states),
        actions=torch.stack(actions),
        rewards=torch.stack(rewards),
        dhat=torch.stack(dhats),
        survival=torch.stack(survivals),
        goals=torch.stack(goals),
        e_root=torch.stack(e_roots),
        frames=torch.stack(frames),
        reward_terms={k: float(v / horizon) for k, v in zip(term_names, term_sum.tolist())},
    )
