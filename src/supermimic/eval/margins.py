"""Branching runs for counterfactual robustness margins (driven by scripts/eval_margins.py).

Duck-typed on the environment (``num_envs``, ``device``, ``reset``, ``step``), the motion command
(``forced_frame``, ``library.clip_start``), the state I/O (``read`` / ``write`` / ``sync``) and the world
model (``step``, ``spec.error_vector``), so it carries no simulator imports and can be tested with stand-ins.

Envs that have not branched yet are copied from the control (env 0) at every step. Identical inputs do
not keep parallel envs identical over long clips (float rounding differs per env origin, and tracking is
chaotic over thousands of steps), so without the copy a branch would start from a state that differs
from the nominal one by more than its perturbation. The reported drift is the largest one-step
divergence before the copy, in normalised error units, the same units as the perturbation sizes.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from supermimic.eval.perturb import apply_state_delta


def branch_times(length: int, first: int, every: int, window: int, max_times: int) -> list[int]:
    """Branch frames from ``first`` on, every ``every`` frames, leaving a full window before the clip end."""
    times = list(range(first, length - window - 1, every))
    if len(times) > max_times:
        idx = np.unique(np.linspace(0, len(times) - 1, max_times).round().astype(int))
        times = [times[i] for i in idx]
    return times


def run_branch_pass(env, cmd, io, policy, wm, xi, clip, times, chunk, kind, scales, dirs, mask, err_scale, limits, window, action_steps, wm_k):
    """One synchronous play of ``clip`` with the branches in ``chunk`` ((time idx, dir idx, scale idx) per env 1..)."""
    n, dev = env.num_envs, env.device
    branch_t = torch.full((n,), -1, dtype=torch.long, device=dev)
    scale = torch.zeros(n, device=dev)
    dir_idx = torch.zeros(n, dtype=torch.long, device=dev)
    for j, (ti, di, si) in enumerate(chunk):
        branch_t[j + 1] = times[ti]
        scale[j + 1] = scales[si]
        dir_idx[j + 1] = di
    if dirs is not None:
        delta = dirs[dir_idx] * scale[:, None]  # state / action kinds: fixed per env
    t_end = max(times[ti] for ti, _, _ in chunk) + window
    times_set = set(times)

    cmd.forced_frame[:] = cmd.library.clip_start[clip]
    obs, _ = env.reset()
    fail_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    finished = torch.zeros(n, dtype=torch.bool, device=dev)
    hist_s, hist_a = [], []
    wm_err: dict[int, torch.Tensor] = {}
    drift = 0.0
    for step in range(t_end):
        s = io.read()
        nominal = not bool(finished[0])
        if nominal:
            hist_s.append(s[0].clone())
            if step in times_set and step >= wm_k:
                sp = hist_s[step - wm_k][None]
                for j in range(wm_k):
                    sp = wm.step(sp, hist_a[step - wm_k + j][None], xi[None])
                wm_err[step] = wm.spec.error_vector(sp, s[0:1])[0] * mask
            same = (branch_t > step) | (branch_t < 0)
            same[0] = False
            same &= ~finished
            if same.any():
                dev_norm = wm.spec.error_vector(s[same], s[0:1].expand(int(same.sum()), -1)).norm(dim=-1)
                drift = max(drift, float(dev_norm.max()))
                io.sync(same.nonzero().squeeze(-1), 0)
        ids = ((branch_t == step) & ~finished).nonzero().squeeze(-1)
        if kind != "action" and ids.numel() > 0:
            if kind == "state":
                d = delta[ids]
            else:  # wm: the model's own error direction and size at this state, scaled
                e = wm_err.get(step)
                d = scale[ids, None] * (e[None] if e is not None else torch.full((1, 70), float("nan"), device=dev))
            if torch.isfinite(d).all():
                io.write(ids, apply_state_delta(s[0:1].expand(ids.numel(), -1), d, err_scale, limits))
        act = policy(obs["policy"], obs["goal"])
        if nominal:
            hist_a.append(act[0].clone())  # noise-free protocol: the executed action is the policy action
        if kind == "action":
            on = (branch_t >= 0) & (step >= branch_t) & (step < branch_t + action_steps)
            if on.any():
                act = act + torch.where(on[:, None], delta, torch.zeros_like(delta))
        obs, _, term, tout, _ = env.step(act)
        newly = (term | tout) & ~finished
        fail_step = torch.where(newly & term, torch.full_like(fail_step, step), fail_step)
        finished |= newly

    fs_all = fail_step.tolist()
    out = []
    for j, (ti, di, si) in enumerate(chunk):
        bt, fs = times[ti], fs_all[j + 1]
        valid = not (0 <= fs < bt)  # failed before its branch time: the nominal run already failed there
        out.append((valid, valid and bt <= fs < bt + window))
    return out, {t: e.norm().item() for t, e in wm_err.items()}, drift, fs_all[0]


def margin_stats(times_rows: list[dict]) -> dict:
    r = [x["margin_min"] for x in times_rows if x["margin_min"] is not None]
    exceeds = [x["model_error_exceeds_margin"] for x in times_rows if x["model_error_exceeds_margin"] is not None]
    finite = [v for v in r if math.isfinite(v)]
    return {
        "branch_times": len(times_rows),
        "median_margin_min": float(np.median(r)) if r else math.nan,
        "frac_times_with_failure_in_grid": len(finite) / max(len(r), 1),
        "frac_model_error_exceeds_margin": float(np.mean(exceeds)) if exceeds else math.nan,
    }
