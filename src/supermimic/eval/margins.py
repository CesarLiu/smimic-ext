"""Branching runs for counterfactual robustness margins (driven by scripts/eval_margins.py).

Duck-typed on the environment (``num_envs``, ``device``, ``reset``, ``step``), the motion command
(``forced_frame``, ``library.clip_start``), the state I/O (``read`` / ``write`` / ``sync``) and the world
model (``step``, ``spec.error_vector``), so it carries no simulator imports and can be tested with stand-ins.

Envs that have not branched yet are copied from the control (env 0) at every step. Identical inputs do
not keep parallel envs identical over long clips (float rounding differs per env origin, and tracking is
chaotic over thousands of steps), so without the copy a branch would start from a state that differs
from the nominal one by more than its perturbation. The copy covers the simulator state only, not the
observation history the policy acts on, so copied envs also take the control's action (otherwise the
one-step drift is ~5e-3 typical, ~1 at its worst, on L4). The reported drift is the largest one-step
divergence before the copy, in normalised error units, the same units as the perturbation sizes.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from supermimic.eval.perturb import perturb_state

# One-step drift levels counted per synced env-step (normalised units): "typical" drift should stay below
# DRIFT_TOL; above DRIFT_EVENT (the smallest state perturbation) it could stand in for a perturbation.
DRIFT_TOL, DRIFT_EVENT = 0.01, 0.25


def branch_times(length: int, first: int, every: int, window: int, max_times: int) -> list[int]:
    """Branch frames from ``first`` on, every ``every`` frames, leaving a full window before the clip end."""
    times = list(range(first, length - window - 1, every))
    if len(times) > max_times:
        idx = np.unique(np.linspace(0, len(times) - 1, max_times).round().astype(int))
        times = [times[i] for i in idx]
    return times


def run_branch_pass(env, cmd, io, policy, wm, xi, clip, times, chunk, kind, scales, dirs, mask, spec, limits, window, action_steps, wm_k, adv_fn=None, adv_dirs=None, drift_counts=None):
    """One synchronous play of ``clip`` with the branches in ``chunk`` ((time idx, dir idx, scale idx) per env 1..).

    ``mask`` (err_dim,) selects the perturbed error components; ``spec`` is the robot's ``StateSpecBase``
    (or the G1 ``err_scale`` tensor) and turns error-space perturbations into states.

    ``drift_counts`` ({"env_steps", "over_tol", "events"}, optional) accumulates the synced env-steps and
    how many of them drifted by more than DRIFT_TOL / DRIFT_EVENT.

    ``kind == "adv"``: ``adv_fn(step, hist_states, hist_actions)`` gives the unit direction at a branch time
    from the nominal history (states up to and including ``step``, actions before it). Directions are cached
    in ``adv_dirs`` ({step: (err_dim,)}), which the caller shares across the passes of one clip.

    Returns ([(valid, failed, steps from branch to failure or -1)] per spec, {t: wm error norm}, drift,
    nominal fail step).
    """
    n, dev = env.num_envs, env.device
    err_dim = mask.numel()
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
        same = None
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
                if drift_counts is not None:
                    drift_counts["env_steps"] += dev_norm.numel()
                    drift_counts["over_tol"] += int((dev_norm > DRIFT_TOL).sum())
                    drift_counts["events"] += int((dev_norm > DRIFT_EVENT).sum())
                io.sync(same.nonzero().squeeze(-1), 0)
        ids = ((branch_t == step) & ~finished).nonzero().squeeze(-1)
        if kind != "action" and ids.numel() > 0:
            if kind == "state":
                d = delta[ids]
            elif kind == "wm":  # the model's own error direction and size at this state, scaled
                e = wm_err.get(step)
                d = scale[ids, None] * (e[None] if e is not None else torch.full((1, err_dim), float("nan"), device=dev))
            else:  # adv: the model's steepest-descent direction of the imagined return at this state
                if step not in adv_dirs:
                    adv_dirs[step] = adv_fn(step, hist_s, hist_a) if nominal else torch.full((err_dim,), float("nan"), device=dev)
                d = scale[ids, None] * adv_dirs[step][None]
            if torch.isfinite(d).all():
                io.write(ids, perturb_state(s[0:1].expand(ids.numel(), -1), d, spec, limits))
        act = policy(obs["policy"], obs["goal"])
        if same is not None and same.any():  # their observations predate the copy: act as the control does
            act = torch.where(same[:, None], act[0:1], act)
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
        failed = valid and bt <= fs < bt + window
        out.append((valid, failed, fs - bt if failed else -1))  # the offset lets callers re-cut shorter windows
    return out, {t: e.norm().item() for t, e in wm_err.items()}, drift, fs_all[0]


def calibration_stats(times_rows: list[dict]) -> dict:
    """``adv`` rows: the model's predicted margin against the simulator margin over the same horizon.

    Agreement = both find no failure in the grid, or both do and are within a factor of 2. ``missed``: the
    simulator fails but the model predicts no failure (the dangerous case); ``false_alarm``: the reverse.
    """
    pairs = [(x["model_margin"], x["sim_margin_model_horizon"]) for x in times_rows if x.get("model_margin") is not None and x.get("sim_margin_model_horizon") is not None]
    n = max(len(pairs), 1)
    fin = math.isfinite
    log_ratio = [math.log2(m / s) for m, s in pairs if fin(m) and fin(s)]
    agree = sum((not fin(m) and not fin(s)) or (fin(m) and fin(s) and abs(math.log2(m / s)) <= 1.0) for m, s in pairs)
    return {
        "times": len(pairs),
        "sim_fails_in_grid": sum(fin(s) for _, s in pairs) / n,
        "model_fails_in_grid": sum(fin(m) for m, _ in pairs) / n,
        "agree": agree / n,
        "missed": sum(fin(s) and not fin(m) for m, s in pairs) / n,
        "false_alarm": sum(fin(m) and not fin(s) for m, s in pairs) / n,
        "median_log2_model_over_sim": float(np.median(log_ratio)) if log_ratio else math.nan,
    }


def compare_directions(random_rows: dict, adv_rows: dict) -> dict:
    """Simulator margin along the model's worst-case direction against the worst of the random directions.

    Both map (clip, frame) -> ``margin_min``; only keys present in both with a margin count. A grid without
    failure (inf) is clipped to ``cap`` (twice the largest finite margin seen) so ratios stay defined.
    """
    keys = [k for k in adv_rows if k in random_rows and adv_rows[k] is not None and random_rows[k] is not None]
    if not keys:
        return {"times": 0}
    finite = [v for k in keys for v in (adv_rows[k], random_rows[k]) if math.isfinite(v)]
    cap = 2.0 * max(finite) if finite else 1.0
    capped = lambda v: v if math.isfinite(v) else cap  # noqa: E731
    lr = np.array([math.log2(capped(adv_rows[k]) / capped(random_rows[k])) for k in keys])
    return {
        "times": len(keys),
        "median_log2_adv_over_random": float(np.median(lr)),
        "frac_adv_smaller": float((lr < 0).mean()),
        "frac_adv_at_most_half": float((lr <= -1).mean()),
        "frac_adv_larger": float((lr > 0).mean()),
    }


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
