"""Counterfactual robustness margins along motions (EXPERIMENTS, g-series).

If the state (or action) at time t were slightly different, would the motion still be tracked? At branch
times along each clip the state is perturbed by a grid of sizes in several directions, the same policy
continues in the simulator, and a branch *fails* if the training termination fires within ``--window``
steps. The margin r(t) is the smallest perturbation size that fails. SuperMimic's policy learns on
world-model predictions, so the comparison that matters is r(t) against the world model's own k-step
prediction error e(t) at the same state: where e(t) >= r(t), the model's error alone is enough to break
the policy.

  uv run python scripts/eval_margins.py --ckpt logs/g1/ckpt_latest.pt --npz-dir data/npz/bones_seed \
      --clip-list data/splits/g1/heldout_cat.txt --categories data/splits/g1/clip_categories.csv --kind state

Kinds (all sizes in the world model's normalised error units, see ``eval/perturb.py``, except ``action``):
  * ``state``  : random unit directions in the ``--components`` subspace, sizes ``--scales``;
  * ``wm``     : the world model's actual k-step error at that state, scaled by ``--scales`` (1 = exactly
                 the model's prediction): a margin below 1 means the model's own error breaks the policy;
  * ``action`` : random action offsets (policy action units) held for ``--action-steps`` steps.

How branching works: all envs play the same clip from its first frame with the deterministic benchmark
protocol (no noise, no randomisation), and until its branch time every env is copied from the control
env (env 0, never perturbed) at each step, so a branch starts from the nominal state plus its
perturbation. Copying is needed because parallel envs do not stay identical over long clips on their
own (a first run on 240 s LAFAN1 clips drifted apart by 52 in raw state units). "Determinism drift" is
the largest one-step divergence before the copy, as a normalised error norm (the same units as
``--scales``); it should be far below the smallest nonzero scale. A perturbation is written to the
simulator at the start of step t and the policy sees it from step t+1 on (a disturbance, as in a push
test). Scale 0 is always included: those branches only rewrite the unchanged state, so their failure rate
measures how much the write itself disturbs the simulation.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.sim_context import SimContext
from supermimic.envs.state_io import StateIO
from supermimic.envs.tracking_env_cfg import FOOT_GEOM_REGEX
from supermimic.eval.benchmark import load_policy_from_checkpoint, load_world_model_from_checkpoint
from supermimic.eval.margins import branch_times, margin_stats, run_branch_pass
from supermimic.eval.perturb import component_mask, margins_from_outcomes, random_directions

# The first LAFAN1 run failed only at the top of the old grid (median r_min 16), so the grid now goes to
# 64 with half steps above 8. The wm kind is in multiples of the model's own error (8-step norm ~1).
DEFAULT_SCALES = {
    "state": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0],
    "wm": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0],
    "action": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
}


@torch.no_grad()
def main(
    ckpt: Path,
    npz_dir: Path = Path("data/npz/bones_seed"),
    clips: list[str] | None = None,
    clip_list: Path | None = None,
    categories: Path | None = None,
    kind: Literal["state", "wm", "action"] = "state",
    components: Literal["vel", "pos", "all"] = "vel",
    scales: list[float] | None = None,
    num_dirs: int = 8,
    branch_every: int = 25,
    window: int = 100,
    max_times: int = 40,
    action_steps: int = 1,
    wm_k: int = 8,
    max_clips: int | None = 200,
    num_envs: int = 2048,
    seed: int = 0,
    device: str = "cuda:0",
    out: Path | None = None,
) -> None:
    """``components``: which state parts a ``state`` / ``wm`` perturbation touches. ``vel`` (default) avoids
    pushing feet into the ground, whose contact resolution adds a kick of its own; ``all`` includes poses."""
    if clip_list is not None:
        clips = [l.strip() for l in open(clip_list) if l.strip()]
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    if max_clips is not None and len(files) > max_clips:
        files = sorted(random.Random(seed).sample(files, max_clips))
    if not files:
        raise SystemExit(f"no clips found under {npz_dir}")
    scales = sorted(set([0.0] + list(scales or DEFAULT_SCALES[kind])))
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=files, num_envs=num_envs, future_frames=cfg["env"]["future_frames"], episode_length_s=1.0e5,
            action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False,
            sampling_mode="start", record_transitions=False, anchor_obs=cfg["env"].get("anchor_obs", False), seed=seed,
        ),
        device=device,
    )
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        io = StateIO(env)
        xi = SimContext(env, FOOT_GEOM_REGEX).read()[0]  # benchmark protocol: the same in every env
        wm = load_world_model_from_checkpoint(ckpt, env, device)
        limits = env.scene["robot"].data.soft_joint_pos_limits[0]
        mask = component_mask(components, device)
        n_dirs = 1 if kind == "wm" else num_dirs
        dirs = None
        if kind == "state":
            dirs = random_directions(num_dirs, 70, mask, seed).to(device)
        elif kind == "action":
            dirs = random_directions(num_dirs, 29, None, seed).to(device)
        cat_of: dict[str, str] = {}
        if categories is not None:
            with open(categories) as f:
                cat_of = {r["name"]: r["category"] for r in csv.DictReader(f)}

        clip_rows, all_times, by_cat = [], [], defaultdict(list)
        zero_fail, zero_n, nonmono, nonmono_n, max_drift = 0, 0, 0, 0, 0.0
        for ci in range(lib.num_clips):
            name = lib.clips[ci].name
            times = branch_times(int(lib.clip_len[ci]), wm_k, branch_every, window, max_times)
            if not times:
                clip_rows.append({"name": name, "skipped": "shorter than the branch window"})
                continue
            specs = [(ti, di, si) for ti in range(len(times)) for di in range(n_dirs) for si in range(len(scales))]
            failed = np.zeros((len(times), n_dirs, len(scales)), dtype=bool)
            valid = np.ones_like(failed)
            err_norm: dict[int, float] = {}
            nominal_fail = -1
            for start in range(0, len(specs), num_envs - 1):
                chunk = specs[start : start + num_envs - 1]
                res, en, drift, nf = run_branch_pass(env, cmd, io, policy, wm, xi, ci, times, chunk, kind, scales, dirs, mask, wm.spec.err_scale, limits, window, action_steps, wm_k)
                err_norm.update(en)
                max_drift = max(max_drift, drift)
                nominal_fail = nf
                for (ti, di, si), (v, fl) in zip(chunk, res):
                    valid[ti, di, si], failed[ti, di, si] = v, fl
            m = margins_from_outcomes(failed, scales)
            zero_fail += int((failed[..., 0] & valid[..., 0]).sum())
            zero_n += int(valid[..., 0].sum())
            ok = valid.all(-1)
            nonmono += int((m["nonmonotonic"] & ok).sum())
            nonmono_n += int(ok.sum())
            rows = []
            for ti, t in enumerate(times):
                e = err_norm.get(t)
                if not ok[ti].any():
                    rows.append({"frame": t, "margin_min": None, "margin_median": None, "wm_error_norm": e, "model_error_exceeds_margin": None})
                    continue
                mg = m["margin"][ti][ok[ti]]
                if kind == "wm" and e is not None:
                    mg = mg * e  # lambda -> norm units
                mmin = float(mg.min())
                exceeds = None if e is None else (bool(e >= mmin) if kind != "action" else None)
                rows.append({"frame": t, "margin_min": mmin, "margin_median": float(np.median(mg)), "wm_error_norm": e, "model_error_exceeds_margin": exceeds})
            cat = cat_of.get(name, "<uncategorised>")
            st = margin_stats(rows)
            clip_rows.append({"name": name, "category": cat, "frames": int(lib.clip_len[ci]), "nominal_fail_step": nominal_fail, **st, "times": rows,
                              "failed": failed.tolist(), "valid": valid.tolist()})
            all_times += rows
            by_cat[cat] += rows
            print(f"[{ci + 1}/{lib.num_clips}] {name}: {len(times)} times | median r_min {st['median_margin_min']:.3g} | model error >= margin at {100 * st['frac_model_error_exceeds_margin']:.0f}% of times" + (f" | nominal FAIL@{nominal_fail}" if nominal_fail >= 0 else ""), flush=True)

        summary = {
            "ckpt": str(ckpt), "kind": kind, "components": components, "scales": scales, "num_dirs": n_dirs,
            "window_steps": window, "branch_every": branch_every, "wm_k": wm_k, "clips": len(files),
            **margin_stats(all_times),
            "zero_scale_failure_rate": zero_fail / max(zero_n, 1),
            "nonmonotonic_rate": nonmono / max(nonmono_n, 1),
            "max_determinism_drift": max_drift,
            "by_category": {c: margin_stats(v) for c, v in sorted(by_cat.items())},
        }
        print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}, indent=1))
        if max_drift > 0.01:
            print(f"[warn] unbranched envs diverged from the control by {max_drift:.2e} in one step (normalised): compare with the smallest scale {min(x for x in scales if x > 0):g}")
        if summary["zero_scale_failure_rate"] > 0.01:
            print(f"[warn] {100 * summary['zero_scale_failure_rate']:.1f}% of scale-0 branches failed: rewriting the state disturbs the simulation")
        if cat_of:
            print(f"{'category':40s} {'times':>6s} {'median r':>9s} {'e>=r %':>7s}")
            for c, s in sorted(summary["by_category"].items(), key=lambda kv: -(kv[1]["frac_model_error_exceeds_margin"] or 0)):
                print(f"{c[:40]:40s} {s['branch_times']:6d} {s['median_margin_min']:9.3g} {100 * s['frac_model_error_exceeds_margin']:7.1f}")
        out = out or ckpt.parent / f"margins_{ckpt.stem}_{(clip_list.stem if clip_list else 'clips')}_{kind}.json"
        Path(out).write_text(json.dumps({"summary": summary, "clips": clip_rows}, indent=1))
        print("saved", out)
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
