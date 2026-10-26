"""Benchmark evaluation (paper Appendix F.1 / plan 5.1).

Every clip is played once from its first frame with the deterministic policy:
no observation/action noise, no reset noise, no domain randomization (foot friction 1.0),
deterministic physics. A clip *fails* if the training termination (pelvis height /
tilt vs reference) fires before the clip ends. Normalised return = sum of per-step
rewards / (max step reward * number of steps).

Also reported: the anchor-aligned body position error (``eval/metrics.py``), which
separates local tracking quality from global xy / yaw drift (EXPERIMENTS, drift diagnostic).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.eval.metrics import anchor_aligned_body_error
from supermimic.policy.nets import Policy, PolicyCfg


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float
    failed: bool
    fail_step: int  # -1 if not failed
    terms: dict | None = None  # per-term mean reward per step (dt-scaled)
    mpjpe_aligned: float = float("nan")  # mean anchor-aligned body position error [m] over the executed steps


def load_policy_from_checkpoint(path: str | Path, device: str) -> tuple[Policy, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    pc = ck["config"]["policy"]
    pcfg = PolicyCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pc.items()})
    policy = Policy(pcfg).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    return policy, ck["config"]


@torch.no_grad()
def run_benchmark(policy: Policy, npz_files: list[str], device: str = "cuda:0", num_envs: int | None = None, future_frames: int = 10, verbose: bool = True, anchor_obs: bool | str = False) -> list[ClipResult]:
    num_envs = min(num_envs or 1024, len(npz_files))
    args = SuperMimicEnvCfgArgs(
        npz_files=npz_files,
        num_envs=num_envs,
        future_frames=future_frames,
        episode_length_s=1.0e5,  # clip end decides
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_obs,
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        n_clips = lib.num_clips
        max_r = 0.1  # paper reward upper bound per step (dt * 5)

        # Clip queue: env e plays clips e, e+N, e+2N, ...
        current = torch.arange(num_envs, device=device)
        nxt = current + num_envs
        cmd.forced_frame[:] = lib.clip_start[current]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[torch.clamp(nxt, max=n_clips - 1)]
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        ret_sum = torch.zeros(num_envs, device=device)
        steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        term_names = list(env.reward_manager.active_terms)
        term_sum = torch.zeros(num_envs, len(term_names), device=device)
        mpjpe_sum = torch.zeros(num_envs, device=device)
        mpjpe_n = torch.zeros(num_envs, dtype=torch.long, device=device)
        ai = cmd.motion_anchor_body_index
        origins = env.scene.env_origins
        results: dict[int, ClipResult] = {}
        remaining = n_clips
        while remaining > 0:
            target = cmd.frame.clone()  # the reward pairs the post-step robot state with this target frame
            act = policy(obs["policy"], obs["goal"])
            obs, rew, term, tout, _ = env.step(act)
            ret_sum += rew
            term_sum += env.reward_manager.term_rewards
            steps += 1
            # The done step is skipped: its robot state has already been auto-reset.
            ok = active & ~(term | tout)
            if ok.any():
                rb = lib.bodies(target)
                err = anchor_aligned_body_error(
                    cmd.robot_body_pos_w, cmd.robot_anchor_pos_w, cmd.robot_anchor_quat_w,
                    rb.pos_w + origins[:, None, :], rb.pos_w[:, ai] + origins, rb.quat_w[:, ai],
                )
                mpjpe_sum += torch.where(ok, err, torch.zeros_like(err))
                mpjpe_n += ok.long()
            done = (term | tout) & active
            if done.any():
                ids = done.nonzero().squeeze(-1)
                for e in ids.tolist():
                    ci = int(current[e])
                    if ci < n_clips and ci not in results:
                        st = int(steps[e])
                        results[ci] = ClipResult(
                            name=lib.clips[ci].name,
                            frames=lib.clips[ci].length,
                            steps=st,
                            norm_return=float(ret_sum[e]) / (max_r * max(st, 1)),
                            failed=bool(term[e]),
                            fail_step=st if bool(term[e]) else -1,
                            terms={n: float(term_sum[e, i]) / max(st, 1) for i, n in enumerate(term_names)},
                            mpjpe_aligned=float(mpjpe_sum[e]) / int(mpjpe_n[e]) if int(mpjpe_n[e]) > 0 else float("nan"),
                        )
                        remaining -= 1
                        if verbose and (len(results) % 50 == 0 or len(results) == n_clips):
                            print(f"  evaluated {len(results)}/{n_clips}")
                # advance the queue for these envs (the auto-reset already used forced_frame)
                current[ids] = nxt[ids]
                nxt[ids] = nxt[ids] + num_envs
                finished = current[ids] >= n_clips
                active[ids[finished]] = False
                cmd.forced_frame[ids] = lib.clip_start[torch.clamp(nxt[ids], max=n_clips - 1)]
                ret_sum[ids] = 0.0
                term_sum[ids] = 0.0
                mpjpe_sum[ids] = 0.0
                mpjpe_n[ids] = 0
                steps[ids] = 0
        return [results[i] for i in range(n_clips)]
    finally:
        env.close()


def summarize(results: list[ClipResult]) -> dict:
    n = len(results)
    failed = sum(r.failed for r in results)
    mean_ret = sum(r.norm_return for r in results) / max(n, 1)
    # Length-weighted return (paper Fig. 3a uses the benchmark aggregate; weighting by steps).
    tot_steps = sum(r.steps for r in results)
    w_ret = sum(r.norm_return * r.steps for r in results) / max(tot_steps, 1)
    out = {"clips": n, "failed": failed, "mean_norm_return": mean_ret, "step_weighted_norm_return": w_ret}
    with_err = [r for r in results if r.mpjpe_aligned == r.mpjpe_aligned]  # not NaN
    if with_err:
        out["mean_mpjpe_aligned_m"] = sum(r.mpjpe_aligned for r in with_err) / len(with_err)
        out["step_weighted_mpjpe_aligned_m"] = sum(r.mpjpe_aligned * r.steps for r in with_err) / max(sum(r.steps for r in with_err), 1)
    if results and results[0].terms:
        out["terms"] = {k: sum(r.terms[k] * r.steps for r in results) / max(tot_steps, 1) for k in results[0].terms}
    return out


def save_results(results: list[ClipResult], path: str | Path) -> None:
    Path(path).write_text(json.dumps({"summary": summarize(results), "clips": [asdict(r) for r in results]}, indent=1))
