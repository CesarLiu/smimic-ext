"""CPU fixtures for the return-based reset sampler regression (no simulator).

``build_library`` writes three synthetic clips as npz and loads them with the real ``MotionLibrary``
(CPU, full mode). ``run_all`` drives the motion command's sampler on a scripted episode sequence —
per-step reward logging, episode finalisation with terminations, truncations and clip ends, and
resets drawn from the sampler — under four configurations, and returns everything the sampler keeps
or produces. ``tests/data/make_sampler_golden.py`` records it; ``tests/test_task_interface.py``
compares the current code against the recording.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand, MultiClipMotionCommandCfg
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, TRACKED_BODIES

CLIP_LENGTHS = (130, 75, 260)
BIN = 50
N_ENVS = 16
STEPS = 150


def write_clips(folder: Path) -> list[Path]:
    rng = np.random.default_rng(0)
    names = ["pelvis"] + [b for b in TRACKED_BODIES if b != "pelvis"]
    nb = len(names)
    files = []
    for i, n in enumerate(CLIP_LENGTHS):
        q = rng.normal(size=(n, nb, 4))
        f = folder / f"clip{i}.npz"
        np.savez(
            f,
            body_names=np.array(names),
            joint_pos=rng.normal(size=(n, 29)) * 0.1,
            joint_vel=rng.normal(size=(n, 29)),
            body_pos_w=rng.normal(size=(n, nb, 3)),
            body_quat_w=q / np.linalg.norm(q, axis=-1, keepdims=True),
            body_lin_vel_w=rng.normal(size=(n, nb, 3)),
            body_ang_vel_w=rng.normal(size=(n, nb, 3)),
        )
        files.append(f)
    return files


def build_library() -> MotionLibrary:
    with tempfile.TemporaryDirectory() as d:
        files = write_clips(Path(d))
        return MotionLibrary(files, TRACKED_BODIES, torch.zeros(29), device="cpu", bin_frames=BIN)


class StubEnv:
    def __init__(self):
        self.num_envs = N_ENVS
        self.device = "cpu"
        self.cfg = SimpleNamespace(episode_length_s=1.0)
        self.step_dt = 0.02
        self.reward_buf = torch.zeros(N_ENVS)
        self.termination_manager = SimpleNamespace(terminated=torch.zeros(N_ENVS, dtype=torch.bool))


def make_command(lib: MotionLibrary, cfg: MultiClipMotionCommandCfg, env: StubEnv) -> MultiClipMotionCommand:
    """The sampler-relevant part of ``MultiClipMotionCommand.__init__`` without a simulator."""
    cmd = object.__new__(MultiClipMotionCommand)
    cmd.cfg = cfg
    cmd._env = env
    cmd.metrics = {}
    cmd.library = lib
    n = env.num_envs
    cmd.frame = torch.zeros(n, dtype=torch.long)
    cmd.clip_end = torch.zeros(n, dtype=torch.bool)
    cmd._init_reset_sampler(math.ceil(env.cfg.episode_length_s / env.step_dt) + 2)
    for k in ("sampling_entropy", "episode_return_to_go"):
        cmd.metrics[k] = torch.zeros(n)
    return cmd


CONFIGS = {
    "return_const": {"sampling_mode": "return", "gmax_mode": "const"},
    "return_remaining_bootstrap": {"sampling_mode": "return", "gmax_mode": "remaining", "bootstrap": True},
    "uniform": {"sampling_mode": "uniform"},
    "start": {"sampling_mode": "start"},
}


def drive(lib: MotionLibrary, name: str) -> dict:
    opts = dict(CONFIGS[name])
    bootstrap = opts.pop("bootstrap", False)
    cfg = MultiClipMotionCommandCfg(
        npz_files=[], anchor_body_name=ANCHOR_BODY, body_names=TRACKED_BODIES, bin_frames=BIN,
        max_logged_steps=40, min_remaining_steps=10, ema_alpha=0.2, **opts,
    )
    env = StubEnv()
    cmd = make_command(lib, cfg, env)
    if bootstrap:
        cmd.bootstrap_fn = lambda ids: 0.5 + 0.01 * ids.float()
    g = torch.Generator().manual_seed(1)
    torch.manual_seed(2)
    cmd.frame[:] = cmd._sample_frames(N_ENVS)
    sampled, ent = [], []
    for _ in range(STEPS):
        env.reward_buf = torch.rand(N_ENVS, generator=g) * 0.1
        done = torch.rand(N_ENVS, generator=g) < 0.08
        env.termination_manager.terminated = done & (torch.rand(N_ENVS, generator=g) < 0.6)
        # per-step path of _update_command for the envs that were not just reset
        cmd._log_step_rewards(~done)
        last = lib.last_frame_of(cmd.frame)
        cmd.frame += 1
        over = cmd.frame > last
        cmd.clip_end |= over
        cmd.frame = torch.where(over, last, cmd.frame)
        ids = (done | cmd.clip_end).nonzero().squeeze(-1)
        if ids.numel():
            mask = torch.zeros(N_ENVS, dtype=torch.bool)
            mask[ids] = True
            cmd._log_step_rewards(mask)
            cmd._finalize_episodes(ids)
            cmd.ep_len[ids] = 0
            f = cmd._sample_frames(ids.numel())
            cmd.frame[ids] = f
            cmd.clip_end[ids] = False
            sampled.append(f)
            ent.append(cmd.metrics["sampling_entropy"].clone())
    return {
        "bin_return": cmd.bin_return.clone(),
        "bin_memory": cmd.bin_memory.clone(),
        "sampled": torch.cat(sampled),
        "entropy": torch.stack(ent),
        "rtg0": cmd.metrics["episode_return_to_go"].clone(),
        "ep_len": cmd.ep_len.clone(),
        "frame": cmd.frame.clone(),
    }


def run_all() -> dict:
    torch.set_num_threads(1)
    lib = build_library()
    return {name: drive(lib, name) for name in CONFIGS}
