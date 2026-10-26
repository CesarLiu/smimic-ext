"""World-model prediction error on the states a policy visits on a motion set (EXPERIMENTS, g-series).

The paper explains SuperMimic's scaling with one world model of the shared body dynamics. If that holds,
the model's error on held-out motions matches its error on training motions, and a held-out failure comes
from the policy, not the model. This script runs the checkpoint's policy in the simulator on the given
clips, records transitions as training does, and reports the k-step open-loop error of the checkpoint's
world model on those windows: the normalised metric of the training log's ``wm/online_nse_*step``,
overall, per category and per clip. Run it on train_eval.txt and on the held-out lists with the same
protocol and compare.

  uv run python scripts/eval_wm_heldout.py --ckpt logs/g1/ckpt_latest.pt --npz-dir data/npz/bones_seed \
      --clip-list data/splits/g1/heldout_cat.txt --categories data/splits/g1/clip_categories.csv

``--protocol train`` (default) uses the checkpoint's training noise / domain randomisation and starts
episodes uniformly over frames (the distribution the model was fit on); ``eval`` uses the benchmark
protocol (no noise or randomisation, episodes from clip starts).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

import torch
import tyro

from supermimic.data.motion_convert import load_g1_model
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def _tup(v):
    return tuple(v) if isinstance(v, list) else v


def world_model_cfg(d: dict) -> WorldModelCfg:
    """WorldModelCfg from a checkpoint's ``config["world_model"]`` (asdict output)."""
    spec = StateSpecCfg(**{k: _tup(v) for k, v in d["state_spec"].items()})
    return WorldModelCfg(**{k: _tup(v) for k, v in d.items() if k != "state_spec"}, state_spec=spec)


@torch.no_grad()
def main(
    ckpt: Path,
    npz_dir: Path = Path("data/npz/bones_seed"),
    clips: list[str] | None = None,
    clip_list: Path | None = None,
    categories: Path | None = None,
    protocol: Literal["train", "eval"] = "train",
    num_envs: int = 1024,
    steps: int = 500,
    k: int = 8,
    max_windows: int = 200_000,
    batch: int = 8192,
    seed: int = 0,
    device: str = "cuda:0",
    out: Path | None = None,
) -> None:
    if clip_list is not None:
        clips = [l.strip() for l in open(clip_list) if l.strip()]
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    if not files:
        raise SystemExit(f"no clips found under {npz_dir}")
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    ec = cfg["env"]
    train_like = protocol == "train"
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=files,
            num_envs=num_envs,
            future_frames=ec["future_frames"],
            episode_length_s=ec["episode_length_s"] if train_like else 1.0e5,
            action_noise_std=ec["action_noise_std"] if train_like else 0.0,
            obs_noise=ec["obs_noise"] if train_like else False,
            reset_noise=ec["reset_noise"] if train_like else False,
            domain_randomization=ec["domain_randomization"] if train_like else False,
            sampling_mode="uniform" if train_like else "start",
            replay_capacity=2 * steps + 2,  # an episode end writes two rows (last transition + terminal state): no ring wrap
            record_transitions=True,
            anchor_obs=ec.get("anchor_obs", False),
            seed=seed,
        ),
        device=device,
    )
    try:
        robot = env.scene["robot"]
        heads = AnalyticHeads(G1Kinematics(load_g1_model(), device=device), TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        wm = WorldModel(world_model_cfg(cfg["world_model"]), heads, device)
        wm.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["wm"])
        wm.eval()
        lib = env.command_manager.get_term("motion").library
        buf = env.recorder_manager.get_term("replay").buffer

        obs, _ = env.reset()
        for _ in range(steps):
            obs, *_ = env.step(policy(obs["policy"], obs["goal"]))

        e, t = buf.window_mask(k).nonzero(as_tuple=True)
        if e.numel() == 0:
            raise SystemExit("no complete k-step windows were recorded; increase --steps")
        if e.numel() > max_windows:
            g = torch.Generator(device="cpu").manual_seed(seed)
            keep = torch.randperm(e.numel(), generator=g)[:max_windows].to(e.device)
            e, t = e[keep], t[keep]
        nwin = e.numel()
        offs = torch.arange(k + 1, device=e.device)
        per_step = torch.zeros(k, device=device)
        clip_n = torch.zeros(lib.num_clips, device=device)
        clip_e1 = torch.zeros(lib.num_clips, device=device)
        clip_ek = torch.zeros(lib.num_clips, device=device)
        for i in range(0, nwin, batch):
            eb, tb = e[i : i + batch], t[i : i + batch]
            tt = (tb[None, :] + offs[:, None]) % buf.T
            ee = eb[None, :].expand_as(tt)
            states, a_exec, xi, clip = buf.s[ee, tt], buf.a_exec[ee[:-1], tt[:-1]], buf.xi[eb, tb], buf.clip[eb, tb]
            s = states[0]
            errs = []
            for j in range(k):
                s = wm.step(s, a_exec[j], xi)
                errs.append(wm.spec.normalized_sq_error(s, states[j + 1]))
            err = torch.stack(errs)  # (k, n)
            per_step += err.sum(1)
            clip_n.index_add_(0, clip, torch.ones_like(err[0]))
            clip_e1.index_add_(0, clip, err[0])
            clip_ek.index_add_(0, clip, err[-1])

        cat_of: dict[str, str] = {}
        if categories is not None:
            with open(categories) as f:
                cat_of = {r["name"]: r["category"] for r in csv.DictReader(f)}
        rows = []
        by_cat: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        for ci in range(lib.num_clips):
            n = float(clip_n[ci])
            if n == 0:
                continue
            name = lib.clips[ci].name
            c = cat_of.get(name, "<uncategorised>")
            rows.append({"name": name, "category": c, "windows": int(n), "nse_1step": float(clip_e1[ci]) / n, f"nse_{k}step": float(clip_ek[ci]) / n})
            acc = by_cat[c]
            acc[0] += n
            acc[1] += float(clip_e1[ci])
            acc[2] += float(clip_ek[ci])
        curve = (per_step / nwin).tolist()
        summary = {
            "ckpt": str(ckpt),
            "protocol": protocol,
            "clips": len(files),
            "clips_with_windows": len(rows),
            "windows": nwin,
            "k": k,
            "nse_per_step": curve,
            "nse_1step": curve[0],
            f"nse_{k}step": curve[-1],
            "by_category": {c: {"windows": int(v[0]), "nse_1step": v[1] / v[0], f"nse_{k}step": v[2] / v[0]} for c, v in sorted(by_cat.items())},
        }
        print(f"{len(rows)}/{len(files)} clips with windows, {nwin} windows ({protocol} protocol)")
        print(f"normalised error: 1-step {curve[0]:.5f} | {k}-step {curve[-1]:.5f}")
        if cat_of:
            print(f"{'category':40s} {'windows':>8s} {'1-step':>9s} {f'{k}-step':>9s}")
            for c, v in sorted(summary["by_category"].items(), key=lambda kv: -kv[1][f"nse_{k}step"]):
                print(f"{c[:40]:40s} {v['windows']:8d} {v['nse_1step']:9.5f} {v[f'nse_{k}step']:9.5f}")
        out = out or ckpt.parent / f"wm_error_{ckpt.stem}_{(clip_list.stem if clip_list else 'clips')}_{protocol}.json"
        Path(out).write_text(json.dumps({"summary": summary, "clips": rows}, indent=1))
        print("saved", out)
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
