"""Can the world model learn the motions it predicts badly, given data? (offline fine-tuning test)

The checkpoint's policy collects transitions on a clip set with the training protocol (reference-state
initialisation anywhere in a clip, action/observation noise, domain randomisation). The clips of every
category are split in half (mirror pairs on the same side). Copies of the checkpoint's world model are then
fine-tuned with the training loss on windows from the train halves only and scored on the test halves:

  * ``all``      : windows of every category (the hard groups included);
  * ``controls`` : windows of the ``--control-prefix`` categories only, same number of steps (placebo: more
                   steps and adaptation to this policy's states, but no data of the hard groups).

If ``all`` brings the hard groups' test error down to the controls' level and ``controls`` does not, the model
can represent those dynamics and was short of data; if neither does, capacity or the state representation is
the limit.

  uv run python scripts/diag_wm_finetune.py --ckpt logs/e8/e8_boneseed_ckpt_0100000.pt --npz-dir data/npz/bones_diag \
      --clip-list data/splits/diag/clips.txt --categories data/splits/diag/clip_categories.csv --out logs/e8/wm_finetune.json
"""

from __future__ import annotations

import copy
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import tyro

from supermimic.data.replay_buffer import WindowBatch
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint, load_world_model_from_checkpoint
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer


def _windows(buf, e: torch.Tensor, t: torch.Tensor, k: int) -> WindowBatch:
    offs = torch.arange(k + 1, device=e.device)
    tt = (t[None, :] + offs[:, None]) % buf.T
    ee = e[None, :].expand(k + 1, e.numel())
    return WindowBatch(
        states=buf.s[ee, tt], a_exec=buf.a_exec[ee[:-1], tt[:-1]], a_clean=buf.a[ee[:-1], tt[:-1]], xi=buf.xi[e, t],
        reward=buf.r[ee[:-1], tt[:-1]], done=buf.done[ee[1:], tt[1:]], frame=buf.frame[ee[:-1], tt[:-1]],
    )


def main(
    ckpt: Path,
    npz_dir: Path = Path("data/npz/bones_diag"),
    clip_list: Path | None = None,
    categories: Path = Path("data/splits/diag/clip_categories.csv"),
    control_prefix: str = "ctrl_",
    num_envs: int = 2048,
    steps: int = 600,
    k: int = 8,
    train_steps: int = 3000,
    eval_every: int = 500,
    batch_size: int = 2048,
    lr: float = 2e-4,
    eval_windows: int = 8192,
    arms: tuple[str, ...] = ("all", "controls"),
    eval_train_half: bool = False,
    seed: int = 0,
    device: str = "cuda:0",
    out: Path | None = None,
) -> None:
    clips = [l.strip() for l in open(clip_list) if l.strip()] if clip_list is not None else None
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    with open(categories) as f:
        cat_of = {r["name"]: r["category"] for r in csv.DictReader(f)}
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    ec = cfg["env"]
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=files, num_envs=num_envs, future_frames=ec["future_frames"], episode_length_s=ec["episode_length_s"],
            action_noise_std=ec["action_noise_std"], obs_noise=ec["obs_noise"], reset_noise=ec["reset_noise"],
            domain_randomization=ec["domain_randomization"], sampling_mode="uniform", replay_capacity=2 * steps + 2,
            record_transitions=True, anchor_obs=ec.get("anchor_obs", False), seed=seed,
        ),
        device=device,
    )
    try:
        wm0 = load_world_model_from_checkpoint(ckpt, env, device)
        lib = env.command_manager.get_term("motion").library
        buf = env.recorder_manager.get_term("replay").buffer
        with torch.no_grad():
            obs, _ = env.reset()
            for _ in range(steps):
                obs, *_ = env.step(policy(obs["policy"], obs["goal"]))

        # ---------------------------------------------------------------- split clips per category (mirror pairs together)
        names = [c.name for c in lib.clips]
        cats = sorted({cat_of[n] for n in names})
        groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for i, n in enumerate(names):
            groups[cat_of[n]][n[:-2] if n.endswith("_M") else n].append(i)
        rng = random.Random(seed)
        is_test = torch.zeros(lib.num_clips, dtype=torch.bool, device=device)
        for c in cats:
            keys = sorted(groups[c])
            rng.shuffle(keys)
            for key in keys[: len(keys) // 2]:
                is_test[groups[c][key]] = True
        cat_idx = torch.tensor([cats.index(cat_of[n]) for n in names], device=device)
        is_ctrl = torch.tensor([cat_of[n].startswith(control_prefix) for n in names], device=device)

        wmask = buf.window_mask(k)
        row_clip = buf.clip
        train_mask = {"all": wmask & ~is_test[row_clip], "controls": wmask & ~is_test[row_clip] & is_ctrl[row_clip]}
        g = torch.Generator(device="cpu").manual_seed(seed)
        test_sets = {}
        for half, sel in (("", is_test), ("@train", ~is_test)) if eval_train_half else (("", is_test),):
            for ci, c in enumerate(cats):
                e, t = (wmask & sel[row_clip] & (cat_idx[row_clip] == ci)).nonzero(as_tuple=True)
                if e.numel() > eval_windows:
                    keep = torch.randperm(e.numel(), generator=g)[:eval_windows].to(e.device)
                    e, t = e[keep], t[keep]
                test_sets[c + half] = (e, t)  # "@train": windows of the train half (in-sample fit)
        share = {c: float((train_mask["all"] & (cat_idx[row_clip] == ci)).sum()) / float(train_mask["all"].sum()) for ci, c in enumerate(cats)}
        print("train-window share per category:", {c: f"{100 * v:.1f}%" for c, v in share.items()})
        print("test windows per category:", {c: int(v[0].numel()) for c, v in test_sets.items()})

        @torch.no_grad()
        def evaluate(wm) -> dict[str, float]:
            res = {}
            for c, (e, t) in test_sets.items():
                errs = []
                for i in range(0, e.numel(), 4096):
                    w = _windows(buf, e[i : i + 4096], t[i : i + 4096], k)
                    errs.append(wm.rollout_error(w.states, w.a_exec, w.xi)[-1] * w.xi.shape[0])
                res[c] = float(sum(errs) / max(e.numel(), 1))
            return res

        base = evaluate(wm0)
        print("step 0 (checkpoint WM):", {c: round(v, 4) for c, v in base.items()})
        curves = {}
        for arm, mask in ((a, train_mask[a]) for a in arms):
            wm = copy.deepcopy(wm0)
            wm.train()
            trainer = WorldModelTrainer(wm, lib, WMTrainerCfg(window=k, batch_size=batch_size, lr=lr, compile=None))
            curve = [{"step": 0, **base}]
            for step in range(1, train_steps + 1):
                e, t = buf._sample_from_mask(mask, batch_size)
                loss, _ = trainer.loss_on(_windows(buf, e, t, k))
                trainer.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
                trainer.opt.step()
                if step % eval_every == 0:
                    wm.eval()
                    r = evaluate(wm)
                    wm.train()
                    curve.append({"step": step, **r})
                    print(f"[{arm}] step {step}:", {c: round(v, 4) for c, v in r.items()}, flush=True)
            curves[arm] = curve
        summary = {"ckpt": str(ckpt), "k": k, "lr": lr, "batch_size": batch_size, "train_window_share": share, "curves": curves}
        out = out or ckpt.parent / "wm_finetune.json"
        Path(out).write_text(json.dumps(summary, indent=1))
        print("saved", out)
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
