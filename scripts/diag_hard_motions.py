"""Where along a motion does the world model go wrong, and is that where the policy fails?

Every clip is played once from its first frame with the benchmark protocol (deterministic, no noise or
randomisation), one environment per clip, and the transitions of that first episode are recorded. Along
each rollout the checkpoint's world model predicts open loop from the recorded executed actions, giving

  * err1[t] : 1-step normalised error of the prediction of s_t (from s_{t-1});
  * errk[t] : k-step error of the prediction of s_t (from s_{t-k}), also split into the six state groups
              [root pos, root rot, root lin vel, root ang vel, joint pos, joint vel];

next to reference features of the target frame (root angular speed, root height, both feet off the
ground, upside down) and the robot-vs-reference tilt error that the termination tests. Run it on the
groups that fail (flips, side rolls, lying) and on control groups that do not (reach jumps, kneeling):

  uv run python scripts/diag_hard_motions.py --ckpt logs/e8/e8_boneseed_ckpt_0100000.pt \
      --npz-dir data/npz/bones_diag --categories data/splits/diag/clip_categories.csv --out logs/e8/diag_hard.npz
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint, load_world_model_from_checkpoint

GROUPS = ("root_pos", "root_rot", "root_lin_vel", "root_ang_vel", "joint_pos", "joint_vel")
GROUP_SLICES = (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 41), slice(41, 70))


def _quat_gravity_z(q: torch.Tensor) -> torch.Tensor:
    """z-component of the world gravity direction (0,0,-1) in the body frame of quaternion q (wxyz): -1 upright, +1 upside down."""
    w, x, y, z = q.unbind(-1)
    return -(1.0 - 2.0 * (x * x + y * y))  # -R[2,2]


@torch.no_grad()
def main(
    ckpt: Path,
    npz_dir: Path = Path("data/npz/bones_diag"),
    clip_list: Path | None = None,
    categories: Path | None = None,
    k: int = 8,
    batch: int = 16384,
    foot_clear: float = 0.12,
    device: str = "cuda:0",
    out: Path | None = None,
) -> None:
    clips = [l.strip() for l in open(clip_list) if l.strip()] if clip_list is not None else None
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    if not files:
        raise SystemExit(f"no clips found under {npz_dir}")
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    ec = cfg["env"]
    max_len = max(int(np.load(f)["joint_pos"].shape[0]) for f in files)
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=files,
            num_envs=len(files),
            future_frames=ec["future_frames"],
            episode_length_s=1.0e5,
            action_noise_std=0.0,
            obs_noise=False,
            reset_noise=False,
            domain_randomization=False,
            sampling_mode="start",
            replay_capacity=max_len + 4,  # one episode per env, never wraps
            record_transitions=True,
            anchor_obs=ec.get("anchor_obs", False),
            seed=0,
        ),
        device=device,
    )
    try:
        wm = load_world_model_from_checkpoint(ckpt, env, device)
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        buf = env.recorder_manager.get_term("replay").buffer
        n = lib.num_clips
        assert n == len(files) == env.num_envs
        ids = torch.arange(n, device=device)
        cmd.forced_frame[:] = lib.clip_start[ids]
        obs, _ = env.reset()
        ended = torch.zeros(n, dtype=torch.bool, device=device)
        failed = torch.zeros(n, dtype=torch.bool, device=device)
        ep_len = torch.zeros(n, dtype=torch.long, device=device)
        for _ in range(int(lib.clip_len.max()) + 2):
            obs, _, term, tout, _ = env.step(policy(obs["policy"], obs["goal"]))
            live = ~ended
            ep_len += live.long()
            done = (term | tout) & live
            failed |= term & live
            ended |= done
            if bool(ended.all()):
                break

        # ------------------------------------------------------------ open-loop errors along the first episode
        T = buf.T
        mask = buf.window_mask(k) & (buf.ep_id == 0)
        e_idx, t_idx = mask.nonzero(as_tuple=True)
        err1 = torch.full((n, T), float("nan"), device=device)
        errk = torch.full((n, T), float("nan"), device=device)
        errk_g = torch.full((n, T, len(GROUPS)), float("nan"), device=device)
        offs = torch.arange(k + 1, device=device)
        for i in range(0, e_idx.numel(), batch):
            eb, tb = e_idx[i : i + batch], t_idx[i : i + batch]
            tt = tb[None, :] + offs[:, None]
            ee = eb[None, :].expand_as(tt)
            states, a_exec, xi = buf.s[ee, tt], buf.a_exec[ee[:-1], tt[:-1]], buf.xi[eb, tb]
            s = wm.step(states[0], a_exec[0], xi)
            err1[eb, tb + 1] = wm.spec.normalized_sq_error(s, states[1])
            for j in range(1, k):
                s = wm.step(s, a_exec[j], xi)
            ev = wm.spec.error_vector(s, states[k])
            errk[eb, tb + k] = (ev * ev).mean(-1)
            errk_g[eb, tb + k] = torch.stack([(ev[:, sl] ** 2).mean(-1) for sl in GROUP_SLICES], -1)

        # ------------------------------------------------------------ reference features and tilt error per row
        from supermimic.envs.tracking_env_cfg import TRACKED_BODIES

        la, ra = TRACKED_BODIES.index("left_ankle_roll_link"), TRACKED_BODIES.index("right_ankle_roll_link")
        valid = buf.valid & (buf.ep_id == 0)
        frame = torch.where(valid, buf.frame, lib.clip_start[:, None].expand(n, T))
        ref_w, ref_z, ref_gz, feet_z = (torch.empty(n, T, device=device) for _ in range(4))
        for e0 in range(0, n, max(1, 200_000 // T)):  # chunk over envs: tracked-body lookups are (rows, 14, ·)
            fr = frame[e0 : e0 + max(1, 200_000 // T)]
            ref = lib.ref71(fr.reshape(-1))
            ref_w[e0 : e0 + fr.shape[0]] = ref[:, 39:42].norm(dim=-1).reshape(fr.shape)
            ref_z[e0 : e0 + fr.shape[0]] = ref[:, 2].reshape(fr.shape)
            ref_gz[e0 : e0 + fr.shape[0]] = _quat_gravity_z(ref[:, 3:7]).reshape(fr.shape)
            rb = lib.bodies(fr.reshape(-1))
            feet_z[e0 : e0 + fr.shape[0]] = torch.minimum(rb.pos_w[:, la, 2], rb.pos_w[:, ra, 2]).reshape(fr.shape)
        airborne = feet_z > foot_clear
        rob_gz = _quat_gravity_z(buf.s[..., 3:7])
        tilt_err = (rob_gz - ref_gz).abs()  # robot vs its target frame (root body); the termination uses 0.8

        cat_of: dict[str, str] = {}
        if categories is not None:
            with open(categories) as f:
                cat_of = {r["name"]: r["category"] for r in csv.DictReader(f)}
        clip_names = [c.name for c in lib.clips]
        cats = [cat_of.get(nm, "<uncategorised>") for nm in clip_names]

        to = lambda x: x.float().cpu().numpy()  # noqa: E731
        res = dict(
            names=np.array(clip_names), categories=np.array(cats), clip_len=to(lib.clip_len), ep_len=to(ep_len), failed=to(failed),
            valid=to(valid), err1=to(err1), errk=to(errk), errk_groups=to(errk_g), ref_ang_speed=to(ref_w), ref_root_z=to(ref_z),
            ref_gravity_z=to(ref_gz), ref_feet_z=to(feet_z), airborne=to(airborne), tilt_err=to(tilt_err), k=k, groups=np.array(GROUPS),
        )
        out = out or ckpt.parent / f"diag_hard_{ckpt.stem}.npz"
        np.savez_compressed(out, **res)

        # ------------------------------------------------------------ summary per category
        summary = summarize(res)
        Path(out).with_suffix(".json").write_text(json.dumps(summary, indent=1))
        print_summary(summary, k)
        print("saved", out)
    finally:
        env.close()


def summarize(r: dict) -> dict:
    k = int(r["k"])
    by = defaultdict(list)
    for i, c in enumerate(r["categories"]):
        by[str(c)].append(i)
    out = {}
    for c, idx in sorted(by.items()):
        idx = np.array(idx)
        ek, gk = r["errk"][idx], r["errk_groups"][idx]
        air, inv, fast = r["airborne"][idx] > 0.5, r["ref_gravity_z"][idx] > 0.0, r["ref_ang_speed"][idx] > 3.0
        ok = ~np.isnan(ek)
        pre = np.zeros_like(ok)  # the k-step errors of the last 25 steps before a failure
        for j, i in enumerate(idx):
            if r["failed"][i] > 0.5:
                L = int(r["ep_len"][i])
                pre[j, max(0, L - 25) : L + 1] = True
        m = lambda sel: float(np.nanmean(ek[sel & ok])) if (sel & ok).any() else float("nan")  # noqa: E731
        out[c] = {
            "clips": int(len(idx)),
            "failed": int((r["failed"][idx] > 0.5).sum()),
            "median_fail_frac": float(np.median((r["ep_len"][idx] / r["clip_len"][idx])[r["failed"][idx] > 0.5])) if (r["failed"][idx] > 0.5).any() else None,
            "windows": int(ok.sum()),
            f"err{k}_all": m(np.ones_like(ok)),
            f"err{k}_airborne": m(air),
            f"err{k}_upside_down": m(inv),
            f"err{k}_ang_speed_gt3": m(fast),
            f"err{k}_last25_before_fail": m(pre),
            "frac_airborne": float(air[ok].mean()) if ok.any() else float("nan"),
            "frac_ang_speed_gt3": float(fast[ok].mean()) if ok.any() else float("nan"),
            "max_ref_ang_speed_median": float(np.median(np.nanmax(np.where(r["valid"][idx] > 0.5, r["ref_ang_speed"][idx], np.nan), axis=1))),
            f"err{k}_groups_all": {g: float(np.nanmean(gk[..., gi][ok])) for gi, g in enumerate(r["groups"])},
            f"err{k}_groups_last25": {g: float(np.nanmean(gk[..., gi][pre & ok])) if (pre & ok).any() else float("nan") for gi, g in enumerate(r["groups"])},
        }
    return out


def print_summary(s: dict, k: int) -> None:
    hdr = f"{'category':24s} {'fail':>8s} {'all':>8s} {'air':>8s} {'inv':>8s} {'w>3':>8s} {'pre-fail':>9s} {'%air':>6s} {'max|w|':>7s}"
    print(f"{k}-step normalised world-model error by state of the reference (nan = no such frames)")
    print(hdr)
    for c, v in s.items():
        print(
            f"{c[:24]:24s} {v['failed']:3d}/{v['clips']:<4d} {v[f'err{k}_all']:8.4f} {v[f'err{k}_airborne']:8.4f} {v[f'err{k}_upside_down']:8.4f} "
            f"{v[f'err{k}_ang_speed_gt3']:8.4f} {v[f'err{k}_last25_before_fail']:9.4f} {100 * v['frac_airborne']:6.1f} {v['max_ref_ang_speed_median']:7.2f}"
        )


if __name__ == "__main__":
    tyro.cli(main)
