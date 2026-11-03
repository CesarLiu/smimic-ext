"""Return-based reset sampling over task time (paper Appendix D.1, plan D14), independent of the robot.

Task time is a set of sequences (motion clips, reference trajectories, ...) concatenated into one
global step index and cut into fixed-size bins (``SequenceLayout``). The sampler keeps a running
estimate G_b of the discounted return earned from each bin, logs the simulator rewards of each
environment's episode, folds finished episodes into G_b and draws reset steps from the mixture
50 % uniform over bins / 40 % softmax(-T G_b / G_max) / 10 % memory of past softmax weights.

``MultiClipMotionCommand`` (the G1 motion-tracking env) owns one; a new task's env builds a layout
from its sequence lengths (``SequenceLayout.from_lengths``) and uses the same sampler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class SequenceLayout:
    seq_start: torch.Tensor  # (S,) first global step of each sequence
    seq_last: torch.Tensor  # (S,) last global step of each sequence
    step_seq: torch.Tensor  # (T,) sequence of each global step
    seq_bin_offset: torch.Tensor  # (S,) index of each sequence's first bin
    bin_start: torch.Tensor  # (B,) first step of each bin
    bin_end: torch.Tensor  # (B,) one past the last step of each bin
    bin_size: int

    @property
    def num_sequences(self) -> int:
        return int(self.seq_start.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.step_seq.shape[0])

    @property
    def num_bins(self) -> int:
        return int(self.bin_start.shape[0])

    def first_of(self, idx: torch.Tensor) -> torch.Tensor:
        return self.seq_start[self.step_seq[idx]]

    def last_of(self, idx: torch.Tensor) -> torch.Tensor:
        return self.seq_last[self.step_seq[idx]]

    def bin_of(self, idx: torch.Tensor) -> torch.Tensor:
        seq = self.step_seq[idx]
        return self.seq_bin_offset[seq] + (idx - self.seq_start[seq]) // self.bin_size

    @classmethod
    def from_lengths(cls, lengths, bin_size: int, device="cpu") -> SequenceLayout:
        """Sequences of the given lengths, concatenated in order (the motion library's layout)."""
        length = torch.as_tensor(lengths, dtype=torch.long, device=device)
        start = torch.cumsum(length, 0) - length
        s = int(length.shape[0])
        bins_per_seq = (length + bin_size - 1) // bin_size
        offset = torch.cumsum(bins_per_seq, 0) - bins_per_seq
        num_bins = int(bins_per_seq.sum().item())
        bin_seq = torch.repeat_interleave(torch.arange(s, device=device), bins_per_seq)
        local = torch.arange(num_bins, device=device) - offset[bin_seq]
        bin_start = start[bin_seq] + local * bin_size
        bin_end = torch.minimum(bin_start + bin_size, start[bin_seq] + length[bin_seq])
        return cls(start, start + length - 1, torch.repeat_interleave(torch.arange(s, device=device), length), offset, bin_start, bin_end, int(bin_size))

    @classmethod
    def from_motion_library(cls, lib) -> SequenceLayout:
        """A view of a ``MotionLibrary``'s clip and bin tensors (no copies)."""
        return cls(lib.clip_start, lib.clip_last, lib.frame_clip, lib.clip_bin_offset, lib.bin_start, lib.bin_end, lib.bin_frames)


@dataclass
class ResetSamplerCfg:
    sampling_mode: Literal["return", "uniform", "start"] = "return"
    uniform_ratio: float = 0.5
    hard_ratio: float = 0.4
    memory_ratio: float = 0.1
    softmax_temp: float = 25.0
    g_max: float = 10.0
    gamma: float = 0.99
    ema_alpha: float = 0.05
    memory_alpha: float = 0.01
    min_remaining_steps: int = 100
    gmax_mode: Literal["const", "remaining"] = "const"
    max_logged_steps: int = 1024


def normalize_return_to_go(rtg: torch.Tensor, remaining: torch.Tensor, gamma: float, g_max: float) -> torch.Tensor:
    """Scale returns-to-go by the best return still available (gmax_mode="remaining").

    ``remaining`` is the number of reward steps left in the sequence from each step. With the per-step
    reward bound g_max * (1 - gamma), the best return-to-go over m steps is g_max * (1 - gamma^m); the
    result is g_max * rtg / best, so perfect tracking scores g_max whether the sequence ends in 5 steps
    or 500.
    """
    best = g_max * (1.0 - torch.pow(torch.as_tensor(gamma, dtype=torch.float64, device=rtg.device), remaining.clamp_min(1).double()))
    return (g_max * rtg.double() / best).clamp(0.0, g_max).to(rtg.dtype)


class ReturnBasedResetSampler:
    """Per-bin return estimates, per-env episode logs and the reset distribution."""

    def __init__(self, layout: SequenceLayout, cfg: ResetSamplerCfg, num_envs: int, episode_steps: int, device):
        self.layout, self.cfg = layout, cfg
        self.device = device
        n, nb = num_envs, layout.num_bins
        self.bin_return = torch.full((nb,), cfg.g_max, device=device)
        self.bin_memory = torch.full((nb,), 1.0 / nb, device=device)
        self.max_ep_steps = min(episode_steps, cfg.max_logged_steps)
        self.ep_rewards = torch.zeros(n, self.max_ep_steps, device=device)
        self.ep_bins = torch.zeros(n, self.max_ep_steps, dtype=torch.long, device=device)
        self.ep_steps = torch.zeros(n, self.max_ep_steps, dtype=torch.long, device=device)
        self.ep_len = torch.zeros(n, dtype=torch.long, device=device)
        self._gamma_pow = cfg.gamma ** torch.arange(self.max_ep_steps, device=device, dtype=torch.float64)
        self.bootstrap_fn = None  # optional: env_ids -> value estimate for truncated episodes

    def log_step(self, ids: torch.Tensor, reward: torch.Tensor, idx: torch.Tensor) -> None:
        """Append the reward earned by envs ``ids`` against target steps ``idx`` to their episode logs."""
        pos = self.ep_len[ids].clamp_max(self.max_ep_steps - 1)
        self.ep_rewards[ids, pos] = reward
        self.ep_bins[ids, pos] = self.layout.bin_of(idx)
        self.ep_steps[ids, pos] = idx
        self.ep_len[ids] = torch.minimum(self.ep_len[ids] + 1, torch.full_like(pos, self.max_ep_steps))

    def finalize(self, env_ids: torch.Tensor, terminated: torch.Tensor, seq_end: torch.Tensor) -> torch.Tensor | None:
        """Fold the finished episodes of ``env_ids`` into the per-bin estimates.

        ``terminated``: the episode ended by a termination; ``seq_end``: it reached the end of its
        sequence. Anything else was truncated (time-out). Returns each episode's return-to-go from its
        first step when the estimates were updated (for logging), otherwise None.
        """
        cfg = self.cfg
        lens = self.ep_len[env_ids]
        if lens.sum() == 0:
            return None
        T = int(lens.max().item())
        r = self.ep_rewards[env_ids, :T].double()
        bins = self.ep_bins[env_ids, :T]
        tmask = torch.arange(T, device=self.device)[None, :] < lens[:, None]
        r = r * tmask
        gp = self._gamma_pow[:T]
        # Return-to-go R_t = sum_{s>=t} gamma^{s-t} r_s via a reverse cumulative sum.
        disc = r * gp[None, :]
        rtg = torch.flip(torch.cumsum(torch.flip(disc, [1]), 1), [1]) / gp[None, :]
        # Truncated (timed-out) episodes: bootstrap with the critic if available, otherwise
        # drop the steps too close to the truncation point (plan D14).
        truncated = ~terminated & ~seq_end
        if truncated.any():
            if self.bootstrap_fn is not None:
                v = self.bootstrap_fn(env_ids).double()  # (n,)
                remaining = (lens[:, None] - torch.arange(T, device=self.device)[None, :]).clamp_min(0).double()
                rtg = rtg + truncated[:, None] * (cfg.gamma ** remaining) * v[:, None]
            else:
                remaining = lens[:, None] - torch.arange(T, device=self.device)[None, :]
                tmask = tmask & ~(truncated[:, None] & (remaining < cfg.min_remaining_steps))
        if cfg.gmax_mode == "remaining":
            steps = self.ep_steps[env_ids, :T]
            remaining = self.layout.last_of(steps) - steps + 1
            rtg = normalize_return_to_go(rtg, remaining, cfg.gamma, cfg.g_max)
        valid = tmask.reshape(-1)
        b = bins.reshape(-1)[valid]
        g = rtg.reshape(-1)[valid].float()
        if b.numel() == 0:
            return None
        nb = self.layout.num_bins
        sums = torch.zeros(nb, device=self.device).index_add_(0, b, g)
        cnts = torch.zeros(nb, device=self.device).index_add_(0, b, torch.ones_like(g))
        touched = cnts > 0
        mean = sums[touched] / cnts[touched]
        self.bin_return[touched] = (1 - cfg.ema_alpha) * self.bin_return[touched] + cfg.ema_alpha * mean
        rtg0 = rtg[:, 0].float()
        self.ep_len[env_ids] = 0
        return rtg0

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``n`` reset steps, and the normalised entropy of the bin distribution (return mode only)."""
        lay, cfg = self.layout, self.cfg
        if cfg.sampling_mode == "start":
            seqs = torch.randint(0, lay.num_sequences, (n,), device=self.device)
            return lay.seq_start[seqs], None
        if cfg.sampling_mode == "uniform":
            # Uniform over steps (long sequences get proportionally more starts), leaving >= 2 steps.
            f = torch.randint(0, lay.num_steps, (n,), device=self.device)
            return torch.minimum(f, lay.last_of(f) - 2).clamp_min(0), None
        assert cfg.sampling_mode == "return"
        nb = lay.num_bins
        hard = torch.softmax(-cfg.softmax_temp * self.bin_return / cfg.g_max, dim=0)
        self.bin_memory = (1 - cfg.memory_alpha) * self.bin_memory + cfg.memory_alpha * hard
        p = cfg.uniform_ratio / nb + cfg.hard_ratio * hard + cfg.memory_ratio * self.bin_memory / self.bin_memory.sum()
        p = p / p.sum()
        b = torch.multinomial(p, n, replacement=True)
        u = torch.rand(n, device=self.device)
        f = lay.bin_start[b] + (u * (lay.bin_end[b] - lay.bin_start[b]).float()).long()
        f = torch.minimum(f, lay.last_of(f) - 2).clamp_min(lay.first_of(f))
        ent = -(p * (p + 1e-12).log()).sum() / math.log(max(nb, 2))
        return f, ent
