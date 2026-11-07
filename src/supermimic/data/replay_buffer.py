"""GPU ring replay buffer laid out as [env, time].

Each row holds the state ``s_t`` at the start of a control step together with the
transition that followed it: clean action ``a_t``, executed action ``a_exec_t``, sim
context ``xi_t``, reward ``r_{t+1}``, the target frame/clip of that step and bookkeeping
flags. When an episode ends an extra *terminal* row is written that holds ``s_{T+1}``
(no transition out of it) and the termination label of that state, so k-step windows
can use it as their last target state and the termination head gets positive labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WindowBatch:
    states: torch.Tensor  # (k+1, n, S)
    a_exec: torch.Tensor  # (k, n, A)
    a_clean: torch.Tensor  # (k, n, A)
    xi: torch.Tensor  # (n, X) context at the window start (constant inside an episode)
    reward: torch.Tensor  # (k, n)
    done: torch.Tensor  # (k, n) termination label of states 1..k
    frame: torch.Tensor  # (k, n) target frame of each transition


@dataclass
class StartBatch:
    state: torch.Tensor  # (n, S)
    xi: torch.Tensor  # (n, X)
    frame: torch.Tensor  # (n,) current target frame
    prev_action: torch.Tensor  # (n, A) clean action of the previous step (0 at episode start)
    hist_states: torch.Tensor  # (hist, n, S) states s_{t-hist+1..t} (backfilled at episode start)
    hist_actions: torch.Tensor  # (hist, n, A) clean actions a_{t-hist..t-1} aligned with hist_states' obs


# torch.multinomial on CUDA rejects more than 2^24 categories; a buffer of num_envs x capacity rows passes
# that with e.g. 65536 environments x 300 rows, so larger buffers sample the valid rows another way
MULTINOMIAL_MAX_CATEGORIES = 2**24


class ReplayBuffer:
    def __init__(self, num_envs: int, capacity: int, state_dim: int, act_dim: int, xi_dim: int, device):
        self.E, self.T = num_envs, capacity
        self.device = torch.device(device)
        E, T = num_envs, capacity
        f = lambda *shape: torch.zeros(E, T, *shape, device=self.device)  # noqa: E731
        self.s = f(state_dim)
        self.a = f(act_dim)
        self.a_exec = f(act_dim)
        self.xi = f(xi_dim)
        self.r = f()
        self.done = f()
        self.frame = torch.zeros(E, T, dtype=torch.long, device=self.device)
        self.clip = torch.zeros(E, T, dtype=torch.long, device=self.device)
        self.ep_id = torch.full((E, T), -1, dtype=torch.long, device=self.device)
        self.terminal = torch.zeros(E, T, dtype=torch.bool, device=self.device)
        self.ep_start = torch.zeros(E, T, dtype=torch.bool, device=self.device)
        self.valid = torch.zeros(E, T, dtype=torch.bool, device=self.device)
        self.ptr = torch.zeros(E, dtype=torch.long, device=self.device)
        self.ep_counter = torch.zeros(E, dtype=torch.long, device=self.device)
        self.total_rows = 0
        self._env_ar = torch.arange(E, device=self.device)

    # ------------------------------------------------------------------ writes

    @torch.no_grad()
    def add(self, env_ids, s, a, a_exec, xi, r, done, frame, clip, terminal, ep_start) -> None:
        p = self.ptr[env_ids]
        self.s[env_ids, p] = s
        self.a[env_ids, p] = a
        self.a_exec[env_ids, p] = a_exec
        self.xi[env_ids, p] = xi
        self.r[env_ids, p] = r
        self.done[env_ids, p] = done
        self.frame[env_ids, p] = frame
        self.clip[env_ids, p] = clip
        self.ep_id[env_ids, p] = self.ep_counter[env_ids]
        self.terminal[env_ids, p] = terminal
        self.ep_start[env_ids, p] = ep_start
        self.valid[env_ids, p] = True
        self.ptr[env_ids] = (p + 1) % self.T
        self.total_rows += int(env_ids.numel())

    @torch.no_grad()
    def end_episode(self, env_ids) -> None:
        self.ep_counter[env_ids] += 1

    @property
    def size(self) -> int:
        return int(self.valid.sum().item())

    # ------------------------------------------------------------------ sampling helpers

    def _rolled(self, x: torch.Tensor, j: int) -> torch.Tensor:
        """x[:, (t + j) % T] for every t."""
        return torch.roll(x, shifts=-j, dims=1) if j else x

    @torch.no_grad()
    def window_mask(self, k: int) -> torch.Tensor:
        """(E,T) True where rows t..t+k are valid, same episode, and t..t+k-1 are non-terminal."""
        ok = self.valid & ~self.terminal
        ep0 = self.ep_id
        for j in range(1, k + 1):
            v = self._rolled(self.valid, j) & (self._rolled(ep0, j) == ep0)
            if j < k:
                v &= ~self._rolled(self.terminal, j)
            ok &= v
        # The row currently under the write pointer may be overwritten next; the rolled window must not
        # wrap across the write pointer. Rows t with t + k crossing ptr are excluded by the ep_id test
        # (new rows carry a newer ep_id) except for same-episode wraps, which are rare; exclude explicitly.
        t = torch.arange(self.T, device=self.device)[None, :]
        dist_to_ptr = (self.ptr[:, None] - t - 1) % self.T  # rows strictly before ptr: 0 = newest
        ok &= dist_to_ptr >= k
        return ok

    @torch.no_grad()
    def _sample_from_mask(self, mask: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        flat = mask.reshape(-1)
        total = int(flat.sum().item())
        if total == 0:
            raise RuntimeError("replay buffer has no sampleable rows yet")
        if flat.numel() <= MULTINOMIAL_MAX_CATEGORIES:
            idx = torch.multinomial(flat.float(), n, replacement=True)
        else:  # CUDA multinomial takes at most 2^24 categories: same (uniform over valid rows) distribution
            rows = flat.nonzero().squeeze(-1)
            idx = rows[torch.randint(0, rows.numel(), (n,), device=rows.device)]
        return idx // self.T, idx % self.T

    @torch.no_grad()
    def sample_windows(self, n: int, k: int) -> WindowBatch:
        e, t = self._sample_from_mask(self.window_mask(k), n)
        offs = torch.arange(k + 1, device=self.device)
        tt = (t[None, :] + offs[:, None]) % self.T  # (k+1, n)
        ee = e[None, :].expand(k + 1, n)
        return WindowBatch(
            states=self.s[ee, tt],
            a_exec=self.a_exec[ee[:-1], tt[:-1]],
            a_clean=self.a[ee[:-1], tt[:-1]],
            xi=self.xi[e, t],
            reward=self.r[ee[:-1], tt[:-1]],
            done=self.done[ee[1:], tt[1:]],
            frame=self.frame[ee[:-1], tt[:-1]],
        )

    @torch.no_grad()
    def sample_starts(self, n: int, hist: int) -> StartBatch:
        mask = self.valid & ~self.terminal
        t_idx = torch.arange(self.T, device=self.device)[None, :]
        dist_to_ptr = (self.ptr[:, None] - t_idx - 1) % self.T
        mask &= dist_to_ptr < self.T - hist - 1  # keep history rows from being overwritten underneath
        e, t = self._sample_from_mask(mask, n)
        offs = torch.arange(hist, device=self.device) - (hist - 1)  # -(hist-1) .. 0
        tt = (t[None, :] + offs[:, None]) % self.T  # (hist, n)
        ee = e[None, :].expand(hist, n)
        same_ep = (self.ep_id[ee, tt] == self.ep_id[e, t][None, :]) & self.valid[ee, tt]
        # Backfill: for rows outside the episode, use the first in-episode row of the window.
        first_ok = same_ep.float().argmax(dim=0)  # first True index per column (same_ep[-1] is always True)
        pos = torch.arange(hist, device=self.device)[:, None].expand(hist, n)
        tt_fixed = torch.where(same_ep, tt, tt[first_ok, torch.arange(n, device=self.device)][None, :].expand(hist, n))
        hist_states = self.s[ee, tt_fixed]
        # Actions preceding each history state: a at row-1 if same episode else 0.
        tprev = (tt_fixed - 1) % self.T
        prev_ok = (self.ep_id[ee, tprev] == self.ep_id[e, t][None, :]) & self.valid[ee, tprev] & ~self.ep_start[ee, tt_fixed]
        hist_actions = torch.where(prev_ok[..., None], self.a[ee, tprev], torch.zeros_like(self.a[ee, tprev]))
        del pos
        return StartBatch(
            state=self.s[e, t],
            xi=self.xi[e, t],
            frame=self.frame[e, t],
            prev_action=hist_actions[-1],
            hist_states=hist_states,
            hist_actions=hist_actions,
        )


class MixedReplay:
    """Two buffers sampled together: a share ``frac_windows`` of every world-model batch and ``frac_starts`` of every
    imagination-start batch comes from ``extra`` (e.g. a fixed set of demonstrations from another controller, kept
    for the whole run), the rest from ``main`` (the learner's own, recent data). Same sampling interface as
    ``ReplayBuffer``."""

    def __init__(self, main: ReplayBuffer, extra: ReplayBuffer, frac_windows: float, frac_starts: float | None = None):
        self.main, self.extra = main, extra
        self.frac_windows = frac_windows
        self.frac_starts = frac_windows if frac_starts is None else frac_starts

    @property
    def size(self) -> int:
        return self.main.size + self.extra.size

    @staticmethod
    def _split(n: int, frac: float) -> int:
        return min(n, max(0, int(round(frac * n))))

    @torch.no_grad()
    def sample_windows(self, n: int, k: int) -> WindowBatch:
        ne = self._split(n, self.frac_windows)
        if ne == 0:
            return self.main.sample_windows(n, k)
        if ne == n:
            return self.extra.sample_windows(n, k)
        a, b = self.main.sample_windows(n - ne, k), self.extra.sample_windows(ne, k)
        return WindowBatch(states=torch.cat([a.states, b.states], 1), a_exec=torch.cat([a.a_exec, b.a_exec], 1), a_clean=torch.cat([a.a_clean, b.a_clean], 1),
                           xi=torch.cat([a.xi, b.xi], 0), reward=torch.cat([a.reward, b.reward], 1), done=torch.cat([a.done, b.done], 1),
                           frame=torch.cat([a.frame, b.frame], 1))

    @torch.no_grad()
    def sample_starts(self, n: int, hist: int) -> StartBatch:
        ne = self._split(n, self.frac_starts)
        if ne == 0:
            return self.main.sample_starts(n, hist)
        if ne == n:
            return self.extra.sample_starts(n, hist)
        a, b = self.main.sample_starts(n - ne, hist), self.extra.sample_starts(ne, hist)
        return StartBatch(state=torch.cat([a.state, b.state], 0), xi=torch.cat([a.xi, b.xi], 0), frame=torch.cat([a.frame, b.frame], 0),
                          prev_action=torch.cat([a.prev_action, b.prev_action], 0), hist_states=torch.cat([a.hist_states, b.hist_states], 1),
                          hist_actions=torch.cat([a.hist_actions, b.hist_actions], 1))

    def save_extra(self, path) -> None:
        save_buffer(self.extra, path)


def save_buffer(buf: ReplayBuffer, path) -> None:
    """All of a buffer's tensors and counters, for ``load_buffer``."""
    torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in vars(buf).items() if k != "device"}, path)


def load_buffer(path, device) -> ReplayBuffer:
    d = torch.load(path, map_location="cpu")
    buf = ReplayBuffer.__new__(ReplayBuffer)
    buf.device = torch.device(device)
    for k, v in d.items():
        setattr(buf, k, v.to(buf.device) if torch.is_tensor(v) else v)
    return buf
