"""Batched vehicle reference-tracking environment (VEHICLE_PROPOSAL.md, E0).

Pipeline per control step (50 Hz): policy action (normalised steering-angle and acceleration commands)
-> command delay (per-env, from the sim context) -> actuators (steering servo with a time constant and
the vehicle's rate limit; first-order longitudinal response) -> ``VehicleDynamics`` (10 substeps, ABS and
ESC by default) -> reward and termination against the reference (``tracking.py``).

The environment also
* samples the sim context xi per episode (road friction, tyre stiffness, mass, CoG shift, delay);
* records transitions into a ``ReplayBuffer`` the way the G1 recorder does (one row per transition, plus
  a terminal row holding the last state and its termination label);
* draws episode starts from ``ReturnBasedResetSampler`` over the reference library's trajectories.

References (``ReferenceLibrary.generate``) are produced by the same actuator + dynamics pipeline under
the nominal context with a scripted driver, so they are feasible by construction for that context and
replaying their recorded actions reproduces them exactly. Lower-friction contexts can make parts of a
reference infeasible; those episodes simply terminate (VEHICLE_PROPOSAL.md §4, option 3, for now).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.tasks.reset_sampler import ResetSamplerCfg, ReturnBasedResetSampler, SequenceLayout
from supermimic.vehicles import tracking
from supermimic.vehicles.dynamics import (
    AX,
    AY,
    DELTA,
    VX,
    VY,
    YAW,
    AssistCfg,
    Context,
    R,
    VehicleDynamics,
    W,
    X,
    Y,
)
from supermimic.vehicles.dynamics import (
    STATE_DIM as SIM_STATE_DIM,
)
from supermimic.vehicles.params import VehicleParams

ACTION_DIM = 2
XI_NAMES = ("mu", "tire_stiffness", "mass", "cog_shift", "delay")
XI_DIM = len(XI_NAMES)


@dataclass
class VehicleEnvCfg:
    num_envs: int = 4096
    dt: float = 0.02
    substeps: int = 10
    episode_steps: int = 500
    steer_scale: float = 0.2  # [rad] front-wheel angle per unit steering action
    accel_scale: float = 6.0  # [m/s^2] per unit acceleration action
    steer_tau: float = 0.05  # [s] steering servo time constant
    accel_tau: float = 0.15  # [s] longitudinal (engine / brake) response
    max_delay: int = 3  # [control steps]
    domain_randomization: bool = True
    mu_range: tuple[float, float] = (0.6, 1.1)
    tire_stiffness_range: tuple[float, float] = (0.85, 1.15)
    mass_range: tuple[float, float] = (0.9, 1.15)
    cog_shift_range: tuple[float, float] = (-0.1, 0.1)
    reset_noise: bool = True
    reset_lat_std: float = 0.2
    reset_yaw_std: float = 0.02
    reset_speed_std: float = 0.3
    preview: tuple[int, ...] = (1, 2, 4, 6, 10, 15, 20, 30, 40, 50)  # goal: reference steps ahead
    replay_capacity: int = 1000
    assist: AssistCfg = field(default_factory=AssistCfg)
    tracking: tracking.TrackingCfg = field(default_factory=tracking.TrackingCfg)
    sampler: ResetSamplerCfg = field(default_factory=ResetSamplerCfg)

    @property
    def state_dim(self) -> int:
        """Logged (Markov) state: simulator state, actuator state, pending commands, ABS / ESC flags."""
        return SIM_STATE_DIM + 1 + self.max_delay * ACTION_DIM + 5


class Actuators:
    """Command delay, steering servo and longitudinal lag. State: filtered acceleration and the last
    ``max_delay`` commands (newest first)."""

    def __init__(self, cfg: VehicleEnvCfg, n: int, device, dtype=torch.float32):
        self.cfg = cfg
        self.a_act = torch.zeros(n, device=device, dtype=dtype)
        self.pending = torch.zeros(n, cfg.max_delay, ACTION_DIM, device=device, dtype=dtype)
        self.delay = torch.zeros(n, dtype=torch.long, device=device)
        self._alpha = 1.0 - math.exp(-cfg.dt / cfg.accel_tau)

    def reset(self, ids, command: torch.Tensor, a_act: torch.Tensor, delay: torch.Tensor) -> None:
        self.pending[ids] = command[:, None, :]
        self.a_act[ids] = a_act
        self.delay[ids] = delay

    def step(self, action: torch.Tensor, delta: torch.Tensor):
        """-> (steering rate, acceleration request) for the dynamics, held over the control step."""
        cfg = self.cfg
        hist = torch.cat([action[:, None, :], self.pending], 1)  # (n, max_delay + 1, 2), newest first
        cmd = hist[torch.arange(hist.shape[0], device=hist.device), self.delay]
        self.pending = hist[:, : cfg.max_delay]
        steer_rate = (cmd[:, 0] * cfg.steer_scale - delta) / cfg.steer_tau
        self.a_act = self.a_act + self._alpha * (cmd[:, 1] * cfg.accel_scale - self.a_act)
        return steer_rate, self.a_act

    def state(self) -> torch.Tensor:
        return torch.cat([self.a_act[:, None], self.pending.reshape(self.pending.shape[0], -1)], -1)


# ---------------------------------------------------------------------- references


@dataclass
class ReferenceLibrary:
    """Concatenated reference trajectories at the control rate; ``actions[i]`` produced step i -> i+1."""

    x: torch.Tensor
    y: torch.Tensor
    yaw: torch.Tensor
    v: torch.Tensor
    beta: torch.Tensor
    r: torch.Tensor
    vx: torch.Tensor
    vy: torch.Tensor
    delta: torch.Tensor
    a_act: torch.Tensor
    actions: torch.Tensor  # (T, 2)
    lengths: tuple[int, ...]
    bin_size: int = 50

    def __post_init__(self):
        self.layout = SequenceLayout.from_lengths(self.lengths, self.bin_size, self.x.device)

    def ref(self, idx: torch.Tensor) -> tracking.VehicleRef:
        return tracking.VehicleRef(self.x[idx], self.y[idx], self.yaw[idx], self.v[idx], self.beta[idx], self.r[idx])

    def to(self, device) -> ReferenceLibrary:
        t = {k: getattr(self, k).to(device) for k in ("x", "y", "yaw", "v", "beta", "r", "vx", "vy", "delta", "a_act", "actions")}
        return ReferenceLibrary(**t, lengths=self.lengths, bin_size=self.bin_size)

    @classmethod
    def generate(cls, cfg: VehicleEnvCfg, num: int = 64, steps: int = 1000, seed: int = 0, device="cpu", params: VehicleParams | None = None,
                 max_beta: float = 0.12, min_speed: float = 4.0, oversample: float = 2.0) -> ReferenceLibrary:
        """References from a scripted driver under the nominal context and no delay: a slowly varying target
        speed (6-30 m/s) and a lateral-acceleration target (sum of two sinusoids, peak 2-7 m/s^2 per trajectory)
        turned into steering commands through a steady-state understeer approximation. The driver is open-loop
        in yaw, so some candidates leave the stable range; ``oversample`` x ``num`` candidates are simulated and
        the first ``num`` that keep |sideslip| < ``max_beta`` and speed > ``min_speed`` throughout are kept."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        rnd = lambda *shape: torch.rand(*shape, generator=g).to(device)
        dyn = VehicleDynamics(params, cfg.dt, cfg.substeps, cfg.assist)
        p = dyn.p
        n = max(num, math.ceil(oversample * num))
        v0 = 8.0 + 18.0 * rnd(n)
        s = dyn.initial_state(v0)
        flags = dyn.new_flags(n, device)
        ctx = Context.nominal(n, device)
        act = Actuators(cfg, n, device)
        act.reset(slice(None), torch.zeros(n, ACTION_DIM, device=device), torch.zeros(n, device=device), torch.zeros(n, dtype=torch.long, device=device))
        ay_peak = 2.0 + 5.0 * rnd(n)
        f1, f2 = 0.05 + 0.25 * rnd(n), 0.1 + 0.4 * rnd(n)
        ph1, ph2 = 2 * math.pi * rnd(n), 2 * math.pi * rnd(n)
        w1 = 0.4 + 0.6 * rnd(n)
        v_target = v0.clone()
        rows = {k: [] for k in ("x", "y", "yaw", "v", "beta", "r", "vx", "vy", "delta", "a_act", "actions")}
        for i in range(steps):
            t = i * cfg.dt
            ay_t = ay_peak * (w1 * torch.sin(2 * math.pi * f1 * t + ph1) + (1 - w1) * torch.sin(2 * math.pi * f2 * t + ph2))
            if i % 50 == 0:  # new target speed every second, a bounded random walk
                v_target = (v_target + 4.0 * (rnd(n) - 0.5)).clamp(6.0, 30.0)
            vx = s[:, VX].clamp_min(3.0)
            delta_target = p.L * ay_t / vx**2 * (1 + (vx / cfg.assist.esc_char_speed) ** 2)
            a_req = (1.5 * (v_target - s[:, VX])).clamp(-4.0, 4.0)
            action = torch.stack([delta_target / cfg.steer_scale, a_req / cfg.accel_scale], -1)
            vel = torch.sqrt(s[:, VX] ** 2 + s[:, VY] ** 2)
            for k, v in (("x", s[:, X]), ("y", s[:, Y]), ("yaw", s[:, YAW]), ("v", vel), ("beta", torch.atan2(s[:, VY], s[:, VX].clamp_min(0.1))),
                         ("r", s[:, R]), ("vx", s[:, VX]), ("vy", s[:, VY]), ("delta", s[:, DELTA]), ("a_act", act.a_act.clone()), ("actions", action)):
                rows[k].append(v.clone())
            sr, ac = act.step(action, s[:, DELTA])
            s, flags = dyn.step(s, flags, sr, ac, ctx)
        traj = {k: torch.stack(v, 1) for k, v in rows.items()}  # (n, steps, ...)
        ok = (traj["beta"].abs().amax(1) < max_beta) & (traj["v"].amin(1) > min_speed)
        keep = ok.nonzero().squeeze(-1)[:num]
        if keep.numel() < num:
            raise RuntimeError(f"only {keep.numel()} of {n} candidate references stayed stable; raise oversample")
        out = {k: v[keep].reshape(num * steps, *v.shape[2:]) for k, v in traj.items()}
        return cls(**out, lengths=(steps,) * num)


# ---------------------------------------------------------------------- environment


class VehicleTrackingEnv:
    def __init__(self, cfg: VehicleEnvCfg, library: ReferenceLibrary, device="cpu", params: VehicleParams | None = None, seed: int = 0, record: bool = True):
        self.cfg, self.lib, self.device = cfg, library.to(device), device
        n = self.num_envs = cfg.num_envs
        self.dyn = VehicleDynamics(params, cfg.dt, cfg.substeps, cfg.assist)
        self.act = Actuators(cfg, n, device)
        self.sampler = ReturnBasedResetSampler(self.lib.layout, cfg.sampler, n, cfg.episode_steps, device)
        self.buffer = ReplayBuffer(n, cfg.replay_capacity, cfg.state_dim, ACTION_DIM, XI_DIM, device) if record else None
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.s = torch.zeros(n, SIM_STATE_DIM, device=device)
        self.flags = self.dyn.new_flags(n, device)
        self.ctx = Context.nominal(n, device)
        self.xi = torch.zeros(n, XI_DIM, device=device)
        self.frame = torch.zeros(n, dtype=torch.long, device=device)
        self.ep_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.prev_action = torch.zeros(n, ACTION_DIM, device=device)
        self.forced_frame = torch.full((n,), -1, dtype=torch.long, device=device)  # evaluation hook (as the G1 command)
        self.forced_xi: torch.Tensor | None = None  # evaluation hook: fixed context for every reset
        # replay bookkeeping
        self._prev_state = torch.zeros(n, cfg.state_dim, device=device)
        self._prev_xi = torch.zeros(n, XI_DIM, device=device)
        self._prev_frame = torch.zeros(n, dtype=torch.long, device=device)
        self._ep_start = torch.ones(n, dtype=torch.bool, device=device)
        self.reset()

    # ------------------------------------------------------------------ state, observation, goal

    def state(self) -> torch.Tensor:
        """The logged Markov state (``cfg.state_dim``)."""
        return torch.cat([self.s, self.act.state(), self.flags], -1)

    def observation(self) -> torch.Tensor:
        """Noise-free on-board measurements: speed estimate, IMU (yaw rate, ax, ay), steering angle, wheel
        speeds [m/s], filtered acceleration and the previous action."""
        s = self.s
        return torch.cat([s[:, VX:VX + 1], s[:, R:R + 1], s[:, AX:AY + 1], s[:, DELTA:DELTA + 1], s[:, W] * self.dyn.p.R_w, self.act.a_act[:, None], self.prev_action], -1)

    def goal(self) -> torch.Tensor:
        """Reference preview: for each offset, the reference pose relative to the vehicle (body frame),
        the heading change, the reference speed and sideslip."""
        offs = torch.tensor(self.cfg.preview, device=self.device) - 1
        idx = torch.minimum(self.frame[:, None] + offs[None], self.lib.layout.last_of(self.frame)[:, None])
        ref = self.lib.ref(idx.reshape(-1))
        n, k = idx.shape
        dx = ref.x.reshape(n, k) - self.s[:, X:X + 1]
        dy = ref.y.reshape(n, k) - self.s[:, Y:Y + 1]
        c, sn = torch.cos(self.s[:, YAW:YAW + 1]), torch.sin(self.s[:, YAW:YAW + 1])
        feats = torch.stack([c * dx + sn * dy, -sn * dx + c * dy, tracking.wrap(ref.yaw.reshape(n, k) - self.s[:, YAW:YAW + 1]), ref.v.reshape(n, k), ref.beta.reshape(n, k)], -1)
        return feats.reshape(n, k * 5)

    def _obs(self) -> dict[str, torch.Tensor]:
        return {"policy": self.observation(), "goal": self.goal()}

    # ------------------------------------------------------------------ resets

    def _uniform(self, lo, hi, m):
        return lo + (hi - lo) * torch.rand(m, generator=self.gen, device=self.device)

    def _sample_context(self, ids) -> None:
        cfg, m = self.cfg, ids.numel()
        if self.forced_xi is not None:
            xi = self.forced_xi.to(self.device).expand(m, -1).clone()
        elif cfg.domain_randomization:
            xi = torch.stack([
                self._uniform(*cfg.mu_range, m), self._uniform(*cfg.tire_stiffness_range, m), self._uniform(*cfg.mass_range, m),
                self._uniform(*cfg.cog_shift_range, m), torch.randint(0, cfg.max_delay + 1, (m,), generator=self.gen, device=self.device).float(),
            ], -1)
        else:
            xi = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0], device=self.device).expand(m, -1).clone()
        self.xi[ids] = xi
        self.ctx.mu[ids], self.ctx.tire_stiffness[ids], self.ctx.mass[ids], self.ctx.cog_shift[ids] = xi[:, 0], xi[:, 1], xi[:, 2], xi[:, 3]

    def _reset(self, ids: torch.Tensor) -> None:
        if ids.numel() == 0:
            return
        cfg, lib, m = self.cfg, self.lib, ids.numel()
        sampled, _ = self.sampler.sample(m)
        forced = self.forced_frame[ids]
        i = torch.where(forced >= 0, forced, sampled)
        self._sample_context(ids)
        vx, vy, yaw = lib.vx[i].clone(), lib.vy[i].clone(), lib.yaw[i].clone()
        x, y = lib.x[i].clone(), lib.y[i].clone()
        if cfg.reset_noise:
            lat = cfg.reset_lat_std * torch.randn(m, generator=self.gen, device=self.device)
            x, y = x - torch.sin(yaw) * lat, y + torch.cos(yaw) * lat
            yaw = yaw + cfg.reset_yaw_std * torch.randn(m, generator=self.gen, device=self.device)
            vx = (vx + cfg.reset_speed_std * torch.randn(m, generator=self.gen, device=self.device)).clamp_min(1.5)
        s = self.dyn.initial_state(vx, vy=vy, r=lib.r[i], delta=lib.delta[i], x=x, y=y, yaw=yaw)
        self.s[ids] = s
        self.flags[ids] = 0.0
        self.act.reset(ids, lib.actions[i], lib.a_act[i], self.xi[ids, 4].long())
        self.prev_action[ids] = lib.actions[i]
        self.frame[ids] = torch.minimum(i + 1, lib.layout.last_of(i))  # the target of the next state
        self.ep_steps[ids] = 0
        self.sampler.ep_len[ids] = 0
        self._remember(ids)
        self._ep_start[ids] = True

    def reset(self) -> dict[str, torch.Tensor]:
        self._reset(torch.arange(self.num_envs, device=self.device))
        return self._obs()

    def _remember(self, ids) -> None:
        self._prev_state[ids] = self.state()[ids]
        self._prev_xi[ids] = self.xi[ids]
        self._prev_frame[ids] = self.frame[ids]

    # ------------------------------------------------------------------ step

    def step(self, action: torch.Tensor, action_noise: torch.Tensor | None = None):
        """-> obs, reward, terminated, truncated (time-out or end of the reference), info."""
        cfg, lib = self.cfg, self.lib
        a_exec = action if action_noise is None else action + action_noise
        steer_rate, accel = self.act.step(a_exec, self.s[:, DELTA])
        self.s, self.flags = self.dyn.step(self.s, self.flags, steer_rate, accel, self.ctx)
        ref = lib.ref(self.frame)
        terms = tracking.reward_terms(self.s, action, self.prev_action, ref, cfg.tracking)
        reward = terms["total"]
        term = tracking.terminated(self.s, ref, cfg.tracking)
        seq_end = self.frame >= lib.layout.last_of(self.frame)
        self.ep_steps += 1
        timeout = self.ep_steps >= cfg.episode_steps
        done = term | seq_end | timeout
        all_ids = torch.arange(self.num_envs, device=self.device)
        self.sampler.log_step(all_ids, reward, self.frame)
        if self.buffer is not None:
            self.buffer.add(all_ids, s=self._prev_state, a=action, a_exec=a_exec, xi=self._prev_xi, r=reward, done=term.float() * done,
                            frame=self._prev_frame, clip=lib.layout.step_seq[self._prev_frame], terminal=torch.zeros_like(done), ep_start=self._ep_start.clone())
        self._ep_start[:] = False
        self.prev_action = action.detach().clone()
        done_ids = done.nonzero().squeeze(-1)
        live = ~done
        self.frame = torch.where(live, torch.minimum(self.frame + 1, lib.layout.last_of(self.frame)), self.frame)
        if done_ids.numel():
            if self.buffer is not None:  # terminal row: the state the last action reached, with its label
                st = self.state()[done_ids]
                z = torch.zeros(done_ids.numel(), ACTION_DIM, device=self.device)
                self.buffer.add(done_ids, s=st, a=z, a_exec=z, xi=self._prev_xi[done_ids], r=torch.zeros(done_ids.numel(), device=self.device),
                                done=term[done_ids].float(), frame=self.frame[done_ids], clip=lib.layout.step_seq[self.frame[done_ids]],
                                terminal=torch.ones(done_ids.numel(), dtype=torch.bool, device=self.device), ep_start=torch.zeros(done_ids.numel(), dtype=torch.bool, device=self.device))
                self.buffer.end_episode(done_ids)
            self.sampler.finalize(done_ids, term[done_ids], seq_end[done_ids])
            self._reset(done_ids)
        live_ids = live.nonzero().squeeze(-1)
        self._remember(live_ids)
        info = {"reward_terms": terms, "seq_end": seq_end, "timeout": timeout, "done": done}
        return self._obs(), reward, term, (seq_end | timeout) & ~term, info
