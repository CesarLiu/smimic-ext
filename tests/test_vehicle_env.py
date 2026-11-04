"""Vehicle tracking environment (E0): references, tracking reward and termination, actuators and
delays, sim-context sampling, replay recording and the reset sampler."""

from __future__ import annotations

import pytest
import torch

from supermimic.vehicles import tracking
from supermimic.vehicles.dynamics import DELTA, VX, VY, YAW, X, Y
from supermimic.vehicles.env import XI_DIM, ReferenceLibrary, VehicleEnvCfg, VehicleTrackingEnv

STEPS = 300


@pytest.fixture(scope="module")
def lib():
    return ReferenceLibrary.generate(VehicleEnvCfg(), num=6, steps=STEPS, seed=0)


def _nominal_env(lib, n=6, **kw):
    cfg = VehicleEnvCfg(num_envs=n, domain_randomization=False, reset_noise=False, episode_steps=10_000, replay_capacity=400, **kw)
    return VehicleTrackingEnv(cfg, lib)


def test_references_are_stable_and_laid_out(lib):
    assert lib.layout.num_sequences == 6 and lib.layout.num_steps == 6 * STEPS
    assert float(lib.beta.abs().max()) < 0.12 and float(lib.v.min()) > 4.0
    assert float((lib.v * lib.r).abs().max()) > 3.0  # they do corner


def test_replaying_the_reference_actions_tracks_perfectly(lib):
    """References come from the same actuator + dynamics pipeline: under the nominal context, no delay and
    no noise, replaying their actions reproduces them (near-maximal reward, no termination)."""
    env = _nominal_env(lib)
    env.forced_frame[:] = lib.layout.seq_start
    env.reset()
    total, terminations = torch.zeros(6), 0
    for t in range(STEPS - 10):
        _, r, term, _, _info = env.step(lib.actions[lib.layout.seq_start + t])
        total += r
        terminations += int(term.sum())
    assert terminations == 0
    assert float(total.min()) / (STEPS - 10) > 0.999 * env.cfg.tracking.max_step_reward


def test_termination_is_relative_to_the_reference():
    cfg = tracking.TrackingCfg()
    n = 4
    s = torch.zeros(n, 17)
    s[:, VX] = 15.0
    ref = tracking.VehicleRef(torch.zeros(n), torch.zeros(n), torch.zeros(n), torch.full((n,), 15.0), torch.zeros(n), torch.zeros(n))
    # 0: on the reference; 1: 3 m to the side; 2: drifting (beta 0.3) against a drift reference; 3: same drift, straight reference
    s[1, Y] = 3.0
    s[2:, VX], s[2:, VY] = 15.0 * torch.cos(torch.tensor(0.3)), 15.0 * torch.sin(torch.tensor(0.3))
    ref.beta[2] = 0.3
    assert tracking.terminated(s, ref, cfg).tolist() == [False, True, False, False]
    ref.beta[3] = 0.0
    s[3, VY] = 15.0 * torch.sin(torch.tensor(0.4))
    assert bool(tracking.terminated(s[3:], tracking.VehicleRef(*(f[3:] for f in (ref.x, ref.y, ref.yaw, ref.v, ref.beta, ref.r))), cfg)[0])


def test_command_delay_shifts_the_steering_response(lib):
    """Two rows per delay, with and without a steering step at t = 1: their difference is the step's effect,
    which appears at once without delay and three steps later with a delay of three."""
    effect = {}
    for delay in (0, 3):
        env = _nominal_env(lib, n=2)
        env.forced_xi = torch.tensor([1.0, 1.0, 1.0, 0.0, float(delay)])
        env.forced_frame[:] = lib.layout.seq_start[0]
        env.reset()
        base = lib.actions[lib.layout.seq_start[0]].expand(2, -1).clone()
        diff = []
        for t in range(8):
            a = base.clone()
            a[0, 0] += 0.5 if t >= 1 else 0.0
            env.step(a)
            diff.append(float(env.s[0, DELTA] - env.s[1, DELTA]))
        effect[delay] = torch.tensor(diff)
    e0, e3 = effect[0], effect[3]
    assert e0[0] == 0 and e0[1] > 1e-3
    assert torch.equal(e3[:4], torch.zeros(4)) and e3[4] > 1e-3


def test_context_sampling_within_ranges(lib):
    cfg = VehicleEnvCfg(num_envs=256, replay_capacity=50)
    env = VehicleTrackingEnv(cfg, lib)
    xi = env.xi
    assert xi.shape == (256, XI_DIM)
    for k, (lo, hi) in enumerate((cfg.mu_range, cfg.tire_stiffness_range, cfg.mass_range, cfg.cog_shift_range)):
        assert float(xi[:, k].min()) >= lo and float(xi[:, k].max()) <= hi and float(xi[:, k].std()) > 0
    assert set(xi[:, 4].long().tolist()) == set(range(cfg.max_delay + 1))
    torch.testing.assert_close(env.ctx.mu, xi[:, 0])
    assert torch.equal(env.act.delay, xi[:, 4].long())


def test_replay_rows_windows_and_terminal_rows(lib):
    """Transitions land in the replay buffer as the G1 recorder writes them: consecutive rows chain (the
    next row's state is the state the previous action reached) and every episode ends in a terminal row
    with its termination label."""
    cfg = VehicleEnvCfg(num_envs=32, replay_capacity=200, episode_steps=60)
    env = VehicleTrackingEnv(cfg, lib, seed=1)
    torch.manual_seed(0)
    terms = 0
    for _ in range(150):
        a = 0.5 * torch.randn(32, 2)
        _, _, term, _, _info = env.step(a, action_noise=0.1 * torch.randn(32, 2))
        terms += int(term.sum())
    buf = env.buffer
    assert terms > 0 and int(buf.terminal.sum()) > terms  # time-outs and reference ends close episodes too
    # each termination labels the last transition row and the terminal row (no row was overwritten yet)
    assert int((buf.done[buf.terminal] > 0).sum()) == terms
    assert int((buf.done[~buf.terminal & buf.valid] > 0).sum()) == terms
    w = buf.sample_windows(64, 8)
    assert w.states.shape == (9, 64, cfg.state_dim) and w.a_exec.shape == (8, 64, 2)
    # chaining: inside a window the frame advances by one per step until the reference ends
    step = (w.frame[1:] - w.frame[:-1])
    assert bool(((step == 1) | (step == 0)).all())
    st = buf.sample_starts(64, 10)
    assert st.hist_states.shape == (10, 64, cfg.state_dim) and st.xi.shape == (64, XI_DIM)


def test_reset_sampler_learns_from_episodes(lib):
    cfg = VehicleEnvCfg(num_envs=64, replay_capacity=50, episode_steps=40)
    env = VehicleTrackingEnv(cfg, lib, seed=2)
    g_max = cfg.sampler.g_max
    for _ in range(120):
        env.step(torch.zeros(64, 2))  # a do-nothing policy fails on cornering references
    assert float(env.sampler.bin_return.min()) < g_max  # bins were scored
    f, ent = env.sampler.sample(1000)
    assert ent is not None and bool((f <= lib.layout.last_of(f)).all())


def test_observation_and_goal(lib):
    env = _nominal_env(lib, n=3)
    env.forced_frame[:] = lib.layout.seq_start[:3]
    obs = env.reset()
    assert obs["policy"].shape == (3, 12) and obs["goal"].shape == (3, 5 * len(env.cfg.preview))
    # on the reference, the first preview point is the next reference step: close ahead, little lateral offset
    g = obs["goal"].reshape(3, -1, 5)
    v = lib.v[lib.layout.seq_start[:3]]
    torch.testing.assert_close(g[:, 0, 0], v * env.cfg.dt, rtol=0.1, atol=0.05)
    assert float(g[:, 0, 1].abs().max()) < 0.05
    st = env.state()
    assert st.shape == (3, env.cfg.state_dim)
    torch.testing.assert_close(st[:, X], lib.x[lib.layout.seq_start[:3]])
    torch.testing.assert_close(st[:, YAW], lib.yaw[lib.layout.seq_start[:3]])


def test_friction_matched_references_and_episodes():
    """References generated under a friction carry it (seq_mu) and their friction use (util); with
    match_friction every episode runs at its reference's friction."""
    from supermimic.vehicles.dynamics import AssistCfg

    cfg = VehicleEnvCfg(assist=AssistCfg(abs=False, esc=False))
    parts = [ReferenceLibrary.generate(cfg, num=2, steps=120, seed=k, mu=mu, ay_frac=(0.4, 0.95), oversample=4.0) for k, mu in enumerate((0.5, 1.0))]
    lib = ReferenceLibrary.concat(parts)
    assert lib.seq_mu == (0.5, 0.5, 1.0, 1.0) and lib.layout.num_sequences == 4
    assert lib.util.shape == lib.x.shape and float(lib.util.min()) >= 0.0
    assert float(parts[0].util.max()) > 0.3  # they use a good part of the low friction
    env = VehicleTrackingEnv(VehicleEnvCfg(num_envs=40, replay_capacity=20, match_friction=True, assist=cfg.assist), lib, seed=0)
    ref_mu = lib.seq_mu_t[lib.layout.step_seq[env.frame - 1]]
    assert torch.equal(env.xi[:, 0], ref_mu) and torch.equal(env.ctx.mu, ref_mu)
    assert set(env.xi[:, 0].tolist()) == {0.5, 1.0}
    assert float(env.xi[:, 1].std()) > 0  # the other context parameters stay randomised
    plain = ReferenceLibrary(**{k: getattr(lib, k) for k in ReferenceLibrary._FIELDS}, lengths=lib.lengths)
    with pytest.raises(ValueError):
        VehicleTrackingEnv(VehicleEnvCfg(num_envs=4, replay_capacity=10, match_friction=True), plain)


def test_default_references_record_nominal_friction(lib):
    assert lib.seq_mu == (1.0,) * lib.layout.num_sequences and lib.util is not None


def test_raw_actions_are_bounded_in_state_and_penalised_beyond_saturation(lib):
    """tanh saturates, so a policy's raw outputs can drift without limit (the limit study's divergence):
    the actuators hold them clamped to +-action_clip, and the reward penalises them beyond action_bound."""
    env = _nominal_env(lib, n=4)
    env.reset()
    big = torch.tensor([[300.0, -300.0], [3.0, 0.0], [2.0, -2.0], [0.0, 0.0]])
    _, r, _, _, info = env.step(big)
    pend = env.state()[:, 18:18 + 2 * env.cfg.max_delay]
    assert float(pend.abs().max()) <= env.cfg.action_clip
    pen = info["reward_terms"]["action_bound"]
    cfg = env.cfg.tracking
    assert float(pen[0]) < -100 * cfg.dt and float(pen[1]) < 0 and float(pen[2]) == 0 and float(pen[3]) == 0
    torch.testing.assert_close(pen[1], torch.tensor(cfg.w_action_bound * (3.0 - cfg.action_bound) ** 2 * cfg.dt))


def test_non_finite_dynamics_end_the_episode_on_the_last_finite_state(lib):
    env = _nominal_env(lib, n=3)
    env.reset()
    before = env.s.clone()
    orig = env.dyn.step

    def broken_step(s, flags, sr, ac, ctx):
        s2, f2 = orig(s, flags, sr, ac, ctx)
        s2 = s2.clone()
        s2[1, 3] = float("nan")
        return s2, f2

    env.dyn.step = broken_step
    _, r, term, _, info = env.step(torch.zeros(3, 2))
    assert term.tolist() == [False, True, False] and bool(torch.isfinite(r).all())
    rows = env.buffer.s[env.buffer.valid]
    assert bool(torch.isfinite(rows).all())
    assert torch.equal(env.buffer.s[1][env.buffer.terminal[1]][0, :17], before[1])
