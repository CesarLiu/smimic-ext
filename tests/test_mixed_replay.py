"""``MixedReplay``: fixed shares of world-model windows and imagination starts from a second buffer, same batch
layout as ``ReplayBuffer``; buffers survive a save / load round trip."""

from __future__ import annotations

import torch

from supermimic.data.replay_buffer import MixedReplay, load_buffer, save_buffer
from supermimic.vehicles.env import ReferenceLibrary, VehicleEnvCfg, VehicleTrackingEnv


def _filled(seed, n=8, steps=30, chassis=True):
    torch.manual_seed(seed)
    cfg = VehicleEnvCfg(num_envs=n, chassis=chassis, replay_capacity=40)
    env = VehicleTrackingEnv(cfg, ReferenceLibrary.generate(VehicleEnvCfg(), num=2, steps=120, seed=0), seed=seed)
    for _ in range(steps):
        env.step(torch.randn(n, cfg.action_dim))
    return env.buffer


def test_mixed_batches(tmp_path):
    main, extra = _filled(0), _filled(1)
    extra.xi[:] = 123.0  # tag the second buffer's rows
    mix = MixedReplay(main, extra, 0.25)
    w = mix.sample_windows(16, 4)
    ref = main.sample_windows(16, 4)
    for f in ("states", "a_exec", "a_clean", "reward", "done", "frame"):
        assert getattr(w, f).shape == getattr(ref, f).shape
    assert int((w.xi[:, 0] == 123.0).sum()) == 4 and w.xi.shape == ref.xi.shape
    s = mix.sample_starts(20, 3)
    assert int((s.xi[:, 0] == 123.0).sum()) == 5 and s.hist_states.shape[:2] == (3, 20) and s.prev_action.shape == (20, 5)
    assert int((MixedReplay(main, extra, 0.0).sample_windows(8, 4).xi[:, 0] == 123.0).sum()) == 0
    # save / load
    save_buffer(extra, tmp_path / "b.pt")
    back = load_buffer(tmp_path / "b.pt", "cpu")
    assert back.size == extra.size and torch.equal(back.s, extra.s) and torch.equal(back.ptr, extra.ptr)
    assert back.sample_windows(4, 4).states.shape == (5, 4, extra.s.shape[-1])
