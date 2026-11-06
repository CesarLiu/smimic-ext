"""PPO baseline: GAE against a hand computation, and the SuperMimic-vs-PPO comparison script end to end."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

from supermimic.baselines.ppo import PPO, PPOCfg

ROOT = Path(__file__).resolve().parents[1]


def test_gae_matches_hand_computation():
    cfg = PPOCfg(gamma=0.9, lam=0.5)
    agent = PPO(torch.nn.Linear(1, 1), torch.nn.Linear(1, 1), 1, cfg, "cpu")
    r = torch.tensor([[1.0], [0.0], [2.0]])
    v = torch.tensor([[0.5], [0.2], [0.1]])
    d = torch.tensor([[0.0], [1.0], [0.0]])  # the episode ends after step 1
    last = torch.tensor([0.3])
    adv, ret = agent.gae(r, v, d, last)
    d2 = 2.0 + 0.9 * 0.3 - 0.1
    d1 = 0.0 - 0.2  # terminal: no bootstrap
    d0 = 1.0 + 0.9 * 0.2 - 0.5
    expect = torch.tensor([[d0 + 0.45 * d1], [d1], [d2]])
    torch.testing.assert_close(adv, expect)
    torch.testing.assert_close(ret, expect + v)


def test_compare_ppo_runs(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts" / "vehicle"))
    import compare_ppo

    code = compare_ppo.main(["--budget-min", "0.05", "--eval-every-s", "2", "--mu-levels", "0.5", "1.0", "--refs-per-level", "2",
                             "--eval-refs-per-level", "1", "--ref-steps", "120", "--episode-steps", "40", "--num-envs", "16", "--batch", "16",
                             "--warmup", "20", "--policy-warmup", "30", "--horizon", "4", "--ppo-envs", "16", "--ppo-steps", "8",
                             "--methods", "supermimic", "ppo", "--out-dir", str(tmp_path), "--no-plots"])
    rep = json.loads((tmp_path / "report.json").read_text())
    assert code == 0 and set(rep["methods"]) == {"supermimic", "ppo"}
    for res in rep["methods"].values():
        assert len(res["curve"]) >= 2 and res["env_steps"] > 0
        assert all(math.isfinite(v["normalized_return"]) for v in res["scurve_final"].values())
        assert set(res["scurve_traj"]) == {"mu_0.5", "mu_1.0"}


def test_render_scurve(tmp_path):
    """The evaluation video renders from compare_ppo's policies.pt (a few frames, GIF, no ffmpeg needed)."""
    sys.path.insert(0, str(ROOT / "scripts" / "vehicle"))
    import compare_ppo
    import render_scurve

    compare_ppo.main(["--budget-min", "0.02", "--eval-every-s", "100", "--eval-at-steps", "--mu-levels", "1.0", "--refs-per-level", "2",
                      "--eval-refs-per-level", "1", "--ref-steps", "120", "--episode-steps", "40", "--num-envs", "16", "--batch", "16",
                      "--warmup", "20", "--policy-warmup", "30", "--horizon", "4", "--ppo-envs", "16", "--ppo-steps", "8",
                      "--methods", "supermimic", "ppo", "--out-dir", str(tmp_path), "--no-plots"])
    out = tmp_path / "s.gif"
    assert render_scurve.main(["--policies", str(tmp_path / "policies.pt"), "--fps", "1", "--dpi", "40", "--out", str(out)]) == 0
    assert out.exists() and out.stat().st_size > 1000
