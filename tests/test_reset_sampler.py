"""Return normalisation of the return-based reset sampler (gmax_mode="remaining")."""

import torch

from supermimic.envs.mdp.commands import normalize_return_to_go

GAMMA, G_MAX = 0.99, 10.0
R_MAX = G_MAX * (1 - GAMMA)  # per-step reward bound


def _rtg(rewards: list[float]) -> float:
    return sum(r * GAMMA**i for i, r in enumerate(rewards))


def test_perfect_tracking_scores_g_max_at_any_distance_to_the_clip_end():
    for m in (1, 3, 50, 500):
        rtg = torch.tensor([_rtg([R_MAX] * m)])
        out = normalize_return_to_go(rtg, torch.tensor([m]), GAMMA, G_MAX)
        assert abs(float(out) - G_MAX) < 1e-4


def test_a_clip_tail_no_longer_looks_harder_than_a_failure():
    # 5 perfect steps until the clip ends vs 20 perfect steps then a fall with 400 reference steps left.
    tail = torch.tensor([_rtg([R_MAX] * 5)])
    fall = torch.tensor([_rtg([R_MAX] * 20)])
    assert float(tail) < float(fall)  # the constant-g_max ranking (plan D14) calls the tail harder
    out_tail = normalize_return_to_go(tail, torch.tensor([5]), GAMMA, G_MAX)
    out_fall = normalize_return_to_go(fall, torch.tensor([400]), GAMMA, G_MAX)
    assert float(out_tail) > 9.99
    assert float(out_fall) < 2.0
    assert float(out_fall) < float(out_tail)


def test_shape_dtype_and_clamp():
    rtg = torch.tensor([[1.0, 50.0], [0.0, -1.0]])
    remaining = torch.tensor([[10, 10], [0, 7]])
    out = normalize_return_to_go(rtg, remaining, GAMMA, G_MAX)
    assert out.shape == rtg.shape and out.dtype == rtg.dtype
    assert float(out.max()) <= G_MAX and float(out.min()) >= 0.0
