"""``chrono_lift_check.LiftWM`` with any history length: the two-state interface is the hist = 2 case, windows and
batches line up with the predictions, and ``chrono_rollover_control.sample_starts`` returns matching histories."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vehicle"))


def _eps(n=3, T=40, X=16, seed=0):
    rng = np.random.default_rng(seed)
    return [{"x": rng.standard_normal((T + 1, X)), "actions": rng.standard_normal((T, 2)), "tilt": np.zeros(T + 1)} for _ in range(n)]


def _wm(hist, X=16):
    from chrono_lift_check import LiftWM

    torch.manual_seed(0)
    return LiftWM(X, 2, [32], np.zeros(X), np.ones(X), np.ones(X), hist=hist)


def test_two_state_interface_is_hist_2():
    wm = _wm(2)
    x, xp, a, ap = torch.randn(5, 16), torch.randn(5, 16), torch.randn(5, 2), torch.randn(5, 2)
    assert torch.allclose(wm.step(x, xp, a, ap), wm.step_hist(torch.stack([xp, x]), torch.stack([ap, a])))
    acts = torch.randn(4, 5, 2)
    manual, X0, Xp, Ap = [], x, xp, ap
    for t in range(4):
        X1 = wm.step(X0, Xp, acts[t], Ap)
        manual.append(X1)
        Xp, X0, Ap = X0, X1, acts[t]
    assert torch.allclose(wm.rollout(x, xp, acts, ap), torch.stack(manual))


@pytest.mark.parametrize("hist", [2, 5])
def test_windows_batches_and_predictions_line_up(hist):
    from chrono_lift_check import batch, predict, windows

    eps, k = _eps(), 8
    idx = windows(eps, k, hist)
    assert min(t for _, t in idx) == hist - 1 and max(t for _, t in idx) == len(eps[0]["x"]) - k - 1
    Xb, Ab = batch(eps, idx[:7], k, hist)
    assert Xb.shape == (hist + k, 7, 16) and Ab.shape == (hist - 1 + k, 7, 2)
    e, t = idx[3]
    assert np.allclose(Xb[hist - 1, 3].numpy(), eps[e]["x"][t]) and np.allclose(Ab[hist - 1, 3].numpy(), eps[e]["actions"][t])
    pred = predict(_wm(hist), Xb, Ab)
    assert pred.shape == (k, 7, 16) and torch.isfinite(pred).all()


def test_sample_starts_histories():
    pytest.importorskip("casadi")  # chrono_rollover_control imports the NMPC baseline through chrono_g1-style helpers
    from chrono_rollover_control import sample_starts

    rng = np.random.default_rng(0)
    T = 30
    logs = [{"x": rng.standard_normal((T + 1, 16)).tolist(), "a": rng.standard_normal((T, 2)).tolist(), "pose": rng.standard_normal((T + 1, 3)).tolist(),
             "s": list(np.arange(T) * 0.5), "v_ref": 12.0, "path": np.zeros((50, 4))} for _ in range(2)]
    (Xh, Ah, pose, s, pid, _v_ref), paths = sample_starts(logs, 9, rng, hist=4)
    assert Xh.shape == (4, 9, 16) and Ah.shape == (3, 9, 2) and pose.shape == (9, 3) and paths.shape[0] == 2
    # the newest state of each history is the start state itself, the newest past action the one just before it
    for j in range(9):
        L = logs[int(pid[j])]
        t = round(float(s[j]) / 0.5)
        assert np.allclose(Xh[-1, j].numpy(), np.array(L["x"][t], dtype=np.float32)) and np.allclose(Ah[-1, j].numpy(), np.array(L["a"][t - 1], dtype=np.float32))
