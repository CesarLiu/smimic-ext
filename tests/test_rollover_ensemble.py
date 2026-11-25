"""``chrono_rollover_control.WMEnsemble``: identical members reproduce the single model with zero disagreement, so
imagination with the uncertainty penalty equals imagination without it; different members disagree."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vehicle"))
pytest.importorskip("casadi")


def _member(seed, hist=3, X=16):
    from chrono_lift_check import LiftWM

    torch.manual_seed(seed)
    return LiftWM(X, 2, [32], np.zeros(X), np.ones(X), np.ones(X), hist=hist)


def test_identical_members_equal_single_model():
    from chrono_rollover_control import WMEnsemble

    m = _member(0)
    ens = WMEnsemble([m, copy.deepcopy(m), copy.deepcopy(m)])
    Xh, Ah = torch.randn(3, 5, 16), torch.randn(3, 5, 2)
    x, spread = ens.step_hist_spread(Xh, Ah)
    assert torch.allclose(x, m.step_hist(Xh, Ah), atol=1e-6) and torch.allclose(spread, torch.zeros(5), atol=1e-10)


def test_different_members_disagree_and_penalty_lowers_return():
    from chrono_rollover_control import WMEnsemble, imagine, sample_starts

    ens = WMEnsemble([_member(0), _member(1), _member(2)])
    Xh, Ah = torch.randn(3, 5, 16), torch.randn(3, 5, 2)
    assert (ens.step_hist_spread(Xh, Ah)[1] > 0).all()
    rng = np.random.default_rng(0)
    T = 30
    logs = [{"x": rng.standard_normal((T + 1, 16)).tolist(), "a": rng.standard_normal((T, 2)).tolist(), "pose": np.zeros((T + 1, 3)).tolist(),
             "s": list(np.arange(T + 1) * 0.5), "v_ref": 12.0, "path": np.zeros((50, 4))} for _ in range(2)]
    st, paths = sample_starts(logs, 4, rng, hist=3)

    def policy(ob):
        return torch.zeros(ob.shape[0], 2)

    J0 = imagine(ens, policy, st, paths, 5)
    J1 = imagine(ens, policy, st, paths, 5, unc_penalty=10.0)
    assert (J1 < J0).all()
