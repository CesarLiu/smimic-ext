"""G-AS1 pieces that run without Chrono: the torch road lookup matches ``Road.height``, and imagination with six
actions and road inputs gives a finite return whose gradient reaches the policy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vehicle"))
pytest.importorskip("casadi")

GEO = {"a": 1.6, "b": 1.7, "track": 1.8}


def test_torch_road_lookup_matches_numpy(tmp_path):
    from chrono_gas1 import Roads

    roads = Roads([3, 4], tmp_path, GEO)
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 200, (2, 50)), rng.uniform(-10, 10, (2, 50))
    zt = roads.height(torch.tensor([0, 1]), torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)).numpy()
    zn = np.stack([roads.items[k][0].height(x[k], y[k]) for k in range(2)])
    assert np.abs(zt - zn).max() < 1e-4
    assert np.abs(zn).max() > 0.02  # it is a rough road


def test_imagination_runs_and_reaches_policy(tmp_path):
    from chrono_gas1 import STATE_G, Policy6, Roads, features_g, imagine_g, sample_starts_g
    from chrono_lift_check import LiftWM
    from chrono_rollover_control import lane_change_path, placed

    roads = Roads([3], tmp_path, GEO)
    rng = np.random.default_rng(0)
    nx, T = len(STATE_G), 40
    path = placed(lane_change_path(rng)[0], 30.0, 0.0, 0.0)
    logs = [{"x": rng.standard_normal((T + 1, nx)) * 0.1 + np.r_[13.0, np.zeros(nx - 1)], "a": rng.standard_normal((T, 6)).clip(-1, 1),
             "u": rng.standard_normal((T, 10)) * 0.1, "pose": np.c_[30 + 13 * 0.02 * np.arange(T + 1), np.zeros((T + 1, 2))],
             "s": list(13 * 0.02 * np.arange(T + 1)), "v_ref": 13.0, "path": path, "rid": 0}]
    torch.manual_seed(0)
    X = np.concatenate([L["x"] for L in logs])
    wm = LiftWM(nx, 10, [32], X.mean(0), X.std(0) + 1e-3, np.ones(nx) * 0.01, hist=2)
    st, paths = sample_starts_g(logs, 8, rng, 2)
    pol = Policy6(features_g(st[0][-1], st[2], st[3], st[4], st[5], st[1][-1][:, :6], paths, wm, roads, st[6])[0].shape[-1])
    J = imagine_g(wm, pol, st, paths, 10, roads)
    assert J.shape == (8,) and torch.isfinite(J).all()
    J.mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pol.parameters())
