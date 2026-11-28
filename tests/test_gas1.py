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
    # the extra training-objective terms: zero weights change nothing, positive weights only subtract
    with torch.no_grad():
        J0 = imagine_g(wm, pol, st, paths, 10, roads)
        assert torch.equal(J0, imagine_g(wm, pol, st, paths, 10, roads, 0.0, 0.0))
        assert (imagine_g(wm, pol, st, paths, 10, roads, 1.0, 50.0) <= J0 + 1e-6).all()
        assert (imagine_g(wm, pol, st, paths, 10, roads, 0.0, 0.0, 1.0) <= J0 + 1e-6).all()
        # an ensemble of identical members has no disagreement: with the penalty it reproduces the single model
        import copy

        from chrono_rollover_control import WMEnsemble

        ens = WMEnsemble([wm, copy.deepcopy(wm), copy.deepcopy(wm)])
        assert torch.allclose(imagine_g(ens, pol, st, paths, 10, roads, unc_penalty=10.0), J0, atol=1e-4)


def test_actor_critic_imagination(tmp_path):
    """imagine_ac: finite return, gradient to the policy but not to the frozen critic; with every step an event the
    return is r_0 + gamma (1 - d_0) V(s_1); critic inputs and TD(lambda) targets line up."""
    from chrono_gas1 import (
        STATE_G,
        Critic,
        Policy6,
        Roads,
        features_g,
        imagine_ac,
        sample_starts_g,
        train_critic,
    )
    from chrono_lift_check import LiftWM
    from chrono_rollover_control import lane_change_path, placed

    roads = Roads([3], tmp_path, GEO)
    rng = np.random.default_rng(1)
    nx, T = len(STATE_G), 40
    path = placed(lane_change_path(rng)[0], 30.0, 0.0, 0.0)
    logs = [{"x": rng.standard_normal((T + 1, nx)) * 0.1 + np.r_[13.0, np.zeros(nx - 1)], "a": rng.standard_normal((T, 6)).clip(-1, 1),
             "u": rng.standard_normal((T, 10)) * 0.1, "pose": np.c_[30 + 13 * 0.02 * np.arange(T + 1), np.zeros((T + 1, 2))],
             "s": list(13 * 0.02 * np.arange(T + 1)), "v_ref": 13.0, "path": path, "rid": 0}]
    torch.manual_seed(0)
    X = np.concatenate([L["x"] for L in logs])
    wm = LiftWM(nx, 10, [32], X.mean(0), X.std(0) + 1e-3, np.ones(nx) * 0.01, hist=2)
    st, paths = sample_starts_g(logs, 6, rng, 2)
    n_in = features_g(st[0][-1], st[2], st[3], st[4], st[5], st[1][-1][:, :6], paths, wm, roads, st[6])[0].shape[-1]
    pol, critic = Policy6(n_in), Critic(n_in)
    critic.requires_grad_(False)
    G, Xc, Yc, share = imagine_ac(wm, pol, critic, st, paths, 8, roads, event_ltr=10.0)
    assert G.shape == (6,) and torch.isfinite(G).all() and Xc.shape == (8, 6, n_in) and Yc.shape == (8, 6) and share == 0.0
    G.mean().backward()
    assert all(p.grad is not None for p in pol.parameters()) and all(p.grad is None for p in critic.parameters())
    critic.requires_grad_(True)
    assert np.isfinite(train_critic(critic, torch.optim.Adam(critic.parameters(), 1e-3), Xc, Yc, epochs=1, minibatches=2))
    # every step an event: only the first reward and the critic's value of the second state remain
    with torch.no_grad():
        G_cut, _, _, share2 = imagine_ac(wm, pol, critic, st, paths, 8, roads, gamma=0.9, event_ltr=-1.0)
        G1, _, _, _ = imagine_ac(wm, pol, critic, st, paths, 1, roads, gamma=0.9, event_ltr=10.0)
    assert share2 == 1.0 and torch.allclose(G_cut, G1, atol=1e-5)
