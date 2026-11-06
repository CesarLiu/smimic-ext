"""NMPC baseline: tyre identification, closed-loop tracking of the S-bend, and its place in the comparison
scripts (skipped without casadi)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("casadi")

from supermimic.baselines.nmpc import NMPC, AxleTyre, fit_axle, identify_axle_tyres  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "vehicle"))


def test_fit_recovers_magic_formula():
    true = AxleTyre(12.0, 1.4, -7000.0, 0.3)
    a = np.linspace(-0.3, 0.3, 400)
    fit = fit_axle(a, true.force(a))
    np.testing.assert_allclose(fit.force(a), true.force(a), atol=30.0)


def test_identified_tyres_are_physical():
    front, rear, info = identify_axle_tyres()
    assert info["rel_rms_front"] < 0.1 and info["rel_rms_rear"] < 0.1
    for t in (front, rear):
        assert 0.8 <= t.C <= 2.5
        assert abs(t.force(0.02)) < abs(t.force(0.1))  # rises to a peak
    # friction scales the peak, not the cornering stiffness
    assert abs(front.force(0.3, 0.5)) < 0.6 * abs(front.force(0.3, 1.0))
    assert front.force(1e-4, 0.5) == pytest.approx(front.force(1e-4, 1.0), rel=1e-3)


def test_nmpc_tracks_s_bend():
    from limit_study import env_cfg, evaluate

    from supermimic.vehicles.env import ReferenceLibrary
    from supermimic.vehicles.learner import VehicleHeads
    from supermimic.vehicles.params import VehicleParams

    args = types.SimpleNamespace(episode_steps=250, device="cpu", history=5)
    cfg = env_cfg(args, 2, 2)
    ctl = NMPC(cfg)
    lib = ReferenceLibrary.s_curve(cfg, speed=15.0, ay_frac=0.7, mu=0.5)
    r = evaluate(None, VehicleHeads(cfg, VehicleParams().R_w), lib, args, controller=ctl)
    assert r["termination_rate"] == 0.0 and r["normalized_return"] > 0.9
    assert ctl.timing()["failures"] == 0


def test_compare_and_render_with_nmpc(tmp_path):
    import compare_ppo
    import render_scurve

    code = compare_ppo.main(["--methods", "nmpc", "--mu-levels", "1.0", "--eval-refs-per-level", "1", "--refs-per-level", "1",
                             "--ref-steps", "60", "--scurve-speed", "12", "--out-dir", str(tmp_path), "--no-plots"])
    rep = json.loads((tmp_path / "report.json").read_text())
    assert code == 0 and set(rep["methods"]) == {"nmpc"}
    res = rep["methods"]["nmpc"]
    assert res["latency"]["solves"] > 0 and res["scurve_traj"]["mu_1.0"]["steps"] > 0
    out = tmp_path / "s.gif"
    assert render_scurve.main(["--policies", str(tmp_path / "policies.pt"), "--fps", "1", "--dpi", "40", "--out", str(out)]) == 0
    assert out.exists() and out.stat().st_size > 1000


def _env_and_lib(mu, steps_args=None):
    import types

    from limit_study import env_cfg

    from supermimic.vehicles.env import ReferenceLibrary, VehicleTrackingEnv

    args = types.SimpleNamespace(episode_steps=250, device="cpu", history=5, eval_noise=True, eval_delay=2)
    cfg = env_cfg(args, 1, 2, train=False)
    lib = ReferenceLibrary.s_curve(cfg, speed=15.0, ay_frac=0.7, mu=mu)
    return args, cfg, lib, VehicleTrackingEnv(cfg, lib, record=False)


def test_estimator_tracks_sideslip_and_friction():
    """Driving the S-bend with the reference's own commands, the estimator sees vy from IMU integration and, once
    the tyres are near the limit, friction below the prior 1.0 when the road is slippery (and stays near 1 when not)."""
    import torch
    from limit_study import eval_xi, measurement

    from supermimic.vehicles.dynamics import VY
    from supermimic.vehicles.learner import VehicleHeads
    from supermimic.vehicles.params import VehicleParams

    heads = VehicleHeads(_env_and_lib(1.0)[1], VehicleParams().R_w)
    mu_hat, vy_err = {}, {}
    for mu in (0.5, 1.0):
        args, cfg, lib, env = _env_and_lib(mu)
        ctl = NMPC(cfg, cfg=__import__("supermimic.baselines.nmpc", fromlist=["NMPCCfg"]).NMPCCfg(information="onboard"))
        env.forced_xi = eval_xi(args)
        env.forced_frame[:] = lib.layout.seq_start[:1]
        env.reset()
        errs = []
        for k in range(300):
            meas = measurement(env.observation(), heads, args).double().numpy()[0]
            s = env.s[0].double().numpy()
            x0, mhat = ctl.est.update(0, meas, s[0], s[1], s[2], fresh=(k == 0))
            if k > 50:
                errs.append(abs(x0[4] - float(env.s[0, VY])))
            env.step(lib.actions[min(int(env.frame[0]), len(lib.actions) - 1)][None])
        mu_hat[mu], vy_err[mu] = mhat, max(errs)
    assert all(e < 0.15 for e in vy_err.values())
    assert mu_hat[0.5] < 0.75 and mu_hat[1.0] > 0.8
    torch.manual_seed(0)


def test_onboard_nmpc_with_noise_and_delay():
    """Without being told the friction or the delay, with measurement noise and a 2-step command delay, the
    on-board NMPC (assuming 1 step of delay) still completes the S-bend at mu 0.5."""
    from limit_study import evaluate

    from supermimic.baselines.nmpc import NMPCCfg
    from supermimic.vehicles.learner import VehicleHeads
    from supermimic.vehicles.params import VehicleParams

    args, cfg, lib, _env = _env_and_lib(0.5)
    ctl = NMPC(cfg, cfg=NMPCCfg(information="onboard", assumed_delay=1))
    r = evaluate(None, VehicleHeads(cfg, VehicleParams().R_w), lib, args, controller=ctl)
    assert r["termination_rate"] == 0.0 and r["normalized_return"] > 0.9
