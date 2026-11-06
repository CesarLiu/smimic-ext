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
