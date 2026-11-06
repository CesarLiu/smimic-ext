"""Elk test: track geometry and cone scoring (no solver needed), and the demo script end to end from the cached
references (the NMPC-generated references are committed in docs/results, so only the policies need to run)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from supermimic.vehicles.elk import ElkTrack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "vehicle"))


def _track():
    tr = ElkTrack()
    tr.x0 = 10.0
    return tr


def test_lanes_follow_the_layout():
    tr = _track()
    lanes = tr.lanes()
    assert [round(b - a, 1) for a, b, _c, _w in lanes] == [12.0, 11.0, 12.0]  # lane lengths
    assert lanes[1][0] - lanes[0][1] == pytest.approx(13.5) and lanes[2][0] - lanes[1][1] == pytest.approx(12.5)  # transitions
    assert [c for _a, _b, c, _w in lanes] == [0.0, 1.0, 0.0]
    assert [round(w, 3) for *_r, w in lanes] == [round(f * 1.8 + 0.25, 3) for f in (1.1, 1.2, 1.3)]
    cones = tr.cones()
    assert len(cones) % 2 == 0 and all(abs(abs(y - c) - w / 2) < 1e-9 for (x, y) in cones for (a, b, c, w) in lanes if a <= x <= b)


def test_cone_scoring():
    tr = _track()
    xs = [10.0 + 0.5 * k for k in range(0, 30)]
    clean = tr.hits(xs, [0.0] * len(xs), [0.0] * len(xs))
    assert sum(clean) == 0  # centred in the first lane
    off = tr.hits(xs, [0.5] * len(xs), [0.0] * len(xs))  # 0.5 m off-centre: a body corner is out of the 2.23 m lane
    assert sum(1 for h in off if h) > 0
    # a yawed car sticks out sooner than a straight one at the same offset
    yawed = tr.hits(xs, [0.1] * len(xs), [0.3] * len(xs))
    straight = tr.hits(xs, [0.1] * len(xs), [0.0] * len(xs))
    assert sum(yawed) > sum(straight)
    # nothing is scored in the free transition zone between lanes
    gap = tr.hits([26.0, 30.0], [3.0, 3.0], [0.0, 0.0])  # both ends of the body are between lane 1 (ends at 22) and lane 2 (starts at 35.5)
    assert gap == [0, 0]


def test_cached_references_pass_their_own_cones():
    import torch
    from elk_test import load_ref

    blob = torch.load(ROOT / "docs" / "results" / "vehicle_elk_refs.pt", map_location="cpu")
    assert blob
    for key in blob:
        mu, kmh = (float(v) for v in key.split(":"))
        lib, tr = load_ref(blob, mu, kmh, "cpu")
        assert sum(1 for h in tr.hits(lib.x.tolist(), lib.y.tolist(), lib.yaw.tolist()) if h) == 0, key
        assert abs(float(lib.vx[0]) * 3.6 - kmh) < 0.5 and math.isfinite(float(lib.util.max()))


def test_elk_script_with_untrained_policies(tmp_path):
    """Runs the whole demo (cases, cone scoring, report, plot) for freshly initialised policies at one case; they
    drive off or hit cones, which is fine, the pipeline is what is tested."""
    import torch
    from compare_ppo import networks

    import compare_ppo
    import elk_test
    from limit_study import env_cfg
    from supermimic.vehicles.env import VehicleTrackingEnv

    run = compare_ppo.parse(["--episode-steps", "40"])
    cfg = env_cfg(run, 2, 2)
    blob = torch.load(ROOT / "docs" / "results" / "vehicle_elk_refs.pt", map_location="cpu")
    lib, _tr = elk_test.load_ref(blob, 1.0, 60, "cpu")
    pol = networks(run, VehicleTrackingEnv(cfg, lib, record=False), lib)[5]
    torch.save({"args": vars(run), "policies": {"ppo": {k: v for k, v in pol.state_dict().items()}}}, tmp_path / "p.pt")
    code = elk_test.main(["--policies", str(tmp_path / "p.pt"), "--methods", "ppo", "--dry-kmh", "60", "--wet-kmh", "--out-dir", str(tmp_path), "--no-video"])
    assert code == 0
    import json

    rep = json.loads((tmp_path / "report.json").read_text())
    assert len(rep["cases"]) == 1 and "ppo" in rep["cases"][0]["results"] and (tmp_path / "elk_paths.png").exists()
