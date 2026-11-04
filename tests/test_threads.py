"""CPU thread limits of the vehicle scripts (``supermimic.vehicles.threads``)."""

from __future__ import annotations

import torch

from supermimic.vehicles import threads


def test_default_is_capped_on_many_core_machines(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setattr(threads, "usable_cores", lambda: 192)
    assert threads.default_threads() == 8
    monkeypatch.setattr(threads, "usable_cores", lambda: 4)
    assert threads.default_threads() == 4
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert threads.default_threads() == 3  # the user's setting wins


def test_limit_threads_applies_to_torch(monkeypatch):
    before = torch.get_num_threads()
    try:
        assert threads.limit_threads(2) == 2 and torch.get_num_threads() == 2
        import os

        assert os.environ["OMP_NUM_THREADS"] == "2"
    finally:
        threads.limit_threads(before)


def test_every_vehicle_script_takes_threads():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "scripts" / "vehicle"
    sys.path.insert(0, str(root))
    import demo_e2e
    import jacobian_check
    import limit_study

    for mod in (demo_e2e, jacobian_check, limit_study):
        assert mod.parse(["--threads", "3"]).threads == 3
    for name in ("bench_sim.py",):
        assert "limit_threads(args.threads)" in (root / name).read_text()
