"""Pass / fail gates for scripts/run_experiments.sh (RUNBOOK). Exit code 1 if any check fails.

  uv run python scripts/check_run.py lafan --run-dir logs/lafan_ori
  uv run python scripts/check_run.py margins logs/g1/margins_*.json

``lafan``: the end-to-end check of the new code on a short LAFAN1 run (RUNBOOK step 2): return, drift-free
body error, yaw drift, the world-model error script against the training log, and the margin controls.
``margins``: the validity controls of every margin run (drift rates, scale-0 failures).
The retest gate (same margins under a shuffled branch-to-env assignment) is scripts/retest_margins.py.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import tyro

Check = tuple[str, float, bool, str]  # name, value, passed, expectation


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def last_logged_nse8(train_log: Path) -> float:
    """Last finite ``nse8`` value printed by the training loop (wm/online_nse_8step)."""
    vals = [float(v) for v in re.findall(r"nse8 ([0-9.eE+-]+)", Path(train_log).read_text())]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        raise ValueError(f"no finite nse8 value in {train_log}")
    return vals[-1]


def margin_checks(margins: dict, max_over_tol: float = 0.01, max_event_rate: float = 1e-4, max_zero_fail: float = 0.01, tag: str = "") -> list[Check]:
    """Drift is gated by rates, not its maximum (eval_margins.py docstring): an event rate of 1e-4 per env-step
    hits ~1% of 100-step branch windows. Old outputs without the rate fields fail (nan)."""
    s = margins["summary"]
    over_tol, events = s.get("drift_frac_over_tol", math.nan), s.get("drift_event_rate", math.nan)
    return [
        (f"{tag}drift > 0.01 share", over_tol, over_tol <= max_over_tol, f"<= {max_over_tol:g} (synced env-steps; p99 below 0.01)"),
        (f"{tag}drift > 0.25 event rate", events, events <= max_event_rate, f"<= {max_event_rate:g} (~1% of branch windows hit)"),
        (f"{tag}max drift (info)", s["max_determinism_drift"], True, "not gated: rare GPU contact events"),
        (f"{tag}scale-0 failure rate", s["zero_scale_failure_rate"], s["zero_scale_failure_rate"] <= max_zero_fail, f"<= {max_zero_fail:g}"),
    ]


def lafan_checks(
    benchmark: dict, drift: dict, wm: dict, train_nse8: float, margins: dict | None,
    min_return: float = 0.70, max_mpjpe_m: float = 0.08, max_yaw_deg: float = 15.0, wm_ratio: tuple[float, float] = (0.5, 2.0),
) -> list[Check]:
    b = benchmark["summary"]
    k = wm["summary"].get("k", 8)
    wm_nse = wm["summary"][f"nse_{k}step"]
    ratio = wm_nse / train_nse8 if train_nse8 > 0 else math.inf
    mp = b.get("mean_mpjpe_aligned_m", math.nan)
    checks = [
        ("benchmark mean return", b["mean_norm_return"], b["mean_norm_return"] >= min_return, f">= {min_return} (paper ~0.82; 0.56 = no root feedback)"),
        ("drift-free body error [m]", mp, mp == mp and mp <= max_mpjpe_m, f"<= {max_mpjpe_m} (a few cm)"),
        ("mean yaw error [deg]", drift["summary"]["mean_yaw_deg"], drift["summary"]["mean_yaw_deg"] <= max_yaw_deg, f"<= {max_yaw_deg} (82 without orientation input)"),
        ("wm script / training log nse8", ratio, wm_ratio[0] <= ratio <= wm_ratio[1], f"in [{wm_ratio[0]}, {wm_ratio[1]}] (same metric)"),
    ]
    if margins is not None:
        checks += margin_checks(margins)
    return checks


def _report(checks: list[Check]) -> None:
    for name, value, ok, expect in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} {value:12.5g}   expected {expect}")
    failed = [c for c in checks if not c[2]]
    if failed:
        print(f"{len(failed)} check(s) failed")
        raise SystemExit(1)
    print("all checks passed")


def lafan(run_dir: Path = Path("logs/lafan_ori"), ckpt_stem: str = "ckpt_latest", train_log: Path | None = None) -> None:
    """Gate after the short LAFAN1 run (expects the files run_experiments.sh writes into ``run_dir``)."""
    margins = run_dir / f"margins_{ckpt_stem}.json"
    _report(lafan_checks(
        _load(run_dir / f"benchmark_{ckpt_stem}.json"),
        _load(run_dir / f"drift_{ckpt_stem}.json"),
        _load(run_dir / f"wm_error_{ckpt_stem}.json"),
        last_logged_nse8(train_log or run_dir / "train_stdout.log"),
        _load(margins) if margins.exists() else None,
    ))


def margins(files: tyro.conf.Positional[list[Path]], max_over_tol: float = 0.01, max_event_rate: float = 1e-4, max_zero_fail: float = 0.01) -> None:
    """Validity controls of one or more eval_margins.py outputs."""
    checks: list[Check] = []
    for f in files:
        checks += margin_checks(_load(f), max_over_tol, max_event_rate, max_zero_fail, tag=f"{Path(f).stem}: ")
    _report(checks)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"lafan": lafan, "margins": margins})
