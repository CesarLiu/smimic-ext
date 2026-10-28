"""Gates of scripts/check_run.py on synthetic result files."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_run", ROOT / "scripts/check_run.py")
check_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_run)


def _write_run(d: Path, ret=0.82, mpjpe=0.035, yaw=3.0, wm_nse=0.008, log_nse=0.0075, drift=0.0, zero=0.0) -> None:
    (d / "benchmark_ckpt_latest.json").write_text(json.dumps({"summary": {"mean_norm_return": ret, "mean_mpjpe_aligned_m": mpjpe}}))
    (d / "drift_ckpt_latest.json").write_text(json.dumps({"summary": {"mean_yaw_deg": yaw}}))
    (d / "wm_error_ckpt_latest.json").write_text(json.dumps({"summary": {"k": 8, "nse_8step": wm_nse}}))
    (d / "margins_ckpt_latest.json").write_text(json.dumps({"summary": {"max_determinism_drift": drift, "zero_scale_failure_rate": zero}}))
    (d / "train_stdout.log").write_text(
        "it 50 | wm nse1 nan nse8 nan | imag_ret 0.1\n"
        f"it 100 | wm nse1 0.0010 nse8 0.0123 | x\nit 150 | wm nse1 0.0009 nse8 {log_nse:.4f} | x\n"
    )


def test_last_logged_nse8_skips_nan(tmp_path):
    _write_run(tmp_path)
    assert check_run.last_logged_nse8(tmp_path / "train_stdout.log") == pytest.approx(0.0075)


def test_lafan_gate_passes_on_expected_values(tmp_path, capsys):
    _write_run(tmp_path)
    check_run.lafan(run_dir=tmp_path)
    assert "all checks passed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "bad,name",
    [
        (dict(ret=0.56), "benchmark mean return"),
        (dict(mpjpe=0.30), "drift-free body error"),
        (dict(yaw=82.0), "mean yaw error"),
        (dict(wm_nse=0.05), "wm script / training log"),
        (dict(drift=0.05), "determinism drift"),
        (dict(zero=0.2), "scale-0 failure rate"),
    ],
)
def test_lafan_gate_fails_on_each_bad_value(tmp_path, capsys, bad, name):
    _write_run(tmp_path, **bad)
    with pytest.raises(SystemExit):
        check_run.lafan(run_dir=tmp_path)
    out = capsys.readouterr().out
    assert any(line.strip().startswith("[FAIL]") and name in line for line in out.splitlines())


def test_margins_gate(tmp_path):
    good, bad = tmp_path / "a.json", tmp_path / "b.json"
    good.write_text(json.dumps({"summary": {"max_determinism_drift": 0.0, "zero_scale_failure_rate": 0.0}}))
    bad.write_text(json.dumps({"summary": {"max_determinism_drift": 0.0, "zero_scale_failure_rate": 0.5}}))
    check_run.margins([good])
    with pytest.raises(SystemExit):
        check_run.margins([good, bad])
