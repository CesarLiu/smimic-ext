"""Check the simulator against the CommonRoad multi-body model's recorded responses.

All manoeuvres run as one batch (one row each, each driven by its own inputs), with ABS / ESC off (the
reference model has neither). The E0 gate is "within 5 %", made concrete per manoeuvre type:

* cornering (steady, step, sine, braking in a turn): NRMSE of yaw rate and lateral acceleration over the
  run; steady state: relative error of their final values (mean over the last 0.5 s); step and sine:
  relative error of the peak yaw rate;
* longitudinal (accelerate, brake): NRMSE of the speed and relative error of the speed change.

NRMSE = RMS(ours - reference) / RMS(reference). Lateral quantities of the straight-line manoeuvres are
not scored: they are ~0 (tyre offsets only), so relative errors there carry no information.

    PYTHONPATH=src python -m supermimic.vehicles.validation [--substeps 10] [--float64]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from supermimic.vehicles.threads import add_threads_arg, limit_threads
from supermimic.vehicles.dynamics import (
    AY,
    DELTA,
    ROLL,
    VX,
    VY,
    AssistCfg,
    Context,
    R,
    VehicleDynamics,
)
from supermimic.vehicles.maneuvers import DT, MANEUVERS

REFERENCE = Path(__file__).resolve().parents[3] / "tests" / "data" / "commonroad_mb_reference.pt"
TOL = 0.05
SIGNALS = ("t", "vx", "vy", "r", "delta", "roll", "ay")


def run_maneuvers(dyn: VehicleDynamics, maneuvers=MANEUVERS, dtype=torch.float32, device="cpu") -> dict[str, torch.Tensor]:
    """Simulate every manoeuvre (batched); per manoeuvre a (steps+1, len(SIGNALS)) tensor."""
    n = len(maneuvers)
    s = dyn.initial_state(torch.tensor([m.v0 for m in maneuvers], dtype=dtype, device=device))
    flags = dyn.new_flags(n, device, dtype)
    ctx = Context.nominal(n, device, dtype)
    steps = max(m.steps for m in maneuvers)
    rows = []
    for i in range(steps + 1):
        rows.append(torch.stack([torch.full((n,), i * DT, dtype=dtype, device=device), s[:, VX], s[:, VY], s[:, R], s[:, DELTA], s[:, ROLL], s[:, AY]], -1))
        if i == steps:
            break
        u = [m.inputs(i * DT, float(s[k, VX]), float(s[k, DELTA])) for k, m in enumerate(maneuvers)]
        sr = torch.tensor([x[0] for x in u], dtype=dtype, device=device)
        ac = torch.tensor([x[1] for x in u], dtype=dtype, device=device)
        s, flags = dyn.step(s, flags, sr, ac, ctx)
    out = torch.stack(rows, 1).double().cpu()  # (n, steps+1, signals)
    return {m.name: out[k, : m.steps + 1] for k, m in enumerate(maneuvers)}


def _nrmse(a, b):
    return float(torch.sqrt(((a - b) ** 2).mean()) / torch.sqrt((b**2).mean()).clamp_min(1e-12))


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def metrics(man, ours: torch.Tensor, ref: torch.Tensor, columns) -> dict[str, float]:
    c = {k: i for i, k in enumerate(columns)}
    o = {k: ours[:, i] for i, k in enumerate(SIGNALS)}
    m = {k: ref[:, c[k]].double() for k in ("vx", "r", "ay", "vy")}
    out = {}
    if man.kind in ("accel", "brake"):
        out["nrmse_vx"] = _nrmse(o["vx"], m["vx"])
        out["rel_dv"] = _rel(float(o["vx"][-1] - o["vx"][0]), float(m["vx"][-1] - m["vx"][0]))
        return out
    out["nrmse_r"] = _nrmse(o["r"], m["r"])
    out["nrmse_ay"] = _nrmse(o["ay"], m["ay"])
    if man.kind == "steady":
        last = slice(-25, None)
        out["rel_final_r"] = _rel(float(o["r"][last].mean()), float(m["r"][last].mean()))
        out["rel_final_ay"] = _rel(float(o["ay"][last].mean()), float(m["ay"][last].mean()))
    if man.kind in ("step", "sine"):
        out["rel_peak_r"] = _rel(float(o["r"].abs().max()), float(m["r"].abs().max()))
    return out


def compare(substeps: int = 10, dtype=torch.float32, reference: Path = REFERENCE) -> list[tuple[object, dict[str, float], dict[str, float]]]:
    """[(manoeuvre, metrics, information)] where information has the final sideslip angles and roll angles [deg]."""
    ref = torch.load(reference, weights_only=False)
    dyn = VehicleDynamics(substeps=substeps, assist=AssistCfg(abs=False, esc=False))
    runs = run_maneuvers(dyn, MANEUVERS, dtype)
    c = {k: i for i, k in enumerate(ref["columns"])}
    out = []
    for man in MANEUVERS:
        o, m = runs[man.name], ref["maneuvers"][man.name].double()
        info = {
            "beta_deg": (math.degrees(math.atan2(o[-1, 2], o[-1, 1])), math.degrees(math.atan2(m[-1, c["vy"]], m[-1, c["vx"]]))),
            "roll_deg": (math.degrees(o[-1, 5]), math.degrees(m[-1, c["roll"]])),
        }
        out.append((man, metrics(man, o, m, ref["columns"]), info))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substeps", type=int, default=10)
    ap.add_argument("--float64", action="store_true")
    add_threads_arg(ap)
    args = ap.parse_args()
    limit_threads(args.threads)
    rows = compare(args.substeps, torch.float64 if args.float64 else torch.float32)
    worst = 0.0
    for man, mt, info in rows:
        worst = max(worst, *mt.values())
        ms = "  ".join(f"{k} {v * 100:5.2f}%" for k, v in mt.items())
        print(f"{man.name:26s} {ms}   (beta {info['beta_deg'][0]:+.2f}/{info['beta_deg'][1]:+.2f} deg, roll {info['roll_deg'][0]:+.2f}/{info['roll_deg'][1]:+.2f} deg)")
    print(f"worst metric {worst * 100:.2f}% (gate {TOL * 100:.0f}%)")


if __name__ == "__main__":
    main()
