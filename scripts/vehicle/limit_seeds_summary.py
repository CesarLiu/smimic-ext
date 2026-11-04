"""Combine several ``limit_study.py`` reports (one per seed) into mean / spread per model and friction level,
and the paired difference large - small per seed (the question §3.10 left open: how big is the gap?).

    PYTHONPATH=src python scripts/vehicle/limit_seeds_summary.py logs/limit/seed_*.json [--out logs/limit/summary.json]

Each report must contain both models (``--models small large``, the default). A seed changes the references
as well as the training, so the spread covers both.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

METRICS = ("normalized_return", "near_limit_survived_share", "termination_rate", "mean_steps")


def _stats(xs: list[float]) -> dict:
    return {"mean": statistics.fmean(xs), "std": statistics.stdev(xs) if len(xs) > 1 else 0.0, "min": min(xs), "max": max(xs), "n": len(xs)}


def summarize(reports: list[dict]) -> dict:
    models = sorted(set.intersection(*(set(r["results"]) for r in reports)))
    levels = sorted(set.intersection(*(set(r["results"][m]["after"]) for r in reports for m in models)))
    out = {"seeds": [r["args"]["seed"] for r in reports], "models": models, "levels": levels, "per_model": {}, "large_minus_small": {}}
    for m in models:
        out["per_model"][m] = {lv: {k: _stats([r["results"][m]["after"][lv][k] for r in reports]) for k in METRICS} for lv in levels}
    if {"small", "large"} <= set(models):
        for lv in levels:
            row = {}
            for k in ("normalized_return", "near_limit_survived_share"):
                d = [r["results"]["large"]["after"][lv][k] - r["results"]["small"]["after"][lv][k] for r in reports]
                s = _stats(d)
                s["large_better"] = sum(x > 0 for x in d)
                s["tie"] = sum(x == 0 for x in d)
                # rough 95 % interval of the mean difference (normal approximation; few seeds -> wide)
                s["ci95"] = [s["mean"] - 1.96 * s["std"] / math.sqrt(len(d)), s["mean"] + 1.96 * s["std"] / math.sqrt(len(d))]
                row[k] = s
            out["large_minus_small"][lv] = row
    return out


def table(summary: dict) -> str:
    lines = [f"seeds {summary['seeds']}", "", "after training: mean +- std (min..max)"]
    for lv in summary["levels"]:
        for m in summary["models"]:
            st = summary["per_model"][m][lv]
            r, s = st["normalized_return"], st["near_limit_survived_share"]
            lines.append(f"  {lv:7s} {m:6s} return {r['mean']:.3f} +- {r['std']:.3f} ({r['min']:.3f}..{r['max']:.3f})   "
                         f"near-limit survived {s['mean']:.2f} +- {s['std']:.2f} ({s['min']:.2f}..{s['max']:.2f})")
    if summary["large_minus_small"]:
        lines += ["", "large - small, paired by seed: mean [95% CI]  (large better / ties / seeds)"]
        for lv, row in summary["large_minus_small"].items():
            cells = [f"{k.replace('normalized_', '').replace('near_limit_survived_share', 'near-limit survived')} {d['mean']:+.3f} "
                     f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] ({d['large_better']}/{d['tie']}/{d['n']})" for k, d in row.items()]
            lines.append(f"  {lv:7s} " + "   ".join(cells))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    reports = [json.loads(Path(p).read_text()) for p in args.reports]
    if len({r["args"]["seed"] for r in reports}) != len(reports):
        print("warning: some reports share a seed", file=sys.stderr)
    summary = summarize(reports)
    print(table(summary))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
