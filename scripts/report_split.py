"""One table for a held-out experiment (EXPERIMENTS, g-series): benchmark results of the training-eval,
in-category and leave-category-out sets side by side, optionally with the world-model error of each, and a
per-category breakdown.

  uv run python scripts/report_split.py --categories data/splits/g1/clip_categories.csv \
      --bench train_eval=logs/g1/benchmark_train_eval.json heldout_in=logs/g1/benchmark_heldout_in.json \
              heldout_cat=logs/g1/benchmark_heldout_cat.json \
      --wm train_eval=logs/g1/wm_train_eval.json heldout_cat=logs/g1/wm_heldout_cat.json

Benchmark JSONs come from scripts/eval_benchmark.py, world-model JSONs from scripts/eval_wm_heldout.py.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import tyro


def _pairs(items: list[str]) -> dict[str, Path]:
    out = {}
    for it in items:
        name, _, path = it.partition("=")
        if not path:
            raise SystemExit(f"expected name=path, got {it!r}")
        out[name] = Path(path)
    return out


def _mean(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return sum(xs) / len(xs) if xs else math.nan


def main(bench: list[str], categories: Path | None = None, wm: list[str] | None = None, per_category: list[str] | None = None, out: Path | None = None) -> None:
    """``per_category``: sets to break down by category (default: every set given with --bench)."""
    benches = {k: json.load(open(p))["clips"] for k, p in _pairs(bench).items()}
    wms = {k: json.load(open(p)) for k, p in _pairs(wm or []).items()}
    cat_of: dict[str, str] = {}
    if categories is not None:
        with open(categories) as f:
            cat_of = {r["name"]: r["category"] for r in csv.DictReader(f)}

    def stats(clips):
        n = len(clips)
        return {
            "clips": n,
            "failed": sum(c["failed"] for c in clips),
            "fail_rate": sum(c["failed"] for c in clips) / max(n, 1),
            "mean_norm_return": _mean([c["norm_return"] for c in clips]),
            "mean_mpjpe_aligned_cm": 100 * _mean([c.get("mpjpe_aligned", math.nan) for c in clips]),
        }

    report: dict = {"sets": {}, "by_category": {}}
    print(f"{'set':14s} {'clips':>7s} {'failed':>7s} {'fail %':>7s} {'return':>7s} {'mpjpe cm':>9s} {'wm 1-step':>10s} {'wm k-step':>10s}")
    for name, clips in benches.items():
        s = stats(clips)
        w = wms.get(name, {}).get("summary", {})
        kk = w.get("k", 8)
        s["wm_nse_1step"] = w.get("nse_1step", math.nan)
        s["wm_nse_kstep"] = w.get(f"nse_{kk}step", math.nan)
        report["sets"][name] = s
        print(f"{name:14s} {s['clips']:7d} {s['failed']:7d} {100 * s['fail_rate']:7.2f} {s['mean_norm_return']:7.3f} {s['mean_mpjpe_aligned_cm']:9.2f} {s['wm_nse_1step']:10.5f} {s['wm_nse_kstep']:10.5f}")

    if cat_of:
        for name in per_category or list(benches):
            groups = defaultdict(list)
            for c in benches[name]:
                groups[cat_of.get(c["name"], "<uncategorised>")].append(c)
            wcat = wms.get(name, {}).get("summary", {}).get("by_category", {})
            kk = wms.get(name, {}).get("summary", {}).get("k", 8)
            rows = {}
            for cat, clips in groups.items():
                s = stats(clips)
                s["wm_nse_kstep"] = wcat.get(cat, {}).get(f"nse_{kk}step", math.nan)
                rows[cat] = s
            report["by_category"][name] = rows
            print(f"\n{name} by category (worst fail rate first)")
            print(f"  {'category':38s} {'clips':>6s} {'fail %':>7s} {'return':>7s} {'mpjpe cm':>9s} {'wm k-step':>10s}")
            for cat, s in sorted(rows.items(), key=lambda kv: (-kv[1]["fail_rate"], kv[1]["mean_norm_return"]))[:40]:
                print(f"  {cat[:38]:38s} {s['clips']:6d} {100 * s['fail_rate']:7.2f} {s['mean_norm_return']:7.3f} {s['mean_mpjpe_aligned_cm']:9.2f} {s['wm_nse_kstep']:10.5f}")
    if out is not None:
        Path(out).write_text(json.dumps(report, indent=1))
        print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)
