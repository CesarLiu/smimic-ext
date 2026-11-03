"""Retest gate for margin runs: the same checkpoint and clips evaluated twice, the second time with a shuffled
branch-to-env assignment (eval_margins.py ``--shuffle-seed``), so the simulator's rare nondeterministic contact
events land on different branches. Exit code 1 if the noise would change what we conclude.

  uv run python scripts/retest_margins.py --state-a logs/x/margins_state.json --state-b logs/x/margins_state_rt.json \
      --adv-a logs/x/margins_adv.json --adv-b logs/x/margins_adv_rt.json

Checks: per branch time, ``margin_min`` of the two runs within one grid step (no failure in the grid counts as
one step above the top) for at least ``min_agree`` of the times; and, given both kinds, the adv-vs-random
comparison (scripts/compare_margins.py) reaches the same verdict (median log2(adv / random) below zero or not)
with medians within ``max_median_shift`` (in log2 units).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import tyro

from supermimic.eval.margins import compare_directions


def _load(path: Path) -> tuple[dict, list[float]]:
    d = json.loads(Path(path).read_text())
    rows = {(c["name"], t["frame"]): t["margin_min"] for c in d["clips"] for t in c.get("times", [])}
    return rows, d["summary"]["scales"]


def grid_agreement(a: dict, b: dict, scales: list[float]) -> tuple[float, int]:
    """Share of keys in both (with a margin) whose grid indices differ by at most one."""
    idx = lambda v: len(scales) if not math.isfinite(v) else int(np.searchsorted(scales, v))  # noqa: E731
    keys = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    if not keys:
        return math.nan, 0
    return float(np.mean([abs(idx(a[k]) - idx(b[k])) <= 1 for k in keys])), len(keys)


def main(state_a: Path, state_b: Path, adv_a: Path | None = None, adv_b: Path | None = None,
         min_agree: float = 0.9, max_median_shift: float = 0.5) -> None:
    checks = []  # name, value, passed, expectation
    runs = {"state": (state_a, state_b)}
    if adv_a is not None and adv_b is not None:
        runs["adv"] = (adv_a, adv_b)
    loaded = {}
    for kind, (pa, pb) in runs.items():
        (ra, scales), (rb, _) = _load(pa), _load(pb)
        loaded[kind] = (ra, rb)
        agree, n = grid_agreement(ra, rb, scales)
        checks.append((f"{kind}: margins within one grid step ({n} times)", agree, agree >= min_agree, f">= {min_agree:g}"))
    if "adv" in loaded:
        ca = compare_directions(loaded["state"][0], loaded["adv"][0])
        cb = compare_directions(loaded["state"][1], loaded["adv"][1])
        ma, mb = ca.get("median_log2_adv_over_random", math.nan), cb.get("median_log2_adv_over_random", math.nan)
        print(f"  adv vs random, run a: {json.dumps(ca)}")
        print(f"  adv vs random, run b: {json.dumps(cb)}")
        checks.append(("adv smaller than random: same verdict", float((ma < 0) == (mb < 0)), (ma < 0) == (mb < 0), f"median log2 {ma:+.2f} vs {mb:+.2f}"))
        checks.append(("adv/random median log2 shift", abs(ma - mb), abs(ma - mb) <= max_median_shift, f"<= {max_median_shift:g}"))
    for name, value, ok, expect in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:48s} {value:10.4g}   expected {expect}")
    failed = [c for c in checks if not c[2]]
    if failed:
        print(f"{len(failed)} check(s) failed")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    tyro.cli(main)
