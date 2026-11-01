"""Does the world model know where the policy is weak? Random vs model worst-case directions (g-series).

Compares two eval_margins.py outputs on the same clips and branch times: ``--kind state`` (random
directions, worst of ``--num-dirs``) and ``--kind adv`` (the model's steepest-descent direction). A
negative median log2(adv / random) means the model's direction breaks the policy with a smaller push
than any random one; -1 = half the size. Also prints the adv run's model-vs-simulator calibration.

  uv run python scripts/compare_margins.py --random logs/g1/margins_heldout_cat_state.json \
      --adv logs/g1/margins_heldout_cat_adv.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import tyro

from supermimic.eval.margins import compare_directions


def _rows(path: Path) -> tuple[dict, dict, dict]:
    d = json.loads(Path(path).read_text())
    rows, cat = {}, {}
    for c in d["clips"]:
        for t in c.get("times", []):
            rows[(c["name"], t["frame"])] = t["margin_min"]
            cat[(c["name"], t["frame"])] = c.get("category", "<uncategorised>")
    return rows, cat, d["summary"]


def main(random: Path, adv: Path, out: Path | None = None) -> None:
    r_rows, _, r_sum = _rows(random)
    a_rows, cat, a_sum = _rows(adv)
    if r_sum.get("components") != a_sum.get("components") or r_sum.get("window_steps") != a_sum.get("window_steps"):
        print(f"[warn] the runs differ in components or window: {r_sum.get('components')}/{r_sum.get('window_steps')} vs {a_sum.get('components')}/{a_sum.get('window_steps')}")
    result = {"overall": compare_directions(r_rows, a_rows), "model_vs_sim": a_sum.get("model_vs_sim")}
    by_cat = defaultdict(lambda: ({}, {}))
    for k in a_rows:
        by_cat[cat[k]][1][k] = a_rows[k]
        if k in r_rows:
            by_cat[cat[k]][0][k] = r_rows[k]
    result["by_category"] = {c: compare_directions(r, a) for c, (r, a) in sorted(by_cat.items())}
    print(json.dumps(result, indent=1))
    if out:
        Path(out).write_text(json.dumps(result, indent=1))
        print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)
