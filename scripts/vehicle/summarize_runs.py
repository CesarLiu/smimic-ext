"""Mean +- sd over seeds of ``compare_ppo.py`` runs, per configuration (VEHICLE_PROPOSAL.md section 11.10).

    python scripts/vehicle/summarize_runs.py logs/sweep/stability [logs/sweep/ppo_seeds ...] [--json out.json]

Expects ``<dir>/<config>/<seed dir>/report.json`` (as ``sweep.py`` writes them). Per configuration and method:
final held-out return (mean over friction levels), its spread over seeds, the lowest held-out return in the second
half of each run (a stability measure: oscillating runs dip there), the S-bend return and mean |lateral error|,
the held-out termination rate, and the held-out return at fixed numbers of simulation steps.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

MARKS = [500_000, 1_000_000, 2_000_000, 5_000_000, 13_369_344]


def run_stats(m: dict) -> dict:
    curve = m["curve"]
    hf, sf = m["heldout_final"], m["scurve_final"]
    def mean(d, k):
        return sum(v[k] for v in d.values()) / len(d)

    last = curve[-1]["env_steps"] if curve else 0
    late = [c["heldout_return"] for c in curve if c["env_steps"] >= last / 2] or [mean(hf, "normalized_return")]
    at = {}
    for mark in MARKS:  # the first evaluation at or past the mark (within 10 %)
        c = next((c for c in curve if mark <= c["env_steps"] <= 1.1 * mark), None)
        if c:
            at[mark] = c["heldout_return"]
    return {"final": mean(hf, "normalized_return"), "late_min": min(late), "scurve": mean(sf, "normalized_return"),
            "scurve_lat": mean(sf, "mean_abs_lateral_error_m"), "term": mean(hf, "termination_rate"), "at": at,
            "env_steps": last, "train_s": m.get("train_s")}


def ms(v: list[float]) -> str:
    if not v:
        return "-"
    return f"{st.mean(v):.3f} ± {st.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    table = {}
    for d in args.dirs:
        for rep in sorted(Path(d).glob("*/*/report.json")):
            config = rep.parent.parent.name
            for method, m in json.loads(rep.read_text())["methods"].items():
                if "heldout_final" in m and m.get("curve") is not None:
                    table.setdefault(f"{config} [{method}]", []).append({"seed": rep.parent.name, **run_stats(m)})
    head = ["configuration", "seeds", "final held-out", "worst seed", "late-phase min", "S-bend return", "S-bend |lat| [m]", "held-out term.",
            *[f"@{m / 1e6:g}M steps" for m in MARKS]]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for name, runs in table.items():
        def col(k, runs=runs):
            return [r[k] for r in runs]

        row = [name, str(len(runs)), ms(col("final")), f"{min(col('final')):.3f}", ms(col("late_min")), ms(col("scurve")),
               ms(col("scurve_lat")), ms(col("term")), *[ms([r["at"][m] for r in runs if m in r["at"]]) for m in MARKS]]
        print("| " + " | ".join(row) + " |")
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
