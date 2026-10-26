"""Invariants of scripts/data/make_split.py on a synthetic BONES-SEED-like manifest."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_split", ROOT / "scripts/data/make_split.py")
make_split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_split)

CATEGORIES = {f"cat{i}": [f"move{i}_{j}" for j in range(3)] for i in range(12)}


def _write_inputs(tmp: Path) -> tuple[Path, Path]:
    rows, meta = [], []
    for c, moves in CATEGORIES.items():
        for m in moves:
            for take in range(1, 5):
                for actor in (101, 202):
                    base = f"{m}_R_{take:03d}__A{actor}"
                    for name in (base, base + "_M"):
                        rows.append((name, 250 + 10 * take))
                        meta.append((name + ".bvh", c))
    manifest = tmp / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "frames", "keep", "reason"])
        w.writerows([(n, fr, 1, "") for n, fr in rows] + [("dropped_R_001__A1", 100, 0, "name:chair")])
    metadata = tmp / "meta.csv"
    with open(metadata, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["move_id", "filename", "category"])
        w.writerows([(i, n, c) for i, (n, c) in enumerate(meta)])
    return manifest, metadata


def _read(out: Path):
    with open(out / "clip_categories.csv") as f:
        rows = list(csv.DictReader(f))
    return rows, json.loads((out / "split.json").read_text())


def test_move_of():
    assert make_split.move_of("high_jump_R_001__A450") == "high_jump"
    assert make_split.move_of("fancy_spin_R_001__A430_M") == "fancy_spin"
    assert make_split.move_of("no_suffix_here") == "no_suffix_here"


@pytest.mark.parametrize("use_metadata", [True, False])
def test_split_invariants(tmp_path, use_metadata):
    manifest, metadata = _write_inputs(tmp_path)
    out = tmp_path / "split"
    kw = dict(metadata=metadata, category_col="category") if use_metadata else {}
    held = ["cat3", "cat7"] if use_metadata else ["move3_0", "move7_1"]
    make_split.main(manifest, out=out, heldout_categories=held, in_frac=0.1, train_hours=0.6, train_eval_size=50, seed=1, **kw)
    rows, summary = _read(out)
    split = {r["name"]: r["split"] for r in rows}
    cat = {r["name"]: r["category"] for r in rows}
    assert "dropped_R_001__A1" not in split  # filtered clips never enter any split
    # Whole held-out categories, and none of them in training.
    assert {cat[n] for n, s in split.items() if s == "heldout_cat"} == set(held)
    assert all(split[n] == "heldout_cat" for n in split if cat[n] in held)
    # Mirror pairs stay on one side.
    for n, s in split.items():
        if n.endswith("_M"):
            assert split[n[:-2]] == s
    # Training hours respect the budget up to one mirror group.
    assert 0.6 <= summary["splits"]["train"]["hours"] < 0.6 + 2 * 300 / 50 / 3600
    assert summary["heldout_in_category_seen_in_train"] == 1.0
    with open(out / "train_manifest.csv") as f:
        train = {r["name"] for r in csv.DictReader(f)}
    assert train == {n for n, s in split.items() if s == "train"}
    assert set((out / "train_eval.txt").read_text().split()) <= train
    assert set((out / "heldout_in.txt").read_text().split()) == {n for n, s in split.items() if s == "heldout_in"}


def test_random_categories_are_reproducible(tmp_path):
    manifest, metadata = _write_inputs(tmp_path)
    picks = []
    for run in ("a", "b"):
        make_split.main(manifest, out=tmp_path / run, metadata=metadata, category_col="category", num_heldout_categories=3, min_category_clips=10, max_category_frac=0.2, seed=7)
        picks.append(sorted(json.loads((tmp_path / run / "split.json").read_text())["heldout_categories"]))
    assert picks[0] == picks[1] and len(picks[0]) == 3
