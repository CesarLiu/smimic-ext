"""Held-out splits of a filtered motion set for generalisation tests (EXPERIMENTS, g-series).

Two held-out sets from one training run:
  * heldout_cat : every clip of a few whole categories (leave-category-out);
  * heldout_in  : a random fraction of the remaining clips (in-category held-out). Mirror pairs
                  (``..._M``) are kept on the same side of the split so a clip never meets its mirror.
The rest is the training pool, optionally subsampled to a fixed number of hours.

Categories come from a metadata table column (``--metadata`` + ``--category-col``) or, without one,
from the clip name with its take / actor / mirror suffix removed (``high_jump_R_001__A450`` ->
``high_jump``; a heuristic, check it with ``--inspect-only``). Start with ``--inspect-only`` to see the metadata
columns and the category sizes:

  uv run python scripts/data/make_split.py --manifest data/npz/bones_seed/manifest_filtered.csv \
      --metadata data/raw/bones_seed/metadata/seed_metadata_v004.csv --inspect-only
  uv run python scripts/data/make_split.py --manifest data/npz/bones_seed/manifest_filtered.csv \
      --metadata data/raw/bones_seed/metadata/seed_metadata_v004.csv --category-col <column> \
      --heldout-categories <c1> <c2> ... --train-hours 30 --out data/splits/g1

Outputs (in ``--out``): train_manifest.csv (for ``--data.manifest`` / build_library.py), train_eval.txt,
heldout_in.txt, heldout_cat.txt (for eval_benchmark.py ``--clip-list``), clip_categories.csv, split.json.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import tyro

FPS = 50
_SUFFIX = re.compile(r"(?:_[RL])?_\d+__A\d+(?:_M)?$")
_EXTENSIONS = (".csv", ".npz", ".bvh", ".fbx", ".usd")


def move_of(name: str) -> str:
    """Clip name without take / actor / mirror suffix (falls back to the full name)."""
    m = _SUFFIX.search(name)
    return name[: m.start()] if m else name


def mirror_group(name: str) -> str:
    return name[:-2] if name.endswith("_M") else name


def _clip_key(value) -> str:
    s = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    for ext in _EXTENSIONS:
        if s.lower().endswith(ext):
            return s[: -len(ext)]
    return s


def read_manifest(path: Path) -> dict[str, int]:
    """name -> frames for the clips the filter kept."""
    with open(path) as f:
        return {r["name"]: int(r.get("frames") or 0) for r in csv.DictReader(f) if r.get("keep", "1") in ("1", "True", "true")}


def read_metadata(path: Path):
    import pandas as pd

    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


def detect_name_column(df, names: set[str]) -> tuple[str | None, float]:
    from pandas.api.types import is_object_dtype, is_string_dtype

    best, best_cov = None, 0.0
    for col in df.columns:
        if not (is_object_dtype(df[col]) or is_string_dtype(df[col])):  # pandas 3 reads text as str dtype
            continue
        keys = {_clip_key(v) for v in df[col].dropna()}
        cov = len(names & keys) / max(len(names), 1)
        if cov > best_cov:
            best, best_cov = col, cov
    return best, best_cov


def categories_from_metadata(df, name_col: str, category_col: str, names: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    dup = 0
    for n, c in zip(df[name_col], df[category_col]):
        k = _clip_key(n)
        if k not in names:
            continue
        if k in out:
            dup += 1
            continue
        out[k] = "<missing>" if c != c or c is None else str(c)  # NaN check
    if dup:
        print(f"[warn] {dup} duplicate metadata rows ignored (first row per clip used)")
    return out


def inspect(names: dict[str, int], df) -> None:
    hours = sum(names.values()) / FPS / 3600
    print(f"{len(names)} kept clips, {hours:.2f} h")
    moves = Counter(move_of(n) for n in names)
    unmatched = sum(1 for n in names if not _SUFFIX.search(n))
    print(f"\nname-derived moves: {len(moves)} distinct; {unmatched} names do not match the take/actor suffix pattern")
    for m, c in moves.most_common(25):
        print(f"  {c:6d}  {m}")
    if df is None:
        return
    col, cov = detect_name_column(df, set(names))
    print(f"\nmetadata: {len(df)} rows, {len(df.columns)} columns; clip-name column guess: {col!r} ({100 * cov:.1f}% of kept clips matched)")
    for c in df.columns:
        vals = df[c].dropna()
        nun = vals.nunique()
        sample = ", ".join(repr(v) for v in vals.unique()[:5])
        print(f"  {c:32s} {str(df[c].dtype):8s} {nun:8d} distinct  e.g. {sample[:110]}")


def _hours(clips, frames: dict[str, int]) -> float:
    return sum(frames[c] for c in clips) / FPS / 3600


def main(
    manifest: Path,
    out: Path | None = None,
    metadata: Path | None = None,
    name_col: str | None = None,
    category_col: str | None = None,
    inspect_only: bool = False,
    heldout_categories: list[str] | None = None,
    num_heldout_categories: int = 5,
    min_category_clips: int = 50,
    max_category_frac: float = 0.03,
    in_frac: float = 0.05,
    train_hours: float | None = None,
    train_eval_size: int = 2000,
    seed: int = 0,
) -> None:
    frames = read_manifest(manifest)
    names = set(frames)
    df = read_metadata(metadata) if metadata is not None else None
    if inspect_only:
        inspect(frames, df)
        return
    if out is None:
        raise SystemExit("--out is required unless --inspect-only")

    # ---------------- categories
    if category_col is not None:
        if df is None:
            raise SystemExit("--category-col needs --metadata")
        if name_col is None:
            name_col, cov = detect_name_column(df, names)
            print(f"clip-name column: {name_col!r} ({100 * cov:.1f}% matched)")
        cat = categories_from_metadata(df, name_col, category_col, names)
        missing = names - cat.keys()
        if len(missing) > 0.02 * len(names):
            raise SystemExit(f"{len(missing)} of {len(names)} kept clips have no metadata row; check --name-col (try --inspect-only)")
        for n in missing:
            cat[n] = "<missing>"
        source = f"metadata:{category_col}"
    else:
        cat = {n: move_of(n) for n in names}
        source = "name"
    by_cat: dict[str, list[str]] = defaultdict(list)
    for n in sorted(names):
        by_cat[cat[n]].append(n)

    # ---------------- leave-category-out set
    rng = random.Random(seed)
    if heldout_categories:
        unknown = [c for c in heldout_categories if c not in by_cat]
        if unknown:
            raise SystemExit(f"unknown categories {unknown}; available e.g. {sorted(by_cat)[:20]}")
        chosen = list(heldout_categories)
    else:
        eligible = sorted(c for c, v in by_cat.items() if min_category_clips <= len(v) <= max_category_frac * len(names) and c != "<missing>")
        if len(eligible) < num_heldout_categories:
            raise SystemExit(f"only {len(eligible)} categories with {min_category_clips}..{max_category_frac:.0%} of the clips; pass --heldout-categories")
        chosen = sorted(rng.sample(eligible, num_heldout_categories))
    heldout_cat = sorted(n for c in chosen for n in by_cat[c])

    # ---------------- in-category held-out set (mirror pairs stay together)
    pool = sorted(names - set(heldout_cat))
    groups: dict[str, list[str]] = defaultdict(list)
    for n in pool:
        groups[mirror_group(n)].append(n)
    keys = sorted(groups)
    rng.shuffle(keys)
    target = in_frac * len(pool)
    heldout_in: list[str] = []
    rest_keys: list[str] = []
    for k in keys:
        if len(heldout_in) < target:
            heldout_in += groups[k]
        else:
            rest_keys.append(k)

    # ---------------- training set (optionally a fixed number of hours)
    train: list[str] = []
    unused: list[str] = []
    budget = None if train_hours is None else train_hours * 3600 * FPS
    used = 0
    for k in rest_keys:  # already shuffled: a uniform subsample of mirror groups
        clips = groups[k]
        if budget is None or used < budget:
            train += clips
            used += sum(frames[c] for c in clips)
        else:
            unused += clips
    train, unused, heldout_in = sorted(train), sorted(unused), sorted(heldout_in)
    train_eval = sorted(rng.sample(train, min(train_eval_size, len(train))))

    # ---------------- coverage of the in-category set by the actual training set
    train_cats = {cat[n] for n in train}
    train_moves = {move_of(n) for n in train}
    cov_cat = sum(cat[n] in train_cats for n in heldout_in) / max(len(heldout_in), 1)
    cov_move = sum(move_of(n) in train_moves for n in heldout_in) / max(len(heldout_in), 1)
    leak = sum(cat[n] in train_cats for n in heldout_cat)
    assert leak == 0, "a held-out category reached the training set"

    # ---------------- write
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "frames", "keep"])
        for n in train:
            w.writerow([n, frames[n], 1])
    for fname, clips in (("train_eval.txt", train_eval), ("heldout_in.txt", heldout_in), ("heldout_cat.txt", heldout_cat)):
        (out / fname).write_text("".join(c + "\n" for c in clips))
    split_of = {**{n: "train_unused" for n in unused}, **{n: "train" for n in train}, **{n: "heldout_in" for n in heldout_in}, **{n: "heldout_cat" for n in heldout_cat}}
    with open(out / "clip_categories.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "category", "move", "split", "frames"])
        for n in sorted(names):
            w.writerow([n, cat[n], move_of(n), split_of[n], frames[n]])
    summary = {
        "manifest": str(manifest),
        "metadata": str(metadata) if metadata else None,
        "category_source": source,
        "seed": seed,
        "in_frac": in_frac,
        "train_hours_target": train_hours,
        "heldout_categories": {c: {"clips": len(by_cat[c]), "hours": round(_hours(by_cat[c], frames), 3)} for c in chosen},
        "splits": {
            s: {"clips": len(v), "hours": round(_hours(v, frames), 3)}
            for s, v in (("train", train), ("train_unused", unused), ("train_eval", train_eval), ("heldout_in", heldout_in), ("heldout_cat", heldout_cat))
        },
        "heldout_in_category_seen_in_train": round(cov_cat, 4),
        "heldout_in_move_seen_in_train": round(cov_move, 4),
        "num_categories": len(by_cat),
    }
    (out / "split.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)
