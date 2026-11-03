"""Extract a subset of BONES-SEED's g1.tar.gz while streaming it (the 23.5 GB archive never touches the disk).

Keeps every CSV whose name contains one of ``--include`` and a deterministic pseudo-random ``--fraction`` of
the rest (md5 of the clip name), so reruns pick the same clips. Needs the accepted dataset license and
``hf auth login`` (the token is read from the Hugging Face cache).

  uv run python scripts/data/stream_bones_subset.py --out data/raw/bones_ab/g1_csv --fraction 0.05 \
      --include flip_ side_roll stand_up_lying crossed_legs rest_
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path

import tyro

URL = "https://huggingface.co/datasets/bones-studio/seed/resolve/main/g1.tar.gz"


def keep(name: str, include: tuple[str, ...], fraction: float) -> bool:
    if any(s in name for s in include):
        return True
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction


def main(out: Path, fraction: float = 0.05, include: tuple[str, ...] = (), url: str = URL) -> None:
    from huggingface_hub import get_token

    token = get_token()
    if token is None:
        raise SystemExit("not logged in: run `uv run hf auth login` (and accept the BONES-SEED license)")
    out.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    kept = seen = 0
    with urllib.request.urlopen(req) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for m in tar:
            if not (m.isfile() and m.name.endswith(".csv")):
                continue
            seen += 1
            name = Path(m.name).name
            if keep(name[:-4], include, fraction):
                f = tar.extractfile(m)
                (out / name).write_bytes(f.read())
                kept += 1
            if seen % 10000 == 0:
                print(f"  {seen} CSVs scanned, {kept} kept", flush=True)
    print(f"done: {seen} CSVs scanned, {kept} kept in {out}")


if __name__ == "__main__":
    tyro.cli(main)
