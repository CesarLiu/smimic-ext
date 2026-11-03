"""Write ``g1_heads_golden.pt``: the output of ``tests/g1_fixtures.run_all`` (reward, observation,
termination, world-model loss, imagined step and rollouts, one policy + critic update), recorded from
the implementation before the AnalyticHeads interface refactor. ``tests/test_heads_interface.py``
checks the current code against it.

Run from the repository root:  PYTHONPATH=src python tests/data/make_g1_heads_golden.py
Only regenerate it on purpose, when a G1 numeric change is intended.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_fixtures import run_all  # noqa: E402

OUT = Path(__file__).with_name("g1_heads_golden.pt")

if __name__ == "__main__":
    torch.save(run_all(), OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
