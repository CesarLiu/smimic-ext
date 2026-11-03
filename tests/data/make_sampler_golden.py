"""Write ``sampler_golden.pt``: the output of ``tests/sampler_fixtures.run_all`` (the return-based reset
sampler driven through a scripted episode sequence under four configurations), recorded from the
implementation before the task-interface refactor. ``tests/test_task_interface.py`` checks the
current code against it.

Run from the repository root:  PYTHONPATH=src python tests/data/make_sampler_golden.py
Only regenerate it on purpose, when a sampler change is intended.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sampler_fixtures import run_all

OUT = Path(__file__).with_name("sampler_golden.pt")

if __name__ == "__main__":
    torch.save(run_all(), OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
