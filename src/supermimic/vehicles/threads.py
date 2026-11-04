"""CPU thread limits for the vehicle scripts.

PyTorch starts as many intra-op threads as the machine has cores. The vehicle scripts train small networks
(256-512 wide) on small batches, where thread synchronisation costs far more than it saves: on a 192-core
machine an unlimited run occupies every core and is about 8x slower than on 4 cores (E1 default: 1949 s vs
229 s). Every script therefore caps its threads at start-up.
"""

from __future__ import annotations

import os

import torch

DEFAULT_MAX_THREADS = 8


def usable_cores() -> int:
    """Cores this process may run on (affinity / ``taskset``), not the machine total."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        return os.cpu_count() or 1


def default_threads() -> int:
    """``OMP_NUM_THREADS`` if the user set it, else min(8, usable cores)."""
    env = os.environ.get("OMP_NUM_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(DEFAULT_MAX_THREADS, usable_cores()))


def limit_threads(n: int | None = None) -> int:
    """Cap PyTorch (and BLAS / OpenMP libraries loaded later) at ``n`` CPU threads; returns ``n``."""
    n = default_threads() if n is None else max(1, int(n))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n)
    torch.set_num_threads(n)
    try:  # only possible before the first inter-op parallel work; harmless to skip afterwards
        torch.set_num_interop_threads(min(n, 4))
    except RuntimeError:
        pass
    return n


def add_threads_arg(ap) -> None:
    ap.add_argument("--threads", type=int, default=None,
                    help=f"CPU threads (default: OMP_NUM_THREADS if set, else min({DEFAULT_MAX_THREADS}, usable cores))")
