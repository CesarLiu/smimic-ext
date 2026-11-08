"""Project Chrono vehicles as a batch of environments (VEHICLE_PROPOSAL.md section 11, gate G1).

``ChronoPool`` starts one ``chrono_worker.py`` process per vehicle under the PyChrono interpreter and steps them
in parallel (send to all, then read all). It is plain numpy, so it runs in the project's environment while Chrono runs
in its own (``SUPERMIMIC_CHRONO_PYTHON`` or ``python=`` gives that interpreter; VEHICLE_RUNBOOK.md step 10 installs it).

Also here, independent of Chrono: constant-speed path references, path-tracking errors, a pure-pursuit controller
and the single-track inverse dynamics used to identify axle tyre curves from logged data.
"""

from __future__ import annotations

import math
import os
import pickle
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WORKER = Path(__file__).with_name("chrono_worker.py")


def chrono_python(python: str | None = None) -> str:
    py = python or os.environ.get("SUPERMIMIC_CHRONO_PYTHON")
    if not py:
        raise RuntimeError("set SUPERMIMIC_CHRONO_PYTHON to the PyChrono interpreter (VEHICLE_RUNBOOK.md step 10)")
    return py


class ChronoPool:
    """``n`` Chrono vehicles in ``n`` processes. ``reset(cfgs)`` and ``step(s_cmd, a_cmd)`` take one entry per vehicle
    and return a list of observation dicts (see ``chrono_worker.Sim.observe``)."""

    def __init__(self, n: int, python: str | None = None):
        py = chrono_python(python)
        self.procs = [subprocess.Popen([py, "-I", "-u", str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                      for _ in range(n)]
        self.n = n

    def _send(self, i: int, msg: dict) -> None:
        data = pickle.dumps(msg, protocol=4)
        self.procs[i].stdin.write(struct.pack("<I", len(data)) + data)
        self.procs[i].stdin.flush()

    def _recv(self, i: int) -> dict:
        out = self.procs[i].stdout
        hdr = out.read(4)
        if len(hdr) < 4:
            raise RuntimeError(f"Chrono worker {i} exited (code {self.procs[i].poll()})")
        msg = pickle.loads(out.read(struct.unpack("<I", hdr)[0]))
        if isinstance(msg, dict) and "error" in msg:
            raise RuntimeError(f"Chrono worker {i}: {msg['error']}")
        return msg

    def _all(self, msgs: list[dict], ids=None) -> list[dict]:
        ids = list(range(self.n)) if ids is None else list(ids)
        for i, m in zip(ids, msgs):
            self._send(i, m)
        return [self._recv(i) for i in ids]

    def reset(self, cfgs: list[dict], ids=None) -> list[dict]:
        return self._all([{"op": "reset", "cfg": c} for c in cfgs], ids)

    def step(self, s_cmd, a_cmd, ids=None) -> list[dict]:
        return self._all([{"op": "step", "s_cmd": float(s), "a_cmd": float(a)} for s, a in zip(s_cmd, a_cmd)], ids)

    def info(self, ids=None) -> list[dict]:
        ids = list(range(self.n)) if ids is None else list(ids)
        return self._all([{"op": "info"}] * len(ids), ids)

    def close(self) -> None:
        for i, p in enumerate(self.procs):
            if p.poll() is None:
                try:
                    self._send(i, {"op": "close"})
                    self._recv(i)
                except (BrokenPipeError, RuntimeError):
                    pass
                p.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------- references and errors


@dataclass
class PathRef:
    """A path driven at constant speed, sampled every ``dt``: arrays x, y, yaw, kappa (curvature) and v."""

    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    kappa: np.ndarray
    v: float
    dt: float

    def __len__(self) -> int:
        return len(self.x)

    def node(self, k) -> np.ndarray:
        """(len(k), 5) rows x, y, yaw, v, beta (= 0), as ``NMPC.solve`` takes them."""
        k = np.minimum(np.asarray(k), len(self) - 1)
        return np.stack([self.x[k], self.y[k], self.yaw[k], np.full(len(k), self.v), np.zeros(len(k))], -1)


def path_reference(kappa_of_s, length: float, v: float, dt: float = 0.02, x0: float = 0.0, y0: float = 0.0,
                   yaw0: float = 0.0) -> PathRef:
    """Integrate a curvature profile kappa(s) [1/m] along arc length at constant speed ``v``."""
    n = int(length / (v * dt)) + 1
    s = np.arange(n) * v * dt
    kap = np.array([kappa_of_s(si) for si in s])
    ds = v * dt
    yaw = yaw0 + np.concatenate([[0.0], np.cumsum(0.5 * (kap[1:] + kap[:-1]) * ds)])
    x = x0 + np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(yaw[1:]) + np.cos(yaw[:-1])) * ds)])
    y = y0 + np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(yaw[1:]) + np.sin(yaw[:-1])) * ds)])
    return PathRef(x, y, yaw, kap, v, dt)


def s_bend(v: float, ay: float, straight: float = 15.0, arc: float = 25.0, kappa_max: float = 0.1, **kw) -> PathRef:
    """Straight, left arc, right arc, straight, with smooth (half-cosine) curvature transitions; the arc curvature
    gives a lateral acceleration ``ay`` at speed ``v`` (capped at ``kappa_max``)."""
    k = min(ay / v**2, kappa_max)
    ramp = 8.0

    def kap(s):
        def bump(s0):  # 0 -> 1 -> 0 over [s0, s0 + arc] with cosine ramps
            u = s - s0
            if u < 0 or u > arc:
                return 0.0
            return 0.5 * (1 - math.cos(math.pi * min(u, ramp) / ramp)) if u < ramp else (
                0.5 * (1 - math.cos(math.pi * (arc - u) / ramp)) if u > arc - ramp else 1.0)

        return k * (bump(straight) - bump(straight + arc))

    return path_reference(kap, 2 * straight + 2 * arc, v, **kw)


def random_path(v: float, ay: float, length: float = 140.0, seed: int = 0, kappa_max: float = 0.1, **kw) -> PathRef:
    """A smooth random curvature profile (sum of three sinusoids in arc length), scaled so its peak lateral
    acceleration at speed ``v`` is ``ay``; starts with a 10 m straight."""
    rng = np.random.default_rng(seed)
    f = rng.uniform(1 / 60, 1 / 20, 3)
    ph = rng.uniform(0, 2 * math.pi, 3)
    amp = rng.uniform(0.5, 1.0, 3)
    s = np.linspace(0, length, 2000)
    raw = lambda s: float(sum(a * math.sin(2 * math.pi * fi * s + p) for a, fi, p in zip(amp, f, ph)))
    peak = max(abs(raw(si)) for si in s)
    k = min(ay / v**2, kappa_max) / peak

    def kap(si):
        w = min(1.0, max(0.0, (si - 10.0) / 10.0))  # straight, then a 10 m blend
        return w * k * raw(si)

    return path_reference(kap, length, v, **kw)


def path_errors(ref: PathRef, x: float, y: float, yaw: float, hint: int = 0, window: int = 200) -> tuple[float, float, int]:
    """Lateral error to the nearest reference point (left positive) and heading error, searching ``window`` points
    ahead of and behind ``hint``; returns (lateral, heading, nearest index)."""
    lo, hi = max(0, hint - window), min(len(ref), hint + window)
    d2 = (ref.x[lo:hi] - x) ** 2 + (ref.y[lo:hi] - y) ** 2
    j = lo + int(np.argmin(d2))
    c, s = math.cos(ref.yaw[j]), math.sin(ref.yaw[j])
    lat = -s * (x - ref.x[j]) + c * (y - ref.y[j])
    head = (yaw - ref.yaw[j] + math.pi) % (2 * math.pi) - math.pi
    return lat, head, j


def pure_pursuit(ref: PathRef, obs: dict, j: int, wheelbase: float, max_steer: float, lookahead: float = 1.0,
                 min_look: float = 4.0, k_v: float = 1.0) -> tuple[float, float]:
    """Geometric baseline (no vehicle model): steering toward the path point ``lookahead`` s ahead (at least
    ``min_look`` m), acceleration from the speed error. Returns (normalised steering command, a_cmd)."""
    look = max(min_look, lookahead * max(obs["vx"], 1.0))
    k = min(len(ref) - 1, j + int(look / (ref.v * ref.dt)))
    dx, dy = ref.x[k] - obs["x"], ref.y[k] - obs["y"]
    alpha = math.atan2(dy, dx) - obs["yaw"]
    ld = max(math.hypot(dx, dy), 1e-3)
    delta = math.atan2(2 * wheelbase * math.sin(alpha), ld)
    return max(-1.0, min(1.0, delta / max_steer)), k_v * (ref.v - obs["vx"])


# ---------------------------------------------------------------------- single-track identification


def axle_forces(log: dict, m: float, I_z: float, a: float, b: float) -> dict:
    """Single-track inverse dynamics on logged arrays (vx, vy, r, delta, ay, rdot): the axle lateral forces that make
    the bicycle model reproduce the measured lateral and yaw accelerations,
    m ay = Fyf cos(delta) + Fyr,  I_z rdot = a Fyf cos(delta) - b Fyr,
    with the bicycle slip angles. Load transfer, roll, combined slip and soil effects all end up in these."""
    vx = np.maximum(log["vx"], 1.0)
    fyf_c = (I_z * log["rdot"] + b * m * log["ay"]) / (a + b)
    fyr = (a * m * log["ay"] - I_z * log["rdot"]) / (a + b)
    return {"alpha_f": np.arctan((log["vy"] + a * log["r"]) / vx) - log["delta"],
            "alpha_r": np.arctan((log["vy"] - b * log["r"]) / vx),
            "fyf": fyf_c / np.cos(log["delta"]), "fyr": fyr}
