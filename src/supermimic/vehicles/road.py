"""Random rough roads for the Chrono rigid terrain (VEHICLE_PROPOSAL.md section 11.15, G-AS1).

A road is a height grid generated from a seed: a sum of random plane waves (wavelengths 2-25 m, different on the left
and right wheel tracks because the waves run at random angles) plus a few smooth potholes and bumps, flat for the
first stretch so the vehicle can settle. ``write_bmp`` stores it as the 8-bit grey-scale heightmap that
``veh.RigidTerrain.AddPatch`` reads; ``Road.height`` interpolates the same quantised grid, so controllers preview
exactly the surface Chrono drives on (Chrono triangulates the grid; bilinear interpolation differs by < 1 mm here).

Chrono's mapping (probed): pixel column c lies at x = cx - W/2 + c W/(cols-1), row r at y = cy + H/2 - r H/(rows-1),
height = h_min + (h_max - h_min) value / 255.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Road:
    z: np.ndarray  # (rows, cols) heights [m], row 0 at y = cy + H/2
    center: tuple[float, float]
    size: tuple[float, float]  # (W along x, H along y) [m]
    h_range: tuple[float, float]

    @property
    def spacing(self) -> tuple[float, float]:
        rows, cols = self.z.shape
        return self.size[0] / (cols - 1), self.size[1] / (rows - 1)

    def height(self, x, y):
        """Bilinear height at world points (numpy arrays of any shape); clamped at the edges."""
        rows, cols = self.z.shape
        dx, dy = self.spacing
        u = np.clip((np.asarray(x) - (self.center[0] - self.size[0] / 2)) / dx, 0, cols - 1.001)
        v = np.clip(((self.center[1] + self.size[1] / 2) - np.asarray(y)) / dy, 0, rows - 1.001)
        c0, r0 = np.floor(u).astype(int), np.floor(v).astype(int)
        fu, fv = u - c0, v - r0
        z = self.z
        return ((1 - fv) * ((1 - fu) * z[r0, c0] + fu * z[r0, c0 + 1]) + fv * ((1 - fu) * z[r0 + 1, c0] + fu * z[r0 + 1, c0 + 1]))

    def chrono_cfg(self, path: str) -> dict:
        return {"heightmap": path, "hm_center": list(self.center), "hm_size": list(self.size), "hm_range": list(self.h_range)}


def make_road(seed: int, rms: float = 0.04, n_holes: int = 12, hole_depth=(0.05, 0.10), center=(110.0, 0.0), size=(240.0, 40.0), spacing: float = 0.2,
              flat_until: float = 20.0, ramp: float = 10.0, h_range=(-0.25, 0.25)) -> Road:
    rng = np.random.default_rng(seed)
    cols, rows = round(size[0] / spacing) + 1, round(size[1] / spacing) + 1
    x = center[0] - size[0] / 2 + np.arange(cols) * size[0] / (cols - 1)
    y = center[1] + size[1] / 2 - np.arange(rows) * size[1] / (rows - 1)
    X, Y = np.meshgrid(x, y)
    z = np.zeros_like(X)
    lam = np.exp(rng.uniform(np.log(2.0), np.log(25.0), 40))
    amp = lam ** 1.0  # rougher at long wavelengths (a road-like spectrum), rescaled to the target RMS below
    ang = rng.uniform(-np.pi, np.pi, 40)
    ph = rng.uniform(0, 2 * np.pi, 40)
    for L, A, th, p in zip(lam, amp, ang, ph):
        z += A * np.sin(2 * np.pi * (np.cos(th) * X + np.sin(th) * Y) / L + p)
    z *= rms / z.std()
    for _ in range(n_holes):  # smooth potholes and bumps near the driving band
        x0, y0 = rng.uniform(flat_until + ramp, center[0] + size[0] / 2 - 5), rng.uniform(-5, 5)
        r, d = rng.uniform(0.4, 1.0), rng.choice([-1, 1]) * rng.uniform(*hole_depth)
        z += d * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * r**2))
    z *= np.clip((X - flat_until) / ramp, 0, 1)  # flat start, then fade in
    q = np.clip(np.round((z - h_range[0]) / (h_range[1] - h_range[0]) * 255), 0, 255).astype(np.uint8)
    return Road(h_range[0] + (h_range[1] - h_range[0]) * q / 255.0, tuple(center), tuple(size), tuple(h_range)), q


def write_bmp(path: str | Path, img: np.ndarray) -> None:
    """8-bit grey-scale (palettised) BMP; img (rows, cols) uint8, row 0 = top of the image."""
    h, w = img.shape
    pad = (4 - w % 4) % 4
    row = w + pad
    off = 14 + 40 + 1024
    with open(path, "wb") as f:
        f.write(b"BM" + struct.pack("<IHHI", off + row * h, 0, 0, off))
        f.write(struct.pack("<IiiHHIIiiII", 40, w, h, 1, 8, 0, row * h, 2835, 2835, 256, 0))
        f.write(b"".join(bytes((i, i, i, 0)) for i in range(256)))
        f.writelines(img[r].tobytes() + b"\0" * pad for r in range(h - 1, -1, -1))  # bottom-up rows


def road_files(seeds, out_dir: str | Path, **kw) -> list[tuple[Road, str]]:
    """Generate (or regenerate: deterministic) the roads for these seeds as BMPs under out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = []
    for s in seeds:
        road, img = make_road(int(s), **kw)
        p = out / f"road_{int(s)}.bmp"
        write_bmp(p, img)
        res.append((road, str(p.resolve())))
    return res
