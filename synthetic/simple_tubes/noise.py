"""Background and distractors for simple_v1 (none of it enters the ground truth).

Layers, composed per channel on the 4-bit raw (0..15):
  1. haze          smooth fractal Perlin field quantised to levels 0..hmax (hmax <= 3), all channels
  2. blobs         1-4 large irregular distractor blobs, dim Perlin texture (levels 1..4), one colour each
  3. neurons       drawn by render.py on top (per-channel max; levels 10..15)
  4. bright confetti   small spots (1-3 voxels across), levels 8..15, random colour, per-channel max
  5. dark confetti     small spots that clamp all channels to 0..1 (they punch holes in neurons too)

Confetti centres are drawn with probability proportional to a second Perlin field raised to
a power, so density varies inside a volume; the overall density also varies per volume.
Perlin noise is computed on a coarse grid (1/4 resolution, 1/2 for blob texture) and
trilinearly upsampled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .render_util import COLOURS, ball_offsets, flat_index


@dataclass
class NoiseParams:
    coarse: int = 4                       # Perlin grid = shape / coarse, then upsampled
    haze_period: tuple = (100.0, 250.0)   # largest octave, voxels
    haze_octaves: int = 3
    haze_max: tuple = (1, 3)              # top haze level, inclusive range
    haze_threshold: tuple = (0.3, 0.7)    # normalised field below this is level 0
    blobs: tuple = (1, 4)                 # distractor blobs per volume, inclusive
    blob_radius: tuple = (30.0, 100.0)
    blob_axis_scale: tuple = (0.6, 1.4)   # per-axis stretch of the blob
    blob_wobble: float = 0.35             # radius modulation by Perlin
    blob_level_max: tuple = (2, 4)        # texture levels 1..max
    blob_texture_period: tuple = (6.0, 12.0)
    density_period: tuple = (60.0, 200.0)
    density_gamma: tuple = (1.0, 3.0)     # contrast of dense vs sparse regions
    bright_density: tuple = (1e-5, 2e-3)  # spots per voxel, log-uniform per volume
    dark_density: tuple = (1e-5, 2e-3)
    spot_radius: tuple = (0.5, 1.5)       # 1-3 voxels across
    bright_level: tuple = (8, 15)
    dark_level: tuple = (0, 1)


# ---------------------------------------------------------------- Perlin

def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin3d(shape, period: float, rng: np.random.Generator) -> np.ndarray:
    """Gradient (Perlin) noise on a grid, lattice spacing `period` grid cells.
    Returns float32, roughly in [-1, 1]."""
    shape = tuple(int(n) for n in shape)
    n_lat = [int(math.ceil(n / period)) + 2 for n in shape]
    g = rng.normal(size=(*n_lat, 3)).astype(np.float32)
    g /= np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12
    off = rng.uniform(0, 1, size=3)          # random lattice phase per call
    t = [np.arange(n, dtype=np.float32) / period + o for n, o in zip(shape, off)]
    i = [np.floor(a).astype(np.int64) for a in t]
    f = [(a - b).astype(np.float32) for a, b in zip(t, i)]
    w = [_fade(a) for a in f]
    iz, iy, ix = i[0][:, None, None], i[1][None, :, None], i[2][None, None, :]
    fz, fy, fx = f[0][:, None, None], f[1][None, :, None], f[2][None, None, :]
    wz, wy, wx = w[0][:, None, None], w[1][None, :, None], w[2][None, None, :]
    out = np.zeros(shape, dtype=np.float32)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                G = g[iz + dz, iy + dy, ix + dx]
                dot = G[..., 0] * (fz - dz) + G[..., 1] * (fy - dy) + G[..., 2] * (fx - dx)
                out += ((wz if dz else 1 - wz) * (wy if dy else 1 - wy)
                        * (wx if dx else 1 - wx)) * dot
    return out * np.float32(1.7)


def fbm(shape, period: float, octaves: int, rng) -> np.ndarray:
    """Fractal sum of Perlin octaves, normalised to [0, 1] by its 1st/99th percentiles."""
    total = np.zeros(tuple(int(n) for n in shape), dtype=np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        total += np.float32(amp) * perlin3d(shape, max(period / 2 ** o, 1.5), rng)
        norm += amp
        amp *= 0.5
    lo, hi = np.percentile(total[::2, ::2, ::2], [1, 99])
    return np.clip((total - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def upsample(field: np.ndarray, factor: int, shape) -> np.ndarray:
    """Trilinear upsample by an integer factor, cropped to `shape`."""
    up = ndimage.zoom(field, factor, order=1, mode="nearest", grid_mode=True)
    return up[: shape[0], : shape[1], : shape[2]]


def _coarse_shape(shape, c):
    return tuple(int(math.ceil(n / c)) for n in shape)


# ---------------------------------------------------------------- layers

def add_haze(raw, rng, P: NoiseParams) -> dict:
    shape = raw.shape[1:]
    period = rng.uniform(*P.haze_period) / P.coarse
    h = fbm(_coarse_shape(shape, P.coarse), period, P.haze_octaves, rng)
    hmax = int(rng.integers(P.haze_max[0], P.haze_max[1] + 1))
    t0 = float(rng.uniform(*P.haze_threshold))
    h = upsample(h, P.coarse, shape)
    level = np.clip(np.floor((h - t0) / (1.0 - t0) * (hmax + 1)), 0, hmax).astype(np.uint8)
    del h
    for ch in range(raw.shape[0]):
        np.maximum(raw[ch], level, out=raw[ch])
    frac = float((level > 0).mean())
    return {"haze_max": hmax, "haze_threshold": round(t0, 4),
            "haze_period": round(period * P.coarse, 2), "haze_fraction": round(frac, 4)}


def add_blobs(raw, rng, P: NoiseParams) -> list:
    shape = np.asarray(raw.shape[1:])
    out = []
    for _ in range(int(rng.integers(P.blobs[0], P.blobs[1] + 1))):
        c = rng.uniform(0, shape - 1)
        R = float(rng.uniform(*P.blob_radius))
        axes = rng.uniform(*P.blob_axis_scale, size=3) * R
        reach = axes * (1 + P.blob_wobble)
        lo = np.maximum(np.floor(c - reach).astype(int), 0)
        hi = np.minimum(np.ceil(c + reach).astype(int) + 1, shape)
        box = tuple(int(b - a) for a, b in zip(lo, hi))
        colour = COLOURS[int(rng.integers(len(COLOURS)))]
        lmax = int(rng.integers(P.blob_level_max[0], P.blob_level_max[1] + 1))
        tper = float(rng.uniform(*P.blob_texture_period))
        if min(box) <= 0:
            continue
        # shape: ellipsoid radius modulated by a mid-frequency Perlin field
        cs = _coarse_shape(box, P.coarse)
        wob = upsample(perlin3d(cs, max(R / 2 / P.coarse, 1.5), rng), P.coarse, box)
        zz, yy, xx = (np.arange(a, b, dtype=np.float32) for a, b in zip(lo, hi))
        d = (((zz[:, None, None] - c[0]) / axes[0]) ** 2
             + ((yy[None, :, None] - c[1]) / axes[1]) ** 2
             + ((xx[None, None, :] - c[2]) / axes[2]) ** 2)
        inside = np.sqrt(d) <= 1.0 + P.blob_wobble * wob
        del d, wob
        # texture: fine Perlin at half resolution, levels 1..lmax
        hs = _coarse_shape(box, 2)
        tex = upsample(fbm(hs, max(tper / 2, 1.5), 2, rng), 2, box)
        val = np.where(inside, 1 + np.minimum(np.floor(tex * lmax), lmax - 1), 0).astype(np.uint8)
        del tex
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        for ch in range(3):
            if colour[ch]:
                np.maximum(raw[ch][sl], val, out=raw[ch][sl])
        out.append({"center": np.round(c, 1).tolist(), "radius": round(R, 2),
                    "axes": np.round(axes, 2).tolist(), "colour": list(colour),
                    "level_max": lmax, "voxels": int(inside.sum())})
    return out


def _density_field(shape, rng, P: NoiseParams):
    cs = _coarse_shape(shape, P.coarse)
    period = rng.uniform(*P.density_period) / P.coarse
    gamma = float(rng.uniform(*P.density_gamma))
    d = fbm(cs, period, 2, rng) ** gamma
    p = d.ravel().astype(np.float64)
    p /= p.sum()
    return cs, p, {"density_period": round(period * P.coarse, 2), "density_gamma": round(gamma, 3)}


def _spot_centres(n, cs, p, shape, rng, c):
    cells = rng.choice(len(p), size=n, p=p)
    cz, cy, cx = np.unravel_index(cells, cs)
    pos = (np.stack([cz, cy, cx], 1) + rng.uniform(0, 1, size=(n, 3))) * c - 0.5
    return np.clip(pos, -0.49, np.asarray(shape) - 0.51)


def _spot_voxels(centres, radii, shape, batch=20000):
    """Yield (spot index, flat voxel index) pairs for voxels within radius of each centre."""
    off = ball_offsets(2)
    for s in range(0, len(centres), batch):
        c = centres[s:s + batch]
        r = radii[s:s + batch]
        base = np.rint(c).astype(np.int64)
        v = base[:, None, :] + off[None, :, :]
        keep = np.linalg.norm(v - c[:, None, :], axis=-1) <= r[:, None]
        keep &= np.all((v >= 0) & (v < np.asarray(shape)), axis=-1)
        spot = np.broadcast_to(np.arange(s, s + len(c))[:, None], keep.shape)[keep]
        yield spot, flat_index(v[keep], shape)


def add_confetti(raw, rng, P: NoiseParams, kind: str, field) -> dict:
    """kind = 'bright' (per-channel max) or 'dark' (clamp all channels down)."""
    shape = raw.shape[1:]
    cs, p = field
    dens_rng = P.bright_density if kind == "bright" else P.dark_density
    density = float(math.exp(rng.uniform(math.log(dens_rng[0]), math.log(dens_rng[1]))))
    n = int(rng.poisson(density * float(np.prod(shape))))
    stats = {"density": density, "spots": n, "voxels": 0}
    if n == 0:
        return stats
    centres = _spot_centres(n, cs, p, shape, rng, P.coarse)
    radii = rng.uniform(*P.spot_radius, size=n)
    flat_raw = raw.reshape(3, -1)
    if kind == "bright":
        levels = rng.integers(P.bright_level[0], P.bright_level[1] + 1, size=n).astype(np.uint8)
        colours = np.asarray(COLOURS, dtype=bool)[rng.integers(len(COLOURS), size=n)]
        for spot, idx in _spot_voxels(centres, radii, shape):
            for ch in range(3):
                m = colours[spot, ch]
                if m.any():
                    sel = idx[m]
                    flat_raw[ch, sel] = np.maximum(flat_raw[ch, sel], levels[spot[m]])
            stats["voxels"] += len(idx)
    else:
        levels = rng.integers(P.dark_level[0], P.dark_level[1] + 1, size=n).astype(np.uint8)
        for spot, idx in _spot_voxels(centres, radii, shape):
            for ch in range(3):
                flat_raw[ch, idx] = np.minimum(flat_raw[ch, idx], levels[spot])
            stats["voxels"] += len(idx)
    return stats


def background(shape, rng, P: NoiseParams) -> tuple[np.ndarray, dict]:
    """Haze + distractor blobs, (3, Z, Y, X) uint8. Neurons and confetti come after."""
    raw = np.zeros((3, *shape), dtype=np.uint8)
    info = {"haze": add_haze(raw, rng, P), "blobs": add_blobs(raw, rng, P)}
    return raw, info


def confetti(raw, rng, P: NoiseParams) -> dict:
    """Bright then dark confetti, both following one Perlin density field."""
    cs, p, dinfo = _density_field(raw.shape[1:], rng, P)
    bright = add_confetti(raw, rng, P, "bright", (cs, p))
    dark = add_confetti(raw, rng, P, "dark", (cs, p))
    return {**dinfo, "bright": bright, "dark": dark}
