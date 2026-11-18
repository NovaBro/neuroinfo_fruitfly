"""Rasterise tube trees and compose a simple_v1 sample (plan §3.5–3.7, noise in noise.py).

Raw is 3 channels of 4-bit values (0..15) in uint8. Every neuron gets one of the 7
MCFO on/off colour combinations and one brightness level (10..15) on its "on" channels;
overlaps take the per-channel max. Order of layers: haze + distractor blobs, neurons,
bright confetti, dark confetti (see noise.py). Ground truth follows FISBe:
gt_instances[n] holds n + 1 inside neuron n and 0 elsewhere (overlaps allowed); noise
never enters it. Instances are kept as flat voxel indices, so a 400x600x600 sample
never holds all N dense masks at once.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree

from . import noise
from .geometry import GeometryParams, Neuron, build_neuron
from .render_util import COLOURS, ball_offsets, count_components, flat_index


@dataclass
class RenderParams:
    neurons: tuple = (10, 30)   # neurons per volume, inclusive
    level: tuple = (10, 15)     # 4-bit brightness per neuron, inclusive
    min_voxels: int = 500       # redraw a neuron with fewer voxels inside the volume
    max_tries: int = 20
    noise: bool = True          # haze, distractor blobs, confetti (noise.py)
    confetti: bool = True       # bake confetti (False for train: added on the GPU in training)


def sample_seed(base_seed: int, index: int) -> int:
    return base_seed * 1_000_000 + index


def rasterize_branch(points: np.ndarray, radius: np.ndarray, shape) -> np.ndarray:
    """Flat indices of voxels whose centre lies within radius[i] of the nearest
    centreline point i (exact sub-voxel distance to the resampled centreline).

    Per branch, so the nearest point carries the right radius; across branches of
    different radius a single nearest-point lookup would notch the junctions.
    """
    shp = np.asarray(shape, dtype=np.int64)
    r_max = float(radius.max())
    reach = np.all((points > -0.5 - r_max - 1) & (points < shp - 0.5 + r_max + 1), axis=1)
    if not reach.any():
        return np.empty(0, dtype=np.int64)
    pts, rad = points[reach], radius[reach]
    off = ball_offsets(int(math.ceil(r_max)))
    base = np.unique(np.rint(pts).astype(np.int64), axis=0)
    cand = (base[:, None, :] + off[None, :, :]).reshape(-1, 3)
    cand = cand[np.all((cand >= 0) & (cand < shp), axis=1)]
    if len(cand) == 0:
        return np.empty(0, dtype=np.int64)
    flat = np.unique(flat_index(cand, shape))
    coords = np.stack(np.unravel_index(flat, tuple(shape)), axis=1).astype(np.float64)
    d, idx = cKDTree(pts).query(coords, k=1, distance_upper_bound=r_max + 1e-9)
    hit = np.isfinite(d)
    inside = np.zeros(len(flat), dtype=bool)
    inside[hit] = d[hit] <= rad[idx[hit]] + 1e-9
    return flat[inside]


def rasterize_neuron(neuron: Neuron, shape) -> np.ndarray:
    parts = [rasterize_branch(b.points, b.radius, shape) for b in neuron.branches]
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)


def skeleton_voxels(neuron: Neuron, shape) -> np.ndarray:
    """Centreline voxels inside the volume, (k, 3) int16 zyx. Each lies within 0.87
    of the centreline, so inside the tube (radius >= 1.5)."""
    pts = np.vstack([b.points for b in neuron.branches])
    vox = np.rint(pts).astype(np.int64)
    vox = vox[np.all((vox >= 0) & (vox < np.asarray(shape)), axis=1)]
    return np.unique(vox, axis=0).astype(np.int16)


def make_sample(index: int, base_seed: int = 0, gp: GeometryParams | None = None,
                rp: RenderParams | None = None, npar: noise.NoiseParams | None = None) -> dict:
    """Generate one sample. Deterministic in (base_seed, index, params).

    Returns raw (3,Z,Y,X) uint8, per-instance flat voxel indices (`flats`; channel n of
    gt_instances = n + 1 on flats[n]), per-instance metadata, skeleton voxels, noise
    info and generation stats. Geometry and noise use separate random streams.
    """
    gp = gp or GeometryParams()
    rp = rp or RenderParams()
    npar = npar or noise.NoiseParams()
    shape = tuple(int(n) for n in gp.shape)
    seed = sample_seed(base_seed, index)
    rng = np.random.default_rng(seed)
    n_target = int(rng.integers(rp.neurons[0], rp.neurons[1] + 1))

    instances = []
    redraws = 0
    skipped = 0
    for _ in range(n_target):
        accepted = None
        for _ in range(rp.max_tries):
            neuron = build_neuron(rng, gp)
            if neuron is not None:
                flat = rasterize_neuron(neuron, shape)
                if flat.size >= rp.min_voxels:
                    accepted = (neuron, flat)
                    break
            redraws += 1
        if accepted is None:
            skipped += 1
            continue
        neuron, flat = accepted
        n_cc = count_components(flat, shape)
        if n_cc != 1:
            raise RuntimeError(
                f"sample {index} (seed {seed}): neuron {len(instances)} has {n_cc} "
                "26-connected pieces; truncation should make this impossible (generator bug)")
        colour = COLOURS[int(rng.integers(len(COLOURS)))]
        level = int(rng.integers(rp.level[0], rp.level[1] + 1))
        instances.append({"neuron": neuron, "flat": flat, "colour": colour, "level": level})
    if not instances:
        raise RuntimeError(f"sample {index} (seed {seed}): no neuron passed the size check")

    nrng = np.random.default_rng([seed, 1])
    if rp.noise:
        raw, noise_info = noise.background(shape, nrng, npar)
    else:
        raw, noise_info = np.zeros((3, *shape), dtype=np.uint8), None
    raw_flat = raw.reshape(3, -1)
    for inst in instances:
        flat = inst["flat"]
        for ch in range(3):
            if inst["colour"][ch]:
                raw_flat[ch, flat] = np.maximum(raw_flat[ch, flat], inst["level"])
    if rp.noise:
        noise_info["confetti"] = noise.confetti(raw, nrng, npar) if rp.confetti else None

    meta = []
    skeletons = {}
    for n, inst in enumerate(instances):
        neuron = inst["neuron"]
        meta.append({
            "label": n + 1,
            "colour": list(inst["colour"]),
            "level": inst["level"],
            "voxels": int(inst["flat"].size),
            **neuron.to_json(),
        })
        skeletons[f"instance_{n + 1:03d}"] = skeleton_voxels(neuron, shape)
    return {
        "index": index,
        "seed": seed,
        "base_seed": base_seed,
        "shape": shape,
        "raw": raw,
        "flats": [inst["flat"] for inst in instances],
        "instances": meta,
        "skeletons": skeletons,
        "noise": noise_info,
        "stats": {"n_target": n_target, "n_instances": len(instances), "redraws": redraws,
                  "skipped": skipped},
        "params": {"geometry": asdict(gp), "render": asdict(rp),
                   "noise": asdict(npar) if rp.noise else None},
    }
