"""Rasterise SWC skeletons into per-neuron voxel masks (ground truth for SWC2IMG renders).

SWC2IMG only returns an image; its ground truth is the SWC. To score or train on SWC2IMG renders we draw each
neuron's tubes (nodes joined by edges, interpolated radius) into the same voxel grid as the cropped render.
"""

from __future__ import annotations

import numpy as np


def _edge_points(nodes: np.ndarray, step_um: float):
    """Sample points along every parent-child edge (plus isolated nodes): xyz (n, 3) and radius (n,)."""
    xyz = np.stack([nodes["x"], nodes["y"], nodes["z"]], axis=1)
    r = nodes["r"].astype(float)
    order = np.argsort(nodes["id"])
    sorted_ids = nodes["id"][order]
    pos = np.searchsorted(sorted_ids, nodes["parent"])
    pos = np.clip(pos, 0, len(sorted_ids) - 1)
    has_parent = (nodes["parent"] >= 0) & (sorted_ids[pos] == nodes["parent"])
    child = np.nonzero(has_parent)[0]
    parent = order[pos[has_parent]]

    a, b = xyz[child], xyz[parent]
    ra, rb = r[child], r[parent]
    n = np.maximum(np.ceil(np.linalg.norm(b - a, axis=1) / step_um).astype(int), 1) + 1
    seg = np.repeat(np.arange(len(child)), n)
    start = np.repeat(np.cumsum(n) - n, n)
    t = (np.arange(n.sum()) - start) / np.repeat(n - 1, n)
    pts = a[seg] + (b[seg] - a[seg]) * t[:, None]
    rad = ra[seg] + (rb[seg] - ra[seg]) * t
    # nodes with no edge (isolated roots) still get drawn
    lonely = np.ones(len(nodes), bool)
    lonely[child] = False
    lonely[parent] = False
    return np.concatenate([pts, xyz[lonely]]), np.concatenate([rad, r[lonely]])


def rasterize(nodes: np.ndarray, lo_xyz, shape_zyx, voxel_um: float, pad_vox: float = 0.0,
              min_radius_vox: float = 0.5, max_batch: int = 20_000_000) -> np.ndarray:
    """Flat (z, y, x) indices of voxels within each tube.

    Same rule as mcfo_sim's ``render.rasterize_mask`` (so masks are comparable with v1's ``gt_instances``): each
    centreline sample's voxel ``floor(p)`` is dilated by a ball of integer radius ``k = round(max(r / voxel, 0.5))``
    (|offset|^2 <= k^2 + 0.5). ``pad_vox`` adds to k (used to exclude a margin around neurons when measuring
    background).
    """
    if len(nodes) == 0:
        return np.zeros(0, np.int64)
    Z, Y, X = map(int, shape_zyx)
    pts, rad = _edge_points(nodes, step_um=voxel_um / 2)
    p = (pts - np.asarray(lo_xyz, float)) / voxel_um          # voxel coords, x y z
    k = np.round(np.maximum(rad / voxel_um, min_radius_vox)).astype(int) + int(round(pad_vox))
    base = np.floor(p).astype(np.int64)
    inside = np.all((base >= -k[:, None]) & (base < np.array([X, Y, Z]) + k[:, None]), axis=1)
    base, k = base[inside], k[inside]
    out = []
    for kk in np.unique(k):
        bb = np.unique(base[k == kk], axis=0)
        g = np.arange(-kk, kk + 1)
        offs = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        offs = offs[(offs ** 2).sum(1) <= kk ** 2 + 0.5]
        bs = max(1, max_batch // len(offs))
        for i in range(0, len(bb), bs):
            c = (bb[i:i + bs, None, :] + offs[None]).reshape(-1, 3)        # xyz
            c = c[(c[:, 0] >= 0) & (c[:, 0] < X) & (c[:, 1] >= 0) & (c[:, 1] < Y) & (c[:, 2] >= 0) & (c[:, 2] < Z)]
            out.append(np.unique(c[:, 2] * (Y * X) + c[:, 1] * X + c[:, 0]))
    return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)
