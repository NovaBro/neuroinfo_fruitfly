"""Minimal SWC reading, writing, merging and clipping (no external dependencies besides numpy)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Columns: id, type, x, y, z, radius, parent
SWC_DTYPE = [("id", "i8"), ("type", "i8"), ("x", "f8"), ("y", "f8"), ("z", "f8"),
             ("r", "f8"), ("parent", "i8")]


def read_swc(path: str | os.PathLike) -> np.ndarray:
    rows = []
    with open(path) as f:
        for ln in f:
            s = ln.split()
            if not s or s[0].startswith("#"):
                continue
            rows.append((int(float(s[0])), int(float(s[1])), float(s[2]), float(s[3]),
                         float(s[4]), float(s[5]), int(float(s[6]))))
    return np.array(rows, dtype=SWC_DTYPE)


def write_swc(path: str | os.PathLike, nodes: np.ndarray, header: str = "") -> None:
    with open(path, "w") as f:
        for h in header.splitlines():
            f.write(f"# {h}\n")
        f.write("# id type x y z radius parent\n")
        for n in nodes:
            f.write(f"{n['id']} {n['type']} {n['x']:.4f} {n['y']:.4f} {n['z']:.4f} "
                    f"{n['r']:.4f} {n['parent']}\n")


def merge(neurons: list[np.ndarray]) -> np.ndarray:
    """Concatenate several SWC trees, renumbering ids so they don't clash (roots stay roots)."""
    out, offset = [], 0
    for nodes in neurons:
        remap = {old: offset + i + 1 for i, old in enumerate(nodes["id"])}
        n = nodes.copy()
        n["id"] = [remap[i] for i in nodes["id"]]
        n["parent"] = [remap.get(p, -1) for p in nodes["parent"]]
        out.append(n)
        offset += len(nodes)
    return np.concatenate(out) if out else np.zeros(0, dtype=SWC_DTYPE)


def clip_to_box(nodes: np.ndarray, lo_xyz, hi_xyz, margin_um: float = 5.0) -> np.ndarray:
    """Keep nodes inside [lo - margin, hi + margin]; children of dropped nodes become roots.

    The margin lets cable that crosses the box edge render right up to the edge; the caller
    crops the image back to the exact box afterwards.
    """
    lo = np.asarray(lo_xyz, float) - margin_um
    hi = np.asarray(hi_xyz, float) + margin_um
    xyz = np.stack([nodes["x"], nodes["y"], nodes["z"]], axis=1)
    keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    kept = nodes[keep].copy()
    kept_ids = set(kept["id"].tolist())
    kept["parent"] = [p if p in kept_ids else -1 for p in kept["parent"]]
    return kept


def with_anchors(nodes: np.ndarray, lo_xyz, hi_xyz, radius_um: float = 1e-3) -> np.ndarray:
    """Append two isolated, effectively invisible nodes at the box corners.

    SWC2IMG sizes the image to the data extent plus a fixed border. Anchors pin that extent
    to the box, and their positions in the corrected SWC give the exact crop offset.
    """
    start = int(nodes["id"].max()) + 1 if len(nodes) else 1
    anchors = np.array([
        (start, 0, *map(float, lo_xyz), radius_um, -1),
        (start + 1, 0, *map(float, hi_xyz), radius_um, -1),
    ], dtype=SWC_DTYPE)
    return np.concatenate([nodes, anchors])
