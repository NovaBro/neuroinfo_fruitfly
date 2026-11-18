"""Small helpers shared by render.py, noise.py and verify.py."""
from __future__ import annotations

import itertools

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# The 7 non-empty subsets of {ch0, ch1, ch2}.
COLOURS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))

# 13 of the 26 neighbour offsets (one of each +/- pair) for undirected adjacency.
_HALF26 = np.array([o for o in itertools.product((-1, 0, 1), repeat=3)
                    if o > (0, 0, 0)], dtype=np.int64)


def ball_offsets(R: int) -> np.ndarray:
    """Integer offsets within R + 0.9 of the origin: covers every voxel within R of a
    point whose nearest voxel centre is the origin (that centre is <= 0.87 away)."""
    g = np.arange(-R - 1, R + 2)
    off = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return off[np.linalg.norm(off, axis=1) <= R + 0.9]


def flat_index(coords: np.ndarray, shape) -> np.ndarray:
    return np.ravel_multi_index(np.asarray(coords).T, tuple(shape))


def count_components(flat: np.ndarray, shape) -> int:
    """26-connected components of a voxel set given as flat indices.

    Builds the adjacency on the voxel list itself (13 neighbour offsets, searchsorted),
    so the cost scales with the number of voxels, not with the bounding box."""
    flat = np.unique(np.asarray(flat, dtype=np.int64))
    n = len(flat)
    if n == 0:
        return 0
    shape = np.asarray(shape, dtype=np.int64)
    coords = np.stack(np.unravel_index(flat, tuple(shape)), axis=1)
    rows, cols = [], []
    for off in _HALF26:
        nb = coords + off
        ok = np.all((nb >= 0) & (nb < shape), axis=1)
        nbf = np.ravel_multi_index(nb[ok].T, tuple(shape))
        pos = np.minimum(np.searchsorted(flat, nbf), n - 1)
        hit = flat[pos] == nbf
        rows.append(np.nonzero(ok)[0][hit])
        cols.append(pos[hit])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    graph = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    return int(connected_components(graph, directed=False)[0])


def gt_channel(flat: np.ndarray, label: int, shape) -> np.ndarray:
    """Dense (Z, Y, X) uint8 channel holding `label` on the given voxels."""
    out = np.zeros(int(np.prod(shape)), dtype=np.uint8)
    out[flat] = label
    return out.reshape(tuple(shape))
