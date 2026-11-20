#!/usr/bin/env python3
"""experiment_v3 P2: seam check for same-padding predictions (predict_no_gp.py ``test_same_margin``).

Compares two predictions of the same volume, e.g. A = abutting tiles (margin 0, output tile T) and
B = overlapping tiles (margin m), and reports how they differ as a function of the distance to A's tile
borders. If A has seams, the difference is concentrated within a few voxels of A's borders. The same
comparison of B (margin 32) vs C (margin 64) shows whether 32 voxels of context are enough.

  python sbatch/ppp/tools/compare_same_margin.py A.zarr B.zarr --tile 160 [--key volumes/pred_numinst]

Reports, per distance bin: voxels, mean |P(fg)_A - P(fg)_B|, and the fraction of voxels whose
foreground call (P(fg) > 0.5) differs. P(fg) = 1 - channel 0 of pred_numinst.
"""
import argparse

import numpy as np
import zarr


def fg_prob(z, key, sl):
    a = np.asarray(z[key][(slice(None),) + sl], dtype=np.float32)
    return 1.0 - a[0] if a.shape[0] > 1 else a[0]


def axis_dist(n, tile):
    """Per index along one axis: voxels to the nearest interior tile border (0 = touches it). predict_no_gp
    writes output tile k to [k*tile, (k+1)*tile) in volume coordinates, so borders sit before k*tile, k >= 1.
    Capped at ``tile``; volume faces are not borders."""
    x = np.arange(n)[:, None]
    planes = np.arange(tile, n, tile)[None, :]
    if planes.size == 0:
        return np.full(n, tile, dtype=np.int32)
    d = np.where(x < planes, planes - 1 - x, x - planes).min(axis=1)
    return np.minimum(d, tile).astype(np.int32)


def border_dist(shape, tile):
    d = None
    for ax, n in enumerate(shape):
        sh = [1] * len(shape)
        sh[ax] = n
        a = axis_dist(n, tile).reshape(sh)
        d = a if d is None else np.minimum(d, a)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tile", type=int, required=True, help="output tile size of A (borders at multiples)")
    ap.add_argument("--key", default="volumes/pred_numinst")
    ap.add_argument("--slab", type=int, default=64, help="z slab size (memory)")
    args = ap.parse_args()

    za, zb = zarr.open(args.a, "r"), zarr.open(args.b, "r")
    shape = za[args.key].shape[1:]
    assert tuple(shape) == tuple(zb[args.key].shape[1:]), (shape, zb[args.key].shape)
    bins = [0, 2, 4, 8, 16, 32, 10 ** 9]
    acc = {i: [0, 0.0, 0, 0] for i in range(len(bins) - 1)}  # voxels, sum|d|, fg-call flips, fg voxels (A or B)
    for z0 in range(0, shape[0], args.slab):
        z1 = min(shape[0], z0 + args.slab)
        sl = (slice(z0, z1), slice(None), slice(None))
        pa, pb = fg_prob(za, args.key, sl), fg_prob(zb, args.key, sl)
        dist = np.broadcast_to(border_dist(shape, args.tile)[z0:z1], pa.shape)
        diff = np.abs(pa - pb)
        flip = (pa > 0.5) != (pb > 0.5)
        fg = (pa > 0.5) | (pb > 0.5)
        for i in range(len(bins) - 1):
            m = (dist >= bins[i]) & (dist < bins[i + 1])
            acc[i][0] += int(m.sum())
            acc[i][1] += float(diff[m].sum())
            acc[i][2] += int(flip[m].sum())
            acc[i][3] += int(fg[m].sum())
    print(f"A={args.a}\nB={args.b}\nkey={args.key} shape={shape} A tile={args.tile}")
    print("| dist to A tile border | voxels | mean abs dP(fg) | fg-call flips | flips / fg voxels |")
    print("|---|---|---|---|---|")
    for i in range(len(bins) - 1):
        n, s, f, g = acc[i]
        hi = "inf" if bins[i + 1] > 10 ** 8 else bins[i + 1]
        print(f"| [{bins[i]}, {hi}) | {n} | {s / max(n, 1):.5f} | {f} | {f / max(g, 1):.4f} |")


if __name__ == "__main__":
    main()
