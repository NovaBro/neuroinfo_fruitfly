#!/usr/bin/env python3
"""Compare two vote_instances patch-graph caches (<instanced>/<sample>.zarr,
group volumes/blocks) and optionally two instance-segmentation HDFs.

Pairs are compared as undirected row sets, affinities after aligning rows by
pair, so row order differences between implementations are ignored.

  python3 sbatch/ppp/tools/compare_vi_block_cache.py OLD.zarr NEW.zarr \
      [--hdf OLD.hdf NEW.hdf] [--key vote_instances] [--atol 1e-5]
"""
import argparse
import sys

import h5py
import numpy as np
import zarr


def cache_groups(root):
    """{relative group path: (patch_pairs, aff_graph_mat)} for every complete
    block / inter-block group under volumes/blocks."""
    out = {}
    blocks = root['volumes/blocks']

    def visit(path, obj):
        if isinstance(obj, zarr.hierarchy.Group) and \
                'patch_pairs' in obj and 'aff_graph_mat' in obj:
            out[path] = (np.asarray(obj['patch_pairs']),
                         np.asarray(obj['aff_graph_mat']))

    blocks.visititems(visit)
    return out


def pair_map(pairs, affs):
    rows = {}
    for p, v in zip(pairs.tolist(), affs.tolist()):
        key = tuple(min(p[:3], p[3:]) + max(p[:3], p[3:]))
        rows[key] = v
    return rows


def compare_caches(old_path, new_path, atol):
    old = cache_groups(zarr.open(old_path, mode='r'))
    new = cache_groups(zarr.open(new_path, mode='r'))
    common = sorted(set(old) & set(new))
    print("groups: old %d, new %d, common %d" % (len(old), len(new),
                                                 len(common)))
    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))
    if only_old:
        print("only in old (%d): %s" % (len(only_old), only_old[:10]))
    if only_new:
        print("only in new (%d): %s" % (len(only_new), only_new[:10]))

    n_bad = 0
    max_diff = 0.0
    for g in common:
        mo = pair_map(*old[g])
        mn = pair_map(*new[g])
        if mo.keys() != mn.keys():
            n_bad += 1
            print("pair set mismatch %s: old %d new %d (sym diff %d)" % (
                g, len(mo), len(mn), len(mo.keys() ^ mn.keys())))
            continue
        if mo:
            d = np.abs(np.array([mo[k] for k in mo]) -
                       np.array([mn[k] for k in mo]))
            max_diff = max(max_diff, float(d.max()))
            if d.max() > atol:
                n_bad += 1
                print("aff mismatch %s: max abs diff %g" % (g, d.max()))
    print("groups with mismatches: %d / %d, max aff diff %g" % (
        n_bad, len(common), max_diff))
    return n_bad == 0 and not only_old and not only_new


def compare_hdf(old_path, new_path, key):
    with h5py.File(old_path, 'r') as f:
        old = np.asarray(f[key])
    with h5py.File(new_path, 'r') as f:
        new = np.asarray(f[key])
    if old.shape != new.shape:
        print("hdf shape mismatch %s vs %s" % (old.shape, new.shape))
        return False
    fg = (old > 0) | (new > 0)
    o = old[fg].astype(np.int64)
    n = new[fg].astype(np.int64)
    pairs = np.unique(np.stack([o, n], axis=1), axis=0)
    # same partition up to relabeling iff the label pairs form a bijection
    bijective = (len(np.unique(pairs[:, 0])) == len(pairs) and
                 len(np.unique(pairs[:, 1])) == len(pairs))
    print("hdf %s: fg voxels %d, old labels %d, new labels %d, "
          "label pairs %d, identical partition: %s" % (
              key, fg.sum(), len(np.unique(o)), len(np.unique(n)),
              len(pairs), bijective))
    if not bijective:
        print("voxels with differing fg: %d" % np.sum((old > 0) != (new > 0)))
    return bijective


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('old_zarr')
    ap.add_argument('new_zarr')
    ap.add_argument('--hdf', nargs=2, metavar=('OLD', 'NEW'))
    ap.add_argument('--key', default='vote_instances')
    ap.add_argument('--atol', type=float, default=1e-5)
    args = ap.parse_args()

    ok = compare_caches(args.old_zarr, args.new_zarr, args.atol)
    if args.hdf:
        ok = compare_hdf(args.hdf[0], args.hdf[1], args.key) and ok
    print("OK" if ok else "DIFFERENCES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
