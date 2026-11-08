#!/usr/bin/env python3
"""Compare a dense vs sparse ([prediction] sparse_affs) test run of the same checkpoint.

For one sample, checks:
  1. label output (vote_instances / _masked / vote_foreground HDF datasets) identical;
  2. evaluation metrics (per-sample toml) identical;
  3. pred_numinst identical; pred_affs equal at every kept voxel and zero elsewhere;
     every voxel the label stage treats as foreground (numinst_threshs) is kept;
  4. pred_affs size on disk.
Optionally also against a third (older production) dense run with --ref-root.

  python compare_sparse_affs_ab.py --dense-root metrics/ppp/abtest-dense \
      --sparse-root metrics/ppp/abtest-sparse --sample 1k_v1.1/brain_00005 --ckpt 10000 \
      [--ref-root /scratch/.../metrics/ppp/opexp2-f15-lr4e-4]
"""
import argparse
import os
import subprocess
import sys

import h5py
import numpy as np
import toml
import zarr

SUB = "patch_threshold_0_3/fc_threshold_0_3/mws_True/skeletonize_foreground_True/numinst_threshs__0_9_0_1_"
NUMINST_THRESHS = [0.9, 0.1]


def paths(root, split, ckpt, sample):
    return {
        "processed": os.path.join(root, split, "processed", str(ckpt), sample + ".zarr"),
        "instanced": os.path.join(root, split, "instanced", str(ckpt), SUB, sample + ".hdf"),
        "evaluated": os.path.join(root, split, "evaluated", str(ckpt), SUB,
                                  sample + "_vote_instances_cldice_greedy_rm200.toml"),
    }


def du(path):
    if not os.path.exists(path):
        return "missing"
    return subprocess.run(["du", "-sh", path], capture_output=True, text=True).stdout.split()[0]


def compare_labels(a, b, tag):
    ok = True
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        for k in ("vote_instances", "vote_instances_masked", "vote_foreground"):
            if k not in fa or k not in fb:
                print(f"  [{tag}] {k}: missing in one file")
                ok = False
                continue
            x, y = fa[k][...], fb[k][...]
            same = x.shape == y.shape and np.array_equal(x, y)
            n = len(np.unique(x)) - 1
            print(f"  [{tag}] {k}: {'IDENTICAL' if same else 'DIFFERENT'} "
                  f"(labels {n} vs {len(np.unique(y)) - 1}, voxels differ {int((x != y).sum()) if x.shape == y.shape else 'shape'})")
            ok &= same
    return ok


def compare_eval(a, b, tag):
    ta, tb = toml.load(a), toml.load(b)

    def flat(d, p=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, p + k + "."))
            else:
                out[p + k] = v
        return out
    fa, fb = flat(ta), flat(tb)
    diff = [k for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
    key = ["general.Num GT", "general.Num Pred", "confusion_matrix.avFscore", "general.avg_f1_cov_score"]
    print(f"  [{tag}] metrics: {'IDENTICAL' if not diff else f'{len(diff)} keys differ, e.g. {diff[:5]}'}")
    print("  [" + tag + "] " + ", ".join(f"{k.split('.')[-1]}={fa.get(k)}/{fb.get(k)}" for k in key))
    return not diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-root", required=True)
    ap.add_argument("--sparse-root", required=True)
    ap.add_argument("--ref-root")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--ckpt", default="10000")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    D = paths(a.dense_root, a.split, a.ckpt, a.sample)
    S = paths(a.sparse_root, a.split, a.ckpt, a.sample)
    ok = True

    print("1. label output")
    ok &= compare_labels(D["instanced"], S["instanced"], "dense vs sparse")
    if a.ref_root:
        R = paths(a.ref_root, a.split, a.ckpt, a.sample)
        compare_labels(R["instanced"], D["instanced"], "prod-dense vs new-dense (run-to-run)")

    print("2. evaluation")
    ok &= compare_eval(D["evaluated"], S["evaluated"], "dense vs sparse")
    if a.ref_root:
        compare_eval(R["evaluated"], D["evaluated"], "prod-dense vs new-dense (run-to-run)")

    print("3. predictions")
    zd, zs = zarr.open(D["processed"], "r"), zarr.open(S["processed"], "r")
    nd, ns = zd["volumes/pred_numinst"][...], zs["volumes/pred_numinst"][...]
    same_ni = np.array_equal(nd, ns)
    print(f"  pred_numinst: {'IDENTICAL' if same_ni else 'DIFFERENT'}")
    ok &= same_ni
    p0 = ns[0].astype(np.float32)
    thr = zs["volumes/pred_affs"].attrs.get("sparse_fg_thresh")
    kept = (1.0 - p0) > thr
    numinst = np.zeros(ns.shape[1:], np.uint8)
    for i, th in enumerate(NUMINST_THRESHS):
        numinst[ns[i + 1] > th] = i + 1
    label_fg = numinst > 0
    print(f"  sparse_fg_thresh={thr}: kept voxels {kept.mean() * 100:.3f}%, "
          f"label foreground {label_fg.mean() * 100:.3f}%, label fg not kept: {int((label_fg & ~kept).sum())}")
    ok &= not np.any(label_fg & ~kept)
    # pred_affs: compare in chunk-row tiles (343 x volume is ~128 GB uncompressed)
    ad, as_ = zd["volumes/pred_affs"], zs["volumes/pred_affs"]
    zstep, ystep = ad.chunks[1], ad.chunks[2]
    n_bad_kept = n_nonzero_dropped = 0
    for z0 in range(0, ad.shape[1], zstep):
        for y0 in range(0, ad.shape[2], ystep):
            sl = (slice(None), slice(z0, z0 + zstep), slice(y0, y0 + ystep))
            xd, xs = ad[sl], as_[sl]
            k = kept[sl[1:]]
            n_bad_kept += int((xd[:, k] != xs[:, k]).sum())
            n_nonzero_dropped += int((xs[:, ~k] != 0).sum())
    print(f"  pred_affs: kept voxels differing {n_bad_kept}, non-zero outside mask {n_nonzero_dropped}")
    ok &= n_bad_kept == 0 and n_nonzero_dropped == 0

    print("4. disk")
    print(f"  dense  pred_affs {du(os.path.join(D['processed'], 'volumes/pred_affs'))}, "
          f"sparse pred_affs {du(os.path.join(S['processed'], 'volumes/pred_affs'))}")

    print("SPARSE A/B: " + ("PASS (identical labels and metrics)" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
