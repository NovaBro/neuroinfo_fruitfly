#!/usr/bin/env python3
"""Apply the evaluate-time small-instance filter to PPP instance labels without evaluating.

``[evaluation] remove_small_components = N`` (eval_rm*.toml) only acts inside evaluate:
``evalinstseg.util.filter_components`` zeroes every predicted label with <= N voxels
before matching. Inference-only runs (no GT, e.g. himanshu) never evaluate, so their
``vote_instances`` are unfiltered. This script applies the same rule to every
``<inst-root>/**/<dataset>/*.hdf`` and writes the result as a NEW dataset beside the raw
labels (``vote_instances`` is left untouched):

    <key>_rm<N>   uint16, gzip, labels relabelled 1..K (background 0)
                  attrs: rm, source_key, n_before, n_after, voxels_before, voxels_after

Datasets that already exist are skipped (delete them by hand to redo). A summary CSV
(per volume: instance counts and labelled voxels before/after) goes to
``<inst-root>/<dataset>_rm<N>_summary.csv`` unless ``--summary`` is given.

Example (repo root; numpy + h5py only, so it also runs inside the ppp container)::

    python sbatch/ppp/tools/filter_small_instances.py \\
        --inst-root metrics/ppp/himanshu-e1.3-rm1500/test/instanced/23000 \\
        --dataset himanshu_r044 --rm 1500
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

SUMMARY_FIELDS = ["hdf", "stem", "n_before", "n_after", "removed",
                  "voxels_before", "voxels_after", "status"]


def filter_labels(labels: np.ndarray, rm: int) -> tuple[np.ndarray, int, int]:
    """Zero labels with <= rm voxels (filter_components rule), relabel the rest 1..K.

    Returns (filtered uint16 array, n_before, n_after).
    """
    if labels.dtype.kind not in "ui":
        raise TypeError(f"expected integer labels, got {labels.dtype}")
    if labels.size and labels.min() < 0:
        raise ValueError("negative label ids")
    flat = labels.ravel()
    counts = np.bincount(flat)
    ids = np.nonzero(counts)[0]
    ids = ids[ids != 0]
    keep = ids[counts[ids] > rm]
    if len(keep) > np.iinfo(np.uint16).max:
        raise ValueError(f"{len(keep)} instances do not fit uint16")
    lut = np.zeros(len(counts), dtype=np.uint16)
    lut[keep] = np.arange(1, len(keep) + 1, dtype=np.uint16)
    return lut[flat].reshape(labels.shape), int(len(ids)), int(len(keep))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--inst-root", type=Path, required=True,
                   help="instanced/<ckpt> dir of an experiment (searched recursively)")
    p.add_argument("--dataset", required=True, help="dataset folder name, e.g. himanshu_r044")
    p.add_argument("--rm", type=int, default=1500, help="drop labels with <= rm voxels")
    p.add_argument("--key", default="vote_instances", help="source dataset in each .hdf")
    p.add_argument("--summary", type=Path, default=None)
    args = p.parse_args()

    out_key = f"{args.key}_rm{args.rm}"
    hdfs = sorted(h for h in args.inst_root.rglob("*.hdf") if h.parent.name == args.dataset)
    if not hdfs:
        sys.exit(f"no {args.dataset}/*.hdf under {args.inst_root}")
    summary = args.summary or args.inst_root / f"{args.dataset}_rm{args.rm}_summary.csv"
    print(f"{len(hdfs)} files, {args.key} -> {out_key} (drop <= {args.rm} voxels)", flush=True)

    rows = []
    for i, path in enumerate(hdfs, 1):
        row = {k: "" for k in SUMMARY_FIELDS}
        row.update(hdf=str(path), stem=path.stem)
        with h5py.File(path, "a") as h:
            if args.key not in h:
                row["status"] = f"missing:{args.key}"
            elif out_key in h:
                ds = h[out_key]
                row.update({k: ds.attrs.get(k, "") for k in
                            ("n_before", "n_after", "voxels_before", "voxels_after")})
                row["status"] = "skip_exists"
            else:
                labels = h[args.key][...]
                out, n_before, n_after = filter_labels(labels, args.rm)
                stats = dict(n_before=n_before, n_after=n_after,
                             voxels_before=int(np.count_nonzero(labels)),
                             voxels_after=int(np.count_nonzero(out)))
                ds = h.create_dataset(out_key, data=out, dtype=np.uint16, compression="gzip")
                ds.attrs.update(rm=args.rm, source_key=args.key, **stats)
                row.update(stats, status="ok")
        if row["n_before"] != "" and row["n_after"] != "":
            row["removed"] = int(row["n_before"]) - int(row["n_after"])
        print(f"[{i}/{len(hdfs)}] {path.stem}: {row['status']} "
              f"{row['n_before']} -> {row['n_after']}", flush=True)
        rows.append(row)

    with summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["status"].startswith("missing")]
    print(f"summary: {summary} ({len(rows) - len(bad)} ok/skip, {len(bad)} missing)", flush=True)
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
