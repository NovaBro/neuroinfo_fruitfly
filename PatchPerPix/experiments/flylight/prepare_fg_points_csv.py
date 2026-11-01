"""Write subsampled FG point CSVs from gt_fg_rm_5 for sample_fg_points training.

Each line is space-separated z y x voxel coordinates (no header). Stride keeps
points on a regular grid among FG voxels so RandomLocation(ensure_nonempty=...)
can sample without building per-volume mask integral arrays at train startup.

Output layout: <out-dir>/<dataset>/<sample>.csv, where <dataset> is the first
component of each --splits entry (e.g. synth_vh_u8/train -> synth_vh_u8/). Datasets
can reuse sample names (brain_00000), so a flat folder would collide.
"""
import argparse
import os
import sys

import numpy as np
import zarr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

DEFAULT_SPLITS = "completely/train,completely/val,completely/test"


def write_sample_csv(zarr_path, out_csv, stride=4, overwrite=False):
    if os.path.exists(out_csv) and not overwrite:
        print(f"skip (exists): {out_csv}", flush=True)
        return

    root = zarr.open(zarr_path, mode="r")
    if "volumes/gt_fg_rm_5" not in root:
        raise FileNotFoundError(
            f"{zarr_path} missing volumes/gt_fg_rm_5; run prepare_fisbe_for_ppp.py first"
        )
    fg = np.asarray(root["volumes/gt_fg_rm_5"])
    coords = np.argwhere(fg > 0)
    if coords.size == 0:
        raise RuntimeError(f"no FG voxels in {zarr_path}")

    keep = np.all(coords % stride == 0, axis=1)
    coords = coords[keep]
    if coords.size == 0:
        # Fall back to every Nth FG voxel if stride grid is empty.
        coords = np.argwhere(fg > 0)[:: max(1, stride ** 3)]

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    np.savetxt(out_csv, coords, fmt="%d", delimiter=" ")
    print(f"wrote {out_csv}  points={len(coords)}  stride={stride}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fisbe-root",
        default="/scratch/wmz2007/neuroinfo_fruitfly/fisbe",
    )
    parser.add_argument(
        "--out-dir",
        default="/scratch/wmz2007/neuroinfo_fruitfly/fisbe/fg_points",
    )
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    for split in args.splits.split(","):
        split = split.strip()
        if not split:
            continue
        folder = os.path.join(args.fisbe_root, split)
        if not os.path.isdir(folder):
            print(f"skip missing folder: {folder}", flush=True)
            continue
        print(f"processing {folder}", flush=True)
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".zarr"):
                continue
            zarr_path = os.path.join(folder, fn)
            dataset = split.split("/")[0]
            out_csv = os.path.join(
                args.out_dir, dataset, fn.replace(".zarr", ".csv"))
            write_sample_csv(
                zarr_path, out_csv, stride=args.stride, overwrite=args.overwrite
            )


if __name__ == "__main__":
    main()
