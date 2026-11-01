"""Add PatchPerPix training volumes to FISBe zarr files in place."""
import argparse
import os
import sys

import numpy as np

# Line-buffer stdout so nohup/log redirects show progress immediately.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
import zarr
from skimage.morphology import ball, opening

# Match FISBe volumes/raw spatial chunking; avoid full-volume chunks (slow random crops).
SPATIAL_CHUNKS = (64, 64, 64)


def prepare_sample(zarr_path, clipmax=1500, opening_radius=1, overwrite=False):
    root = zarr.open(zarr_path, mode="a")
    volumes = root["volumes"]

    if not overwrite and all(
            key in volumes.array_keys()
            for key in (
                "raw_normalized",
                "gt_instances_rm_5",
                "gt_numinst",
                "gt_fg_rm_5",
            )):
        print(f"skip (already prepared): {zarr_path}", flush=True)
        return

    raw = np.array(volumes["raw"])
    gt_instances = np.array(volumes["gt_instances"])

    raw_normalized = np.clip(raw, 0, clipmax).astype(np.float32) / float(clipmax)
    gt_numinst = np.sum(gt_instances > 0, axis=0).astype(np.uint8)
    gt_fg = (gt_numinst > 0).astype(np.float32)

    # ball(5) erodes thin FISBe neurons to near-zero fg (breaks RandomLocation sampling).
    if opening_radius > 0:
        struct = ball(opening_radius)
        gt_fg_rm_5 = opening(gt_fg, struct).astype(np.float32)
    else:
        gt_fg_rm_5 = gt_fg.astype(np.float32)

    # Keep instance labels; opening is applied only to the foreground mask.
    gt_instances_rm_5 = gt_instances.copy()

    if "raw_normalized" in volumes.array_keys():
        del volumes["raw_normalized"]
    if "gt_instances_rm_5" in volumes.array_keys():
        del volumes["gt_instances_rm_5"]
    if "gt_numinst" in volumes.array_keys():
        del volumes["gt_numinst"]
    if "gt_fg_rm_5" in volumes.array_keys():
        del volumes["gt_fg_rm_5"]

    # Skip chunks that are entirely fill_value. zarr 2 otherwise writes a file
    # per chunk, including empty label space (about N*847 files per 1k brain).
    volumes.zeros(
        "raw_normalized", shape=raw_normalized.shape, dtype="f4",
        chunks=(1,) + SPATIAL_CHUNKS, write_empty_chunks=False)
    volumes["raw_normalized"][...] = raw_normalized

    volumes.zeros(
        "gt_instances_rm_5", shape=gt_instances_rm_5.shape,
        dtype=gt_instances.dtype, chunks=(1,) + SPATIAL_CHUNKS,
        write_empty_chunks=False)
    volumes["gt_instances_rm_5"][...] = gt_instances_rm_5

    volumes.zeros(
        "gt_numinst", shape=gt_numinst.shape, dtype="u1",
        chunks=SPATIAL_CHUNKS, write_empty_chunks=False)
    volumes["gt_numinst"][...] = gt_numinst

    volumes.zeros(
        "gt_fg_rm_5", shape=gt_fg_rm_5.shape, dtype="f4",
        chunks=SPATIAL_CHUNKS, write_empty_chunks=False)
    volumes["gt_fg_rm_5"][...] = gt_fg_rm_5
    print(f"prepared: {zarr_path}", flush=True)


def prepare_folder(folder, clipmax=1500, opening_radius=5, overwrite=False):
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".zarr"):
            continue
        prepare_sample(
            os.path.join(folder, fn),
            clipmax=clipmax,
            opening_radius=opening_radius,
            overwrite=overwrite,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fisbe-root",
        default="/scratch/wmz2007/neuroinfo_fruitfly/fisbe",
    )
    parser.add_argument("--clipmax", type=int, default=1500)
    parser.add_argument("--opening-radius", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--datasets",
        default="completely,partly",
        help=(
            "Comma-separated folder names under fisbe-root. Each existing "
            "train/val/test subdir is prepared. Include partly for "
            "multi-dataset train (Phase 2)."
        ),
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for dataset in datasets:
        for split in ("train", "val", "test"):
            folder = os.path.join(args.fisbe_root, dataset, split)
            if os.path.isdir(folder):
                print(f"processing {folder}", flush=True)
                prepare_folder(
                    folder,
                    clipmax=args.clipmax,
                    opening_radius=args.opening_radius,
                    overwrite=args.overwrite,
                )
            else:
                print(f"skip (missing): {folder}", flush=True)


if __name__ == "__main__":
    main()
