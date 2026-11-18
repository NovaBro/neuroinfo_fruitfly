"""Add PatchPerPix training volumes to FISBe zarr files in place."""
import argparse
import os
import sys

import numpy as np

# Line-buffer stdout so nohup/log redirects show progress immediately.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
import zarr
from numcodecs import Blosc
from skimage.morphology import ball, opening

# Match FISBe volumes/raw spatial chunking; avoid full-volume chunks (slow random crops).
SPATIAL_CHUNKS = (64, 64, 64)
# zstd + bitshuffle, as FISBe raw and the simple_v1 generator use. zarr's default
# (lz4 + byte shuffle) is several times larger on sparse label / mask data.
COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def fg_rm_mask(gt_numinst, opening_radius=1):
    """gt_fg_rm_5: foreground (gt_numinst > 0) opened with ball(opening_radius),
    float32 0/1. Shared with synthetic/simple_tubes (generator writes it directly)."""
    gt_fg = (np.asarray(gt_numinst) > 0).astype(np.float32)
    # ball(5) erodes thin FISBe neurons to near-zero fg (breaks RandomLocation sampling).
    if opening_radius > 0:
        return opening(gt_fg, ball(opening_radius)).astype(np.float32)
    return gt_fg


def write_array(volumes, name, data, chunks, attrs=None):
    """Create volumes/<name> and write through the returned array.

    Writing via volumes[name][...] instead reopens the array, and zarr 2's
    Group.__getitem__ does not keep write_empty_chunks=False (it is not stored in
    .zarray), so every empty chunk was written (N*847 files per FISBe-size volume)."""
    arr = volumes.zeros(name, shape=data.shape, dtype=data.dtype, chunks=chunks,
                        compressor=COMPRESSOR, write_empty_chunks=False)
    arr[...] = data
    if attrs:
        arr.attrs.update(attrs)
    return arr


def background_medians(raw, step=4):
    """Per-channel median of a strided subsample (the render pedestal)."""
    return [float(np.median(raw[c, ::step, ::step, ::step]))
            for c in range(raw.shape[0])]


def prepare_sample(zarr_path, clipmax=1500, opening_radius=1, overwrite=False,
                   subtract_median=False, raw_dtype="float32"):
    """subtract_median: subtract each channel's median before the usual
    clip(0, clipmax) / clipmax. For renders on a large constant pedestal
    (SWC2IMG: background ~2,500, signal ~+330), which plain clipping at 1500
    saturates to 1.0. Off by default (FISBe / synth_MCFO unchanged).
    raw_dtype: storage dtype of raw_normalized. "float16" halves the size of noisy
    renders (SWC2IMG ~1.6 -> ~0.8 GB/volume); training with raw_transport_fp16
    already feeds the GPU float16 raw."""
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

    medians = None
    if subtract_median:
        medians = background_medians(raw)
        raw = raw.astype(np.float32) - np.asarray(
            medians, dtype=np.float32).reshape((-1,) + (1,) * (raw.ndim - 1))
    raw_normalized = (np.clip(raw, 0, clipmax).astype(np.float32) / float(clipmax)).astype(raw_dtype)
    del raw
    gt_numinst = np.sum(gt_instances > 0, axis=0).astype(np.uint8)
    gt_fg_rm_5 = fg_rm_mask(gt_numinst, opening_radius)

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

    # Skip chunks that are entirely fill_value (see write_array).
    write_array(volumes, "raw_normalized", raw_normalized, (1,) + SPATIAL_CHUNKS,
                attrs={"clipmax": clipmax, "subtracted_medians": medians})
    write_array(volumes, "gt_instances_rm_5", gt_instances_rm_5, (1,) + SPATIAL_CHUNKS)
    write_array(volumes, "gt_numinst", gt_numinst, SPATIAL_CHUNKS)
    write_array(volumes, "gt_fg_rm_5", gt_fg_rm_5, SPATIAL_CHUNKS)
    print(f"prepared: {zarr_path}", flush=True)


def prepare_folder(folder, clipmax=1500, opening_radius=5, overwrite=False,
                   subtract_median=False, raw_dtype="float32"):
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".zarr"):
            continue
        prepare_sample(
            os.path.join(folder, fn),
            clipmax=clipmax,
            opening_radius=opening_radius,
            overwrite=overwrite,
            subtract_median=subtract_median,
            raw_dtype=raw_dtype,
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
        "--subtract-median", action="store_true",
        help=("Subtract each channel's median before clip/normalize "
              "(SWC2IMG renders sit on a ~2,500 pedestal)"))
    parser.add_argument(
        "--raw-dtype", default="float32", choices=["float32", "float16"],
        help="storage dtype of volumes/raw_normalized (float16: ~half the size)")
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
                    subtract_median=args.subtract_median,
                    raw_dtype=args.raw_dtype,
                )
            else:
                print(f"skip (missing): {folder}", flush=True)


if __name__ == "__main__":
    main()
