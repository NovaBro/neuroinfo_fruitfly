"""Verify the 'Sk' (tubed skeleton) instance-segmentation target channel.

Calls ``labels_into_channels`` directly on a FISBe instance-label Zarr, bypassing the
config/``check_configuration`` machinery (``Sk`` is not whitelisted there yet). Dumps every
channel to TIFF via the function's own ``save_dir`` debug path and runs the numeric
acceptance checks for the channel.

Example
-------
    python biapy_work_folder/check_skeleton_channel.py --crop 128 256 256
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import zarr

# The 'Sk' channel lives in the local BiaPy checkout; prefer it over any installed copy.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "BiaPy-novabro"))

from biapy.data.pre_processing.labels_to_channels import labels_into_channels  # noqa: E402

DEFAULT_LABEL_DIR = REPO_ROOT / "fisbe" / "biapy-channel-scale-zarr" / "train" / "label"


def load_labels(label_dir: Path, filename: str | None, crop: list[int] | None) -> tuple[np.ndarray, str]:
    """Load one instance-label volume as ``(z, y, x, 1)``, optionally centre-cropped.

    The FISBe label Zarrs are plain arrays already in the ``(z, y, x, 1)`` layout that
    ``labels_into_channels`` expects, so no axis juggling is needed.
    """
    if not label_dir.is_dir():
        raise SystemExit(f"Label dir not found: {label_dir}")

    if filename is None:
        candidates = sorted(p.name for p in label_dir.iterdir() if p.name.endswith(".zarr"))
        if not candidates:
            raise SystemExit(f"No .zarr volumes in {label_dir}")
        filename = candidates[0]

    arr = zarr.open(str(label_dir / filename))
    vol = np.asarray(arr)

    if vol.ndim == 3:  # tolerate a missing trailing channel axis
        vol = vol[..., None]

    if crop is not None:
        # Centre crop: the densest neurite structure sits away from the volume edges.
        starts = [max(0, (vol.shape[d] - crop[d]) // 2) for d in range(3)]
        vol = vol[
            starts[0] : starts[0] + crop[0],
            starts[1] : starts[1] + crop[1],
            starts[2] : starts[2] + crop[2],
        ]

    return np.ascontiguousarray(vol), filename


def count_touching_pairs(lab: np.ndarray) -> int:
    """Count distinct instance pairs that are adjacent along any axis."""
    pairs = set()
    for axis in range(lab.ndim):
        a = np.take(lab, np.arange(lab.shape[axis] - 1), axis=axis)
        b = np.take(lab, np.arange(1, lab.shape[axis]), axis=axis)
        sel = (a != b) & (a > 0) & (b > 0)
        for u, v in zip(a[sel].ravel(), b[sel].ravel()):
            pairs.add((int(min(u, v)), int(max(u, v))))
    return len(pairs)


def describe(name: str, ch: np.ndarray) -> None:
    """Print dtype / unique-value / occupancy stats for one channel."""
    uniq = np.unique(ch)
    shown = uniq[:5]
    print(
        f"  {name:<4} dtype={str(ch.dtype):<8} "
        f"n_unique={uniq.size:<6} "
        f"nonzero={float((ch > 0).mean()) * 100:6.3f}%  "
        f"range=[{ch.min():g}, {ch.max():g}]  "
        f"first_vals={np.array2string(shown, precision=3)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR, help="Dir of instance-label .zarr volumes")
    parser.add_argument("--file", default=None, help="Specific .zarr name (default: first alphabetically)")
    parser.add_argument(
        "--crop",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=None,
        help="Centre-crop size. Full volumes are ~200M voxels, so per-instance 3D skeletonization is slow.",
    )
    parser.add_argument("--dilation", type=int, nargs="+", default=[2], help="Tube dilation: one value, or one per axis")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "biapy_work_folder" / "skeleton_channel_check")
    args = parser.parse_args()

    dilation = args.dilation[0] if len(args.dilation) == 1 else list(args.dilation)

    vol, filename = load_labels(args.label_dir, args.file, args.crop)
    n_instances = int(np.unique(vol).size - (1 if (vol == 0).any() else 0))
    print(f"volume    : {filename}")
    print(f"shape     : {vol.shape}  dtype={vol.dtype}")
    print(f"instances : {n_instances}")
    print(f"dilation  : {dilation}")
    if n_instances <= 1:
        print("\nWARNING: <=1 instance in this crop; labels_into_channels returns an all-zero mask early.")

    extra_opts = {"Db": {"mask_values": False}, "Sk": {"dilation": dilation}}

    # Baseline without 'Sk' so the delta isolates the skeleton block's cost, which becomes a
    # per-batch dataloader cost once the channel is regenerated after augmentation.
    t0 = time.perf_counter()
    labels_into_channels(vol, mode=["F", "Db"], channel_extra_opts=extra_opts)
    t_base = time.perf_counter() - t0

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    mask = labels_into_channels(
        vol,
        mode=["F", "Db", "Sk"],
        channel_extra_opts=extra_opts,
        save_dir=str(args.out),
    )
    t_full = time.perf_counter() - t0

    print(f"\ntiming    : F+Db {t_base:.2f}s | F+Db+Sk {t_full:.2f}s | Sk block ~{t_full - t_base:.2f}s")
    print(f"output    : {mask.shape} dtype={mask.dtype}")
    print(f"tiffs     : {args.out}")

    print("\nper-channel stats:")
    for idx, name in enumerate(["F", "Db", "Sk"]):
        describe(name, mask[..., idx])

    f_ch, sk_ch = mask[..., 0], mask[..., 2]
    sk_bin, f_bin = sk_ch > 0, f_ch > 0

    print("\nacceptance checks:")
    failures = []

    n_uniq = np.unique(sk_ch).size
    ok = n_uniq <= 2
    failures += [] if ok else ["Sk is not binary"]
    print(f"  [{'PASS' if ok else 'FAIL'}] Sk is binary (n_unique={n_uniq}, needs <=2 to stay 'bin' in norm.py)")

    outside = int((sk_bin & ~f_bin).sum())
    ok = outside == 0
    failures += [] if ok else [f"{outside} Sk voxels outside foreground"]
    print(f"  [{'PASS' if ok else 'FAIL'}] Sk contained in F ({outside} voxels outside foreground)")

    ok = bool(sk_bin.any())
    failures += [] if ok else ["Sk is empty"]
    print(f"  [{'PASS' if ok else 'FAIL'}] Sk is non-empty ({int(sk_bin.sum())} voxels)")

    # Per-instance coverage is the point of skeletonizing each label separately rather than the
    # binarized union: every instance should contribute its own centreline.
    labels = [lb for lb in np.unique(vol) if lb != 0]
    missing = [int(lb) for lb in labels if not (sk_bin & (vol[..., 0] == lb)).any()]
    ok = not missing
    failures += [] if ok else [f"{len(missing)} instances without a skeleton"]
    print(
        f"  [{'PASS' if ok else 'FAIL'}] every instance has a centreline "
        f"({len(labels) - len(missing)}/{len(labels)}"
        + (f", missing={missing[:10]}" if missing else "")
        + ")"
    )

    # The per-instance check above only proves the point where instances actually touch: a naive
    # skeletonization of the binarized union would merge those into one centreline.
    n_touching = count_touching_pairs(vol[..., 0])
    print(f"  [INFO] touching instance pairs in this crop: {n_touching}")
    if n_touching == 0:
        print("         (no contact here, so the per-instance choice is untested; try a larger --crop)")

    if failures:
        print("\nFAILED: " + "; ".join(failures))
        sys.exit(1)
    print("\nAll checks passed. Inspect vol_Sk_tubed_skeleton.tif against vol_F_foreground.tif.")


if __name__ == "__main__":
    main()
