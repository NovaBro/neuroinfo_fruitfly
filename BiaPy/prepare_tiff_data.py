"""Convert FISBe zarr volumes to BiaPy-compatible TIFF files."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import zarr
from biapy.data.data_manipulation import imwrite as biapy_imwrite
from tqdm import tqdm


def merge_instance_masks(stacked: np.ndarray) -> np.ndarray:
    """Merge per-instance binary masks (N, Z, Y, X) into one label volume (Z, Y, X)."""
    merged = np.zeros(stacked.shape[1:], dtype=np.uint16)
    for mask in stacked:
        instance_ids = mask > 0
        if not np.any(instance_ids):
            continue
        merged[instance_ids] = mask[instance_ids]
    return merged


def to_biapy_volume(array: np.ndarray) -> np.ndarray:
    """Wrap a ZYX or ZYXC array as a 6D TZCYXS volume for BiaPy."""
    if array.ndim == 3:
        array = array[..., np.newaxis]
    if array.ndim != 4:
        raise ValueError(f"Expected ZYX or ZYXC array, got shape {array.shape}")

    volume = array.transpose(0, 3, 1, 2)  # ZYXC -> ZCYX
    return volume[np.newaxis, :, :, :, :, np.newaxis]  # TZCYXS


def convert_split(zarr_split_dir: Path, tiff_split_dir: Path) -> None:
    raw_dir = tiff_split_dir / "raw"
    label_dir = tiff_split_dir / "label"
    raw_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for zarr_path in sorted(zarr_split_dir.glob("*.zarr")):
        raw = np.array(zarr.open(zarr_path, mode="r", path="volumes/raw"))
        seg = np.array(zarr.open(zarr_path, mode="r", path="volumes/gt_instances"))

        if raw.ndim != 4 or seg.ndim != 4:
            raise ValueError(
                f"{zarr_path.name}: expected 4D raw/seg arrays, got raw={raw.shape}, seg={seg.shape}"
            )

        raw_volume = to_biapy_volume(raw.transpose(1, 2, 3, 0))  # CZYX -> ZYXC -> TZCYXS
        merged_labels = merge_instance_masks(seg)
        label_volume = to_biapy_volume(merged_labels)

        stem = zarr_path.name
        biapy_imwrite(str(raw_dir / f"{stem}.tiff"), raw_volume)
        biapy_imwrite(str(label_dir / f"{stem}_seg.tiff"), label_volume)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("fisbe/completely"),
        help="Root directory containing train/test/val zarr splits",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fisbe/biapy"),
        help="Output directory for BiaPy TIFF splits",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Dataset splits to convert",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing split directories before conversion",
    )
    args = parser.parse_args()

    source_root = args.source.resolve()
    output_root = args.output.resolve()

    for split in args.splits:
        zarr_split_dir = source_root / split
        tiff_split_dir = output_root / split
        if not zarr_split_dir.is_dir():
            raise FileNotFoundError(f"Missing zarr split directory: {zarr_split_dir}")

        if args.clean and tiff_split_dir.exists():
            shutil.rmtree(tiff_split_dir)

        tqdm.write(f"Converting {split} ({len(list(zarr_split_dir.glob('*.zarr')))} volumes)")
        convert_split(zarr_split_dir, tiff_split_dir)


if __name__ == "__main__":
    main()
