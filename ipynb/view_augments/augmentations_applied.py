"""Helpers for viewing applied BiaPy augmentations (TIFF/Zarr grids)."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import zarr
from tqdm import tqdm

FISBE_DIR = Path("fisbe")
_TIFF_SUFFIXES = (".tif", ".tiff")


def fisbe_gt_instance_mip(labels, z_axis=1):
    arr = np.asarray(labels)        # (Z, Y, X)
    mip = arr.max(axis=z_axis)      # (Y, X)

    rgb = np.zeros((*mip.shape, 3), dtype=np.float32)
    for lab in np.unique(mip):
        if lab == 0:
            continue
        color = np.random.randint(72, 255, 3).astype(np.float32)
        rgb[mip == lab] = color
    rgb /= 255.0

    return rgb


def _load_volume(path: Path):
    """Load a TIFF file or Zarr volume into an ndarray."""
    suffix = path.suffix.lower()
    if suffix in _TIFF_SUFFIXES and path.is_file():
        return tifffile.imread(path)
    if suffix == ".zarr" and path.is_dir():
        obj = zarr.open(path.as_posix(), mode="r")
        # Zarr can be either an Array or a Group; use first array for Group.
        if hasattr(obj, "shape"):
            return np.asarray(obj)
        if hasattr(obj, "arrays"):
            arrays = list(obj.arrays())
            if not arrays:
                raise ValueError(f"No arrays found in zarr group: {path}")
            return np.asarray(arrays[0][1])
    raise ValueError(f"Unsupported augmentation volume: {path}")


def _to_czyx(img: np.ndarray, path: Path, *, raw: bool) -> np.ndarray:
    """Normalize loaded volume to CZYX for plotting helpers."""
    arr = np.asarray(img)
    if arr.ndim != 4:
        return arr

    suffix = path.suffix.lower()
    if suffix in _TIFF_SUFFIXES:
        # TIFF convention in this workflow is ZCYX.
        return np.moveaxis(arr, 0, 1)
    if suffix == ".zarr":
        if raw:
            # Raw Zarr convention here is ZYXC.
            return np.moveaxis(arr, -1, 0)
        # Label/instance Zarr is channel-less ZYX in common cases.
        return arr
    return arr


def plot_augmentation_samples(
        sample: str, split: str,
        augmentation_name: str,
        raw: bool, plotting_func: Callable,
        seed=42,
        sample_selection=4,
        subplots=[2, 2],
        fig_size=(8, 8)
):
    sample_stem = sample
    sample_pattern = f"{sample_stem}*"
    SPLIT = split
    AUG_NAME = augmentation_name
    if raw:
        AUG_DATA_DIR = FISBE_DIR / AUG_NAME / SPLIT / 'raw'
    else:
        AUG_DATA_DIR = FISBE_DIR / AUG_NAME / SPLIT / 'label'
    print(AUG_DATA_DIR)
    print(sample_pattern)

    candidates = sorted(AUG_DATA_DIR.glob(sample_pattern))
    sample_augmentations = [
        p for p in candidates
        if (p.is_file() and p.suffix.lower() in _TIFF_SUFFIXES)
        or (p.is_dir() and p.suffix.lower() == ".zarr")
    ]
    total_augmentaions = len(sample_augmentations)
    if total_augmentaions == 0:
        raise FileNotFoundError(
            f"No augmentation volumes found in {AUG_DATA_DIR} for pattern '{sample_pattern}' "
            "(expected .tif/.tiff files or .zarr directories)."
        )
    print(f"Total augmentations for sample found: {total_augmentaions}")
    rng = np.random.default_rng(seed)
    print(f"Selecting ({sample_selection}) samples")
    sample_selection = rng.integers(0, total_augmentaions, sample_selection)
    sample_augmentations = [sample_augmentations[i] for i in sample_selection]

    # Define thread job
    def load_mip(path: Path):
        img = _load_volume(path)
        print(f"Shape: {img.shape}", f"dtype: {img.dtype}", f"max: {img.max()}, min: {img.min()}", f"Image Shape: {img.shape}")
        img = _to_czyx(img, path, raw=raw)
        return path, plotting_func(img, 1)

    # Loading Samples
    with ThreadPoolExecutor(max_workers=2) as ex:
        mips = list(
            tqdm(
                ex.map(load_mip, sample_augmentations),
                total=len(sample_augmentations)
            )
        )

    print(f"Transformed image shape: {mips[0][1].shape}")

    # Plotting Samples
    for chunk_start in range(0, len(mips), 4):
        chunk = mips[chunk_start : chunk_start + 4]
        fig, axes = plt.subplots(nrows=subplots[0], ncols=subplots[0], figsize=fig_size, dpi=150, squeeze=False)
        for i, (path, mip) in enumerate(chunk):
            ax = axes[i // 2, i % 2]
            ax.imshow(mip)
            ax.set_title(path.stem.replace("R38F04-20181005_63_G3", ""), fontsize=8)
            ax.axis("off")
        plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.2)
        plt.show()
