"""Helpers for viewing applied BiaPy augmentations (TIFF grids)."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from tqdm import tqdm

FISBE_DIR = Path("fisbe")


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


def plot_augmentation_samples(
        sample: str, split: str,
        augmentation_name: str,
        raw: bool, plotting_func: Callable,
        seed=42,
        sample_selection=4,
        subplots=[2, 2],
        fig_size=(8, 8)
):
    SAMPLE = sample + "*"
    SPLIT = split
    AUG_NAME = augmentation_name
    if raw:
        AUG_DATA_DIR = FISBE_DIR / AUG_NAME / SPLIT / 'raw'
    else:
        AUG_DATA_DIR = FISBE_DIR / AUG_NAME / SPLIT / 'label'
    print(AUG_DATA_DIR)
    print(SAMPLE)

    sample_augmentations = sorted(AUG_DATA_DIR.glob(f"{SAMPLE}.tif"))
    total_augmentaions = len(sample_augmentations)
    print(f"Total augmentations for sample found: {total_augmentaions}")
    rng = np.random.default_rng(seed)
    print(f"Selecting ({sample_selection}) samples")
    sample_selection = rng.integers(0, total_augmentaions, sample_selection)
    sample_augmentations = [sample_augmentations[i] for i in sample_selection]

    # Define thread job
    def load_mip(path: Path):
        img = tifffile.imread(path)
        print(f"Image Shape: {img.shape}")
        img = np.moveaxis(img, 0, 1)  # BiaPy ZCYX -> CZYX
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
