"""Helpers for viewing applied BiaPy augmentations (TIFF/Zarr grids)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import zarr
from tqdm import tqdm

FISBE_DIR = Path("fisbe")
DEFAULT_BIAPY_AUG_DIR = Path(
    "metrics/biapy/biapy-aug-zarr-seunet-FDb/results/biapy-aug-zarr-seunet-FDb_0/aug"
)
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


def _as_slice(index_slice: slice | tuple[int, int | None]) -> slice:
    """Normalize ``slice`` or ``(start, stop)`` to a half-open ``slice``."""
    if isinstance(index_slice, slice):
        return index_slice
    if isinstance(index_slice, tuple) and len(index_slice) == 2:
        start, stop = index_slice
        return slice(start, stop)
    raise TypeError(
        "index_slice must be a slice or a (start, stop) tuple; "
        f"got {type(index_slice)!r}"
    )


def _biapy_aug_title(path: Path, max_len: int = 48) -> str:
    """Short title from a BiaPy aug dump filename."""
    stem = path.stem
    head, sep, rest = stem.partition("__")
    if not sep:
        return stem if len(stem) <= max_len else stem[: max_len - 1] + "…"
    tag = rest if len(rest) <= 24 else rest[:23] + "…"
    title = f"{head} | {tag}"
    return title if len(title) <= max_len else title[: max_len - 1] + "…"


def _orig_path_for_aug(aug_path: Path, *, raw: bool) -> Path:
    """Map ``*_x_aug_*`` / ``*_y_aug_*`` to the matching orig TIFF."""
    name = aug_path.name
    if raw:
        if "_x_aug_" not in name:
            raise ValueError(f"Expected '_x_aug_' in filename: {aug_path}")
        orig_name = name.replace("_x_aug_", "_orig_x_", 1)
    else:
        if "_y_aug_" not in name:
            raise ValueError(f"Expected '_y_aug_' in filename: {aug_path}")
        orig_name = name.replace("_y_aug_", "_orig_y_", 1)
    return aug_path.with_name(orig_name)


def _czyx_channel_mips(vol: np.ndarray, z_axis: int = 1) -> list[np.ndarray]:
    """Max-project each channel of a CZYX (or ZYX) volume to (Y, X)."""
    arr = np.asarray(vol)
    if arr.ndim == 3:
        return [arr.max(axis=0)]
    if arr.ndim != 4:
        raise ValueError(f"Expected CZYX or ZYX volume, got shape {arr.shape}")
    # CZYX: max over Z (axis 1 by default) → (C, Y, X)
    mips_cyx = arr.max(axis=z_axis)
    return [np.asarray(mips_cyx[c]) for c in range(mips_cyx.shape[0])]


def _resolve_label_channel_names(
    n_channels: int,
    channel_names: Sequence[str] | None,
) -> list[str]:
    if channel_names is not None:
        if len(channel_names) != n_channels:
            raise ValueError(
                f"channel_names length {len(channel_names)} != C={n_channels}"
            )
        return list(channel_names)
    if n_channels == 2:
        return ["F", "Db"]
    return [f"ch{i}" for i in range(n_channels)]


def _auto_channel_cmap(mip: np.ndarray) -> str:
    return "gray" if np.unique(mip).size <= 2 else "viridis"


def plot_biapy_aug_dump_samples(
    index_slice: slice | tuple[int, int | None],
    raw: bool,
    plotting_func: Callable | None = None,
    aug_dir: Path | str = DEFAULT_BIAPY_AUG_DIR,
    fig_size=(10, 5),
    channel_names: Sequence[str] | None = None,
):
    """Plot orig|aug MIP pairs from a BiaPy training aug dump folder.

    Selects from the sorted list of ``*_x_aug_*`` (``raw=True``) or
    ``*_y_aug_*`` (``raw=False``) TIFFs via a Python slice, finds each
    matching ``*_orig_*`` file, and shows one figure per pair.

    - ``raw=True``: requires ``plotting_func``; one 1x2 figure (orig | aug).
    - ``raw=False``: per-channel max-Z MIPs in a 2xC grid (orig / aug rows);
      ``plotting_func`` is ignored.
    """
    if raw and plotting_func is None:
        raise ValueError("plotting_func is required when raw=True")

    aug_dir = Path(aug_dir)
    if not aug_dir.is_dir():
        raise FileNotFoundError(f"Aug dump directory not found: {aug_dir}")

    kind_token = "_x_aug_" if raw else "_y_aug_"
    aug_paths = sorted(
        p for p in aug_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in _TIFF_SUFFIXES
        and kind_token in p.name
    )
    if not aug_paths:
        raise FileNotFoundError(
            f"No '{kind_token}' TIFF files found in {aug_dir}"
        )

    selected = aug_paths[_as_slice(index_slice)]
    if not selected:
        raise ValueError(
            f"index_slice={index_slice!r} selected 0 of {len(aug_paths)} "
            f"'{kind_token}' files in {aug_dir}"
        )
    print(f"Aug dir: {aug_dir}")
    print(f"Total '{kind_token}' files: {len(aug_paths)}; selected: {len(selected)}")

    pairs: list[tuple[Path, Path]] = []
    for aug_path in selected:
        orig_path = _orig_path_for_aug(aug_path, raw=raw)
        if not orig_path.is_file():
            raise FileNotFoundError(
                f"Matching orig not found for {aug_path.name}: expected {orig_path}"
            )
        pairs.append((orig_path, aug_path))

    def load_volume_czyx(path: Path) -> np.ndarray:
        img = _load_volume(path)
        print(
            f"{path.name}: shape={img.shape} dtype={img.dtype} "
            f"min={img.min()} max={img.max()}"
        )
        return _to_czyx(img, path, raw=raw)

    flat_paths = [p for pair in pairs for p in pair]
    with ThreadPoolExecutor(max_workers=2) as ex:
        volumes = list(
            tqdm(ex.map(load_volume_czyx, flat_paths), total=len(flat_paths))
        )

    for i, (orig_path, aug_path) in enumerate(pairs):
        orig_vol = volumes[2 * i]
        aug_vol = volumes[2 * i + 1]
        short = _biapy_aug_title(aug_path)

        if raw:
            orig_mip = plotting_func(orig_vol, 1)
            aug_mip = plotting_func(aug_vol, 1)
            fig, axes = plt.subplots(1, 2, figsize=fig_size, dpi=150, squeeze=False)
            axes[0, 0].imshow(orig_mip)
            axes[0, 0].set_title(f"orig\n{short}", fontsize=8)
            axes[0, 0].axis("off")
            axes[0, 1].imshow(aug_mip)
            axes[0, 1].set_title(f"aug\n{short}", fontsize=8)
            axes[0, 1].axis("off")
        else:
            orig_mips = _czyx_channel_mips(orig_vol)
            aug_mips = _czyx_channel_mips(aug_vol)
            n_c = len(orig_mips)
            if len(aug_mips) != n_c:
                raise ValueError(
                    f"Channel count mismatch for {aug_path.name}: "
                    f"orig C={n_c}, aug C={len(aug_mips)}"
                )
            labels = _resolve_label_channel_names(n_c, channel_names)
            fig, axes = plt.subplots(
                2, n_c, figsize=fig_size, dpi=150, squeeze=False
            )
            for c, name in enumerate(labels):
                axes[0, c].imshow(orig_mips[c], cmap=_auto_channel_cmap(orig_mips[c]))
                axes[0, c].set_title(f"orig · {name}", fontsize=8)
                axes[0, c].axis("off")
                axes[1, c].imshow(aug_mips[c], cmap=_auto_channel_cmap(aug_mips[c]))
                axes[1, c].set_title(f"aug · {name}", fontsize=8)
                axes[1, c].axis("off")
            fig.suptitle(short, fontsize=9)

        plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.2)
        plt.show()

