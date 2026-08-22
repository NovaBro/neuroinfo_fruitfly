"""Helpers for reviewing BiaPy metric / prediction / GT channel outputs."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import zarr

from .base_data import enhance_display

# FISBe volume id: {line}-{YYYYMMDD}_{nn}_{well}; trailing BiaPy aug_id is ignored.
_FISBE_SAMPLE_RE = re.compile(r"^(.+-\d{8}_\d+_[A-Z]\d+)")
_GT_TIFF_SUFFIXES = (".tif", ".tiff")


def _fisbe_sample_id(stem: str) -> str:
    """Strip trailing BiaPy aug_id; keep the FISBe volume id.

    Stems are ``{sample}{aug_id}``, e.g. ``R38F04-20181005_63_G3_c021_r0_k0``:
    sample ``R38F04-20181005_63_G3``, aug ``_c021_r0_k0``. No-aug stems and
    unmatched names are returned unchanged.
    """
    # Match {line}-{YYYYMMDD}_{nn}_{well}; leftover suffix is the augmentation.
    match = _FISBE_SAMPLE_RE.match(stem)
    return match.group(1) if match else stem


def _load_volume(path):
    """Load a TIFF file or Zarr array directory as ndarray."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _GT_TIFF_SUFFIXES:
        return tifffile.imread(path, mode="r")
    if suffix == ".zarr" and path.is_dir():
        return np.asarray(zarr.open(path.as_posix(), mode="r"))
    raise ValueError(f"Unsupported volume input: {path}")


def _as_zpyx(vol):
    """Normalize a 4D volume to (Z, P, Y, X)."""
    arr = np.asarray(vol)
    if arr.ndim != 4:
        return arr
    # TIFF GTs are typically (Z, P, Y, X); Zarr GTs here are often (Z, Y, X, P).
    if arr.shape[1] <= 16:
        return arr
    if arr.shape[-1] <= 16:
        return np.transpose(arr, (0, 3, 1, 2))
    return arr


def get_metric_paths(sub_folder: str, cf_stem: str, run: str = '0'):
    """Help navigate to biapy results stored in metric folder"""
    # 1. Build the BiaPy results directory: metrics/biapy/{stem}/results/{stem}_{run}/{sub_folder}
    result_path = Path(f"metrics/biapy/{cf_stem}/results/{cf_stem}_{run}/{sub_folder}")
    # 2. Collect all TIFF prediction/instance files in that folder
    image_paths = list(result_path.glob('*.tif'))
    return image_paths


def mip_biapy_gt_instance(ax, labels, sample_name, z_axis=0, font_siz=6):
    """Plot BiaPy instance MIP onto `ax`. Callable as plot_fn for plot_image_grid."""
    # 1. Max-project instance IDs along Z → 2D label map (Y, X)
    arr = np.asarray(labels)
    mip = arr.max(axis=z_axis)

    # 2. Paint each nonzero instance a random RGB color (skip background=0)
    rgb = np.zeros((*mip.shape, 3), dtype=np.float32)
    for lab in np.unique(mip):
        if lab == 0:
            continue
        color = np.random.randint(72, 255, 3).astype(np.float32)
        rgb[mip == lab] = color
    # 3. Scale to [0, 1] for imshow
    rgb /= 255.0

    # 4. Draw onto the provided axes
    ax.imshow(rgb)
    ax.set_title(f"GT instance MIP {sample_name}", wrap=True)
    ax.title.set_fontsize(font_siz)
    ax.axis("off")


def plot_image_grid(paths, plot_fn, ncols=4, figsize_cell=(2, 2), dpi=100, **plot_kwargs):
    """Load images from `paths` and draw each with `plot_fn` in a grid of `ncols` columns.

    `plot_fn` signature: plot_fn(ax, data, sample_name, **plot_kwargs)
    """
    # 1. Normalize paths and compute grid layout (rows × ncols)
    paths = [Path(p) for p in paths]
    n = len(paths)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_cell[0] * ncols, figsize_cell[1] * nrows),
        dpi=dpi,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    # 2. Load each TIFF and hand it to plot_fn (e.g. mip_biapy_gt_instance)
    for i, path in enumerate(paths):
        data = tifffile.imread(path, mode="r")
        plot_fn(axes_flat[i], data, path.stem, **plot_kwargs)

    # 3. Hide unused axes in the last incomplete row
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")
    plt.tight_layout()
    plt.show()
    return fig, axes


def resolve_per_image_paths(config_stem, run="0", sub_folder="per_image", n_samples=None):
    """List TIFFs under metrics/biapy/{stem}/results/{stem}_{run}/{sub_folder}."""
    # 1. Locate the BiaPy result subfolder for this config/run
    result_path = Path(f"metrics/biapy/{config_stem}/results/{config_stem}_{run}/{sub_folder}")
    # 2. Sorted glob so sample order is stable across calls
    paths = sorted(result_path.glob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"No .tif files in {result_path}")
    # 3. Optionally truncate to the first n_samples (faster preview)
    if n_samples is not None:
        paths = paths[: max(0, int(n_samples))]
        if not paths:
            raise ValueError("n_samples resolved to an empty path list")
    return paths


def resolve_prediction_channels(n_p_all, p=None, channel_names=None):
    """Normalize `p` to indices and build column labels. Returns (p_idx, labels)."""
    # 1. Resolve which prediction-channel indices to show (None → all)
    if p is None:
        p_idx = list(range(n_p_all))
    else:
        p_idx = [int(p)] if isinstance(p, (int, np.integer)) else [int(x) for x in p]
        bad = [j for j in p_idx if j < 0 or j >= n_p_all]
        if bad:
            raise ValueError(f"p out of range for P={n_p_all}: {bad}")

    # 2. Build column labels: defaults p0.., or subset/full channel_names list
    if channel_names is None:
        labels = [f"p{j}" for j in p_idx]
    elif len(channel_names) == n_p_all:
        labels = [channel_names[j] for j in p_idx]
    elif len(channel_names) == len(p_idx):
        labels = list(channel_names)
    else:
        raise ValueError(
            f"channel_names length {len(channel_names)} must be P={n_p_all} or len(p)={len(p_idx)}"
        )
    return p_idx, labels


def prediction_channel_mip(vol_zpyx, channel, threshold=None):
    """Max-project one prediction channel of (Z, P, Y, X) → (Y, X); optional Fiji-style cutoff."""
    # 1. Take channel slice over Z and max-project → (Y, X)
    mip = np.asarray(vol_zpyx[:, channel]).max(axis=0)
    # 2. Optional cutoff: zero voxels below threshold (keep supra-threshold intensities)
    if threshold is not None:
        mip = np.where(mip >= threshold, mip, 0)
    return mip


def auto_prediction_cmap(mip, cmap=None):
    """Binary (≤2 unique values) → gray; continuous → viridis. `cmap` overrides."""
    # Explicit override wins; else gray for binary masks, viridis for continuous scores
    if cmap is not None:
        return cmap
    return "gray" if np.unique(mip).size <= 2 else "viridis"


def draw_prediction_subplot(ax, mip, *, title=None, ylabel=None, font_siz=6, cmap="viridis"):
    """imshow display-enhanced MIP on `ax` with optional title/ylabel."""
    # 1. Contrast/gamma enhance, then imshow
    ax.imshow(enhance_display(mip), cmap=cmap)
    # 2. Optional labels (top-row titles / left-column sample names)
    if title is not None:
        ax.set_title(title, fontsize=font_siz)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=font_siz)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_per_image_predictions(
    config_stem: str,
    run: str = "0",
    *,
    sub_folder: str = "per_image",
    n_samples=None,
    p=None,
    threshold=None,
    figsize_cell=(3, 3),
    dpi=100,
    channel_names=None,
    font_siz=6,
    cmap=None,
):
    """Plot prediction-channel MIPs for TIFFs in a BiaPy result folder.

    Reads ``metrics/biapy/{config_stem}/results/{config_stem}_{run}/{sub_folder}``.
    Each image is ``(Z, P, Y, X)``; the figure has one row per sample and one column
    per selected prediction channel (max-intensity projection over Z).

    Colormap is chosen per subplot: ``gray`` when the MIP has ≤2 unique values,
    else ``viridis``. Pass ``cmap`` to override for all subplots.

    Parameters
    ----------
    n_samples : int or None
        If set, only load/plot the first ``n_samples`` TIFFs (faster).
    p : int, sequence of int, or None
        Prediction channel index/indices to show. ``None`` = all channels.
    threshold : float or None
        If set, values below ``threshold`` on the MIP are zeroed (Fiji-style cutoff).
    """
    # 1. Resolve TIFF paths under the BiaPy per_image (or similar) folder
    #    Layout is (Z, P, Y, X): Z depth, P prediction channels, then Y/X.
    paths = resolve_per_image_paths(config_stem, run, sub_folder, n_samples)

    # 2. Peek P from the first volume only (memmap) so we can validate `p` before the loop
    first = tifffile.imread(paths[0], mode="r")
    if first.ndim != 4:
        raise ValueError(f"Expected (Z, P, Y, X), got shape {first.shape} for {paths[0].name}")
    n_p_all = int(first.shape[1])
    del first

    # 3. Normalize channel selection → column indices + labels; allocate figure
    p_idx, labels = resolve_prediction_channels(n_p_all, p, channel_names)
    n, n_cols = len(paths), len(p_idx)
    fig, axes = plt.subplots(
        n,
        n_cols,
        figsize=(figsize_cell[0] * n_cols, figsize_cell[1] * n),
        dpi=dpi,
        squeeze=False,
    )

    # 4. Per sample × channel: MIP → enhance → draw (row=sample, col=channel)
    for i, path in enumerate(paths):
        # Memmap + project only selected channels so we do not materialize full volumes.
        data = tifffile.imread(path, mode="r")  # (Z, P, Y, X)
        for col, j in enumerate(p_idx):
            mip = prediction_channel_mip(data, j, threshold=threshold)
            draw_prediction_subplot(
                axes[i, col],
                mip,
                title=labels[col] if i == 0 else None,
                ylabel=path.stem if col == 0 else None,
                font_siz=font_siz,
                cmap=auto_prediction_cmap(mip, cmap),
            )
        del data

    # 5. Title + layout
    thresh_txt = f", thr≥{threshold}" if threshold is not None else ""
    fig.suptitle(f"{config_stem}_{run} / {sub_folder}{thresh_txt}", fontsize=font_siz + 2)
    plt.tight_layout()
    plt.show()
    return fig, axes


def resolve_gt_channel_paths(gt_path, n_samples=None, seed=42):
    """List GT volumes under a BiaPy instance-channel GT folder (label_F.…_Dn.…).

    When ``n_samples`` is set, pick that many unique FISBe volumes (any one
    augmentation each) instead of the first N sorted files/dirs, which would all
    be augs of the same volume.
    """
    # 1. Point at the GT label folder (e.g. fisbe/.../train/label_F.…_Dn.…)
    result_path = Path(gt_path)
    # 2. All supported GT volumes, sorted so n_samples=None is a stable listing
    tiff_paths = [
        p for p in result_path.iterdir() if p.is_file() and p.suffix.lower() in _GT_TIFF_SUFFIXES
    ]
    zarr_paths = [p for p in result_path.iterdir() if p.is_dir() and p.suffix.lower() == ".zarr"]
    paths = sorted([*tiff_paths, *zarr_paths])
    if not paths:
        raise FileNotFoundError(f"No TIFF files or .zarr directories in {result_path}")
    if n_samples is None:
        return paths

    n_samples = max(0, int(n_samples))
    if n_samples == 0:
        raise ValueError("n_samples resolved to an empty path list")

    # 3. Bucket paths by base sample so augs of one volume share a key
    #    R38F04-20181005_63_G3_c021_r0_k0 → sample "R38F04-20181005_63_G3"
    by_sample: dict[str, list[Path]] = {}
    for path in paths:
        by_sample.setdefault(_fisbe_sample_id(path.stem), []).append(path)

    # 4. Draw up to n_samples distinct volumes (not distinct TIFF files)
    sample_ids = list(by_sample)
    rng = np.random.default_rng(seed)
    n_pick = min(n_samples, len(sample_ids))
    chosen_ids = [sample_ids[i] for i in rng.permutation(len(sample_ids))[:n_pick]]

    # 5. For each chosen volume, keep any one of its augmentations
    return [
        by_sample[sample_id][int(rng.integers(0, len(by_sample[sample_id])))]
        for sample_id in chosen_ids
    ]


def plot_gt_instance_channels(
    gt_path,
    *,
    n_samples=None,
    seed=42,
    p=None,
    threshold=None,
    figsize_cell=(3, 3),
    dpi=100,
    channel_names=None,
    font_siz=6,
    cmap=None,
):
    """Plot BiaPy instance-channel GT MIPs (F/C/Db/Dn targets the network learns).

    Reads TIFF files or Zarr arrays from ``gt_path``
    (e.g. ``fisbe/.../train/label_F.…_Dn.…``). Each image is ``(Z, P, Y, X)``
    — same layout as ``per_image`` predictions.
    Reuses the prediction MIP / cmap / draw helpers.

    Parameters
    ----------
    gt_path : str or Path
        Directory of multi-channel GT volumes (TIFF files or .zarr directories).
    n_samples, p, threshold, channel_names, cmap
        Same meaning as ``plot_per_image_predictions``, except ``n_samples``
        selects unique FISBe volumes (any one augmentation each).
        If ``channel_names`` is None and ``P==4``, defaults to ``['F','C','Db','Dn']``.
    seed : int
        RNG seed for unique-volume / augmentation selection when ``n_samples`` is set.
    """
    # 1. List GT volumes — same (Z, P, Y, X) layout as predictions
    #    (F/C often binary; Db/Dn continuous distance-like targets)
    paths = resolve_gt_channel_paths(gt_path, n_samples, seed=seed)

    # 2. Infer channel count P from the first volume
    first = _as_zpyx(_load_volume(paths[0]))
    if first.ndim != 4:
        raise ValueError(f"Expected (Z, P, Y, X), got shape {first.shape} for {paths[0].name}")
    n_p_all = int(first.shape[1])
    print(f"Shape: {first.shape}")
    print(f"dtype: {first.dtype}")
    print(f"min: {first.min()}, max: {first.max()}, mean: {first.mean()}")
    # print(f"unique: {np.unique(first)[:10]}")

    del first

    # 3. Default F/C/Db/Dn names when P==4 and caller did not supply labels
    if channel_names is None and n_p_all == 4:
        channel_names = ["F", "C", "Db", "Dn"]

    # 4. Resolve channels + allocate figure (rows=samples, cols=channels)
    p_idx, labels = resolve_prediction_channels(n_p_all, p, channel_names)
    n, n_cols = len(paths), len(p_idx)
    fig, axes = plt.subplots(
        n,
        n_cols,
        figsize=(figsize_cell[0] * n_cols, figsize_cell[1] * n),
        dpi=dpi,
        squeeze=False,
    )

    # 5. Per sample × channel: MIP → enhance → draw (reuse prediction helpers)
    for i, path in enumerate(paths):
        data = _as_zpyx(_load_volume(path))  # (Z, P, Y, X)
        for col, j in enumerate(p_idx):
            mip = prediction_channel_mip(data, j, threshold=threshold)
            draw_prediction_subplot(
                axes[i, col],
                mip,
                title=labels[col] if i == 0 else None,
                ylabel=path.stem if col == 0 else None,
                font_siz=font_siz,
                cmap=auto_prediction_cmap(mip, cmap),
            )
        del data

    # 6. Title + layout
    thresh_txt = f", thr≥{threshold}" if threshold is not None else ""
    fig.suptitle(f"GT channels: {Path(gt_path).name}{thresh_txt}", fontsize=font_siz + 2)
    plt.tight_layout()
    plt.show()
    return fig, axes


def get_gt_paths(full_path: str):
    """Full path is mean to be .../label_F.erosion..."""
    # 1. Point at a BiaPy GT label directory (label_F.erosion… / etc.)
    result_path = Path(full_path)
    # 2. Collect all TIFF volumes in that folder
    image_paths = list(result_path.glob('*.tif'))
    return image_paths
