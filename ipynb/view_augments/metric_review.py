"""Helpers for reviewing BiaPy metric / prediction / GT channel outputs."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import zarr

from .base_data import enhance_display

# BiaPy watershed seed polarity (see instance_segmentation workflow docs).
_THRESH_ABOVE = frozenset({"F", "P", "Db", "D"})
_THRESH_BELOW = frozenset({"C", "B", "T", "Dn", "Dc"})
_KNOWN_CHANNELS = _THRESH_ABOVE | _THRESH_BELOW

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


def _as_instance_label_volume(vol):
    """Normalize an instance label volume to (Z, Y, X)."""
    arr = np.asarray(vol)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D instance labels, got shape {arr.shape}")
    return arr


def load_gt_instance_labels(gt_dir, stem):
    """Load instance-label GT for a prediction sample stem."""
    gt_path = Path(gt_dir) / f"{_fisbe_sample_id(stem)}.zarr"
    if not gt_path.is_dir():
        raise FileNotFoundError(f"Missing GT volume for {stem!r}: {gt_path}")
    return _as_instance_label_volume(_load_volume(gt_path))


def get_metric_paths(sub_folder: str, cf_stem: str, run: str = '0'):
    """Help navigate to biapy results stored in metric folder"""
    # 1. Build the BiaPy results directory: metrics/biapy/{stem}/results/{stem}_{run}/{sub_folder}
    result_path = Path(f"metrics/biapy/{cf_stem}/results/{cf_stem}_{run}/{sub_folder}")
    # 2. Collect all TIFF prediction/instance files in that folder
    image_paths = list(result_path.glob('*.tif'))
    return image_paths


def mip_biapy_gt_instance(
    ax,
    labels,
    sample_name,
    z_axis=0,
    font_siz=6,
    title=None,
):
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
    panel_title = title if title is not None else f"GT instance MIP {sample_name}"
    ax.set_title(panel_title, wrap=True)
    ax.title.set_fontsize(font_siz)
    ax.axis("off")


def plot_pred_gt_instance_mips(
    config_stem: str,
    run: str = "0",
    *,
    gt_dir,
    sub_folder: str = "per_image_post_processing",
    n_samples=None,
    figsize=(8, 4),
    dpi=100,
    font_siz=6,
):
    """Plot one GT-vs-prediction instance MIP figure per sample.

    Reads prediction TIFFs from
    ``metrics/biapy/{config_stem}/results/{config_stem}_{run}/{sub_folder}``
    and matching instance-label GT Zarr volumes from ``gt_dir``.

    Returns
    -------
    list[tuple[Figure, ndarray, str]]
        One entry per plotted sample: ``(fig, axes, stem)``.
    """
    gt_dir = Path(gt_dir)
    pred_paths = sorted(get_metric_paths(sub_folder, config_stem, run))
    if not pred_paths:
        raise FileNotFoundError(
            f"No prediction TIFFs under metrics/biapy/{config_stem}/results/"
            f"{config_stem}_{run}/{sub_folder}"
        )
    if n_samples is not None:
        pred_paths = pred_paths[: max(0, int(n_samples))]
        if not pred_paths:
            raise ValueError("n_samples resolved to an empty path list")

    results = []
    for pred_path in pred_paths:
        stem = pred_path.stem
        pred = _as_instance_label_volume(tifffile.imread(pred_path, mode="r"))
        gt = load_gt_instance_labels(gt_dir, stem)

        n_gt = int(np.unique(gt).size) - (1 if np.any(gt == 0) else 0)
        n_pred = int(np.unique(pred).size) - (1 if np.any(pred == 0) else 0)

        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        mip_biapy_gt_instance(axes[0], gt, stem, title=f"GT ({n_gt} inst)", font_siz=font_siz)
        mip_biapy_gt_instance(
            axes[1],
            pred,
            stem,
            title=f"Pred ({n_pred} inst)",
            font_siz=font_siz,
        )
        fig.suptitle(stem, fontsize=font_siz + 2)
        fig.tight_layout()
        plt.show()
        results.append((fig, axes, stem))
    return results


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


def channel_threshold_mode(name: str) -> Literal["above", "below"]:
    """Return BiaPy watershed threshold polarity for a channel code."""
    if name in _THRESH_ABOVE:
        return "above"
    if name in _THRESH_BELOW:
        return "below"
    raise ValueError(
        f"Unknown BiaPy channel {name!r}; expected one of {sorted(_KNOWN_CHANNELS)}"
    )


def _require_channel_names(
    channel_names: Sequence[str] | None,
    *,
    n_p_all: int,
    p_idx: list[int],
    threshold,
) -> None:
    n_cols = len(p_idx)
    if channel_names is not None:
        return
    if threshold is not None or n_cols > 1 or n_p_all > 1:
        raise ValueError(
            "channel_names is required when threshold is set or when plotting "
            f"more than one channel (P={n_p_all}, plotting {n_cols}); pass BiaPy "
            "DATA_CHANNELS codes, e.g. ['F', 'Dc', 'Dn', 'P']"
        )


def resolve_prediction_channels(
    n_p_all,
    p=None,
    channel_names=None,
    *,
    threshold=None,
):
    """Normalize `p` to indices and build column labels. Returns (p_idx, labels)."""
    if p is None:
        p_idx = list(range(n_p_all))
    else:
        p_idx = [int(p)] if isinstance(p, (int, np.integer)) else [int(x) for x in p]
        bad = [j for j in p_idx if j < 0 or j >= n_p_all]
        if bad:
            raise ValueError(f"p out of range for P={n_p_all}: {bad}")

    _require_channel_names(channel_names, n_p_all=n_p_all, p_idx=p_idx, threshold=threshold)

    if channel_names is None:
        labels = ["F"] if n_p_all == 1 else []
    elif len(channel_names) == n_p_all:
        labels = [channel_names[j] for j in p_idx]
    elif len(channel_names) == len(p_idx):
        labels = list(channel_names)
    else:
        raise ValueError(
            f"channel_names length {len(channel_names)} must be P={n_p_all} or len(p)={len(p_idx)}"
        )

    for name in labels:
        channel_threshold_mode(name)
    return p_idx, labels


def resolve_channel_thresholds(
    threshold: float | Mapping[str, float] | Sequence[float] | None,
    labels: Sequence[str],
) -> list[float | None]:
    """Resolve per-column thresholds from scalar, dict, or sequence input."""
    n_cols = len(labels)
    if threshold is None:
        return [None] * n_cols
    if isinstance(threshold, (int, float, np.floating)):
        return [float(threshold)] * n_cols
    if isinstance(threshold, Mapping):
        resolved: list[float | None] = []
        missing = [name for name in labels if name not in threshold]
        if missing:
            raise ValueError(
                f"threshold dict missing keys for plotted channels: {missing}; "
                f"expected keys {list(labels)}"
            )
        for name in labels:
            resolved.append(float(threshold[name]))
        return resolved
    values = [float(x) for x in threshold]
    if len(values) != n_cols:
        raise ValueError(
            f"threshold sequence length {len(values)} must match plotted channels ({n_cols})"
        )
    return values


def format_channel_title(name: str, thr: float | None, mode: Literal["above", "below"]) -> str:
    """Build a column title with optional BiaPy-style threshold annotation."""
    if thr is None:
        return name
    op = "≥" if mode == "above" else "≤"
    return f"{name}\nthr{op}{thr:g}"


def build_threshold_suptitle(
    labels: Sequence[str],
    thresholds: Sequence[float | None],
) -> str:
    """Compact suptitle suffix summarizing per-channel threshold directions."""
    parts: list[str] = []
    for name, thr in zip(labels, thresholds):
        if thr is None:
            continue
        mode = channel_threshold_mode(name)
        op = "≥" if mode == "above" else "≤"
        parts.append(f"thr{op}{thr:g} on {name}")
    return "; ".join(parts)


def prediction_channel_mip(
    vol_zpyx,
    channel: int,
    *,
    channel_name: str,
    threshold: float | None = None,
) -> np.ndarray:
    """Max-project one channel of (Z, P, Y, X) to (Y, X) with BiaPy-aware polarity.

    Uses max-intensity projection over Z for all channels. Threshold direction
    still follows BiaPy watershed semantics (``thr≥`` for F/P/Db/D,
    ``thr≤`` for Dc/Dn/C/T/B). When ``threshold`` is set, the result previews
    seed/growth masks rather than raw continuous scores.
    """
    stack = np.asarray(vol_zpyx[:, channel])
    mip = stack.max(axis=0)

    if threshold is not None:
        if channel_threshold_mode(channel_name) == "below":
            mip = np.where(mip <= threshold, mip, 0)
        else:
            mip = np.where(mip >= threshold, mip, 0)
    return mip


def auto_prediction_cmap(mip, cmap=None):
    """Binary (≤2 unique values) → gray; continuous → viridis. `cmap` overrides."""
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
    threshold: float | Mapping[str, float] | Sequence[float] | None = None,
    figsize_cell=(3, 3),
    dpi=100,
    channel_names=None,
    font_siz=6,
    cmap=None,
):
    """Plot prediction-channel MIPs for TIFFs in a BiaPy result folder.

    Reads ``metrics/biapy/{config_stem}/results/{config_stem}_{run}/{sub_folder}``.
    Each image is ``(Z, P, Y, X)``; the figure has one row per sample and one column
    per selected prediction channel.

    All channels use max-intensity projection over Z. Threshold polarity follows
    BiaPy watershed semantics: F/P/Db/D use ``thr≥``; Dc/Dn/C/T/B use ``thr≤``.

    Parameters
    ----------
    n_samples : int or None
        If set, only load/plot the first ``n_samples`` TIFFs (faster).
    p : int, sequence of int, or None
        Prediction channel index/indices to show. ``None`` = all channels.
    threshold : float, mapping, sequence, or None
        Optional per-channel cutoff after max-Z MIP. ``None`` (default) shows the
        raw continuous MIP with no masking. A float applies the same cutoff to
        every column; a dict keys by ``channel_names``; a sequence aligns with
        plotted columns. Polarity follows BiaPy: F/P/Db/D keep ``value ≥ thr``;
        Dc/Dn/C/T/B keep ``value ≤ thr``.
    channel_names : sequence of str or None
        BiaPy ``DATA_CHANNELS`` codes. Required when ``threshold`` is set or
        when plotting more than one channel.
    """
    paths = resolve_per_image_paths(config_stem, run, sub_folder, n_samples)

    first = tifffile.imread(paths[0], mode="r")
    if first.ndim != 4:
        raise ValueError(f"Expected (Z, P, Y, X), got shape {first.shape} for {paths[0].name}")
    n_p_all = int(first.shape[1])
    del first

    p_idx, labels = resolve_prediction_channels(
        n_p_all, p, channel_names, threshold=threshold
    )
    col_thresholds = resolve_channel_thresholds(threshold, labels)
    n, n_cols = len(paths), len(p_idx)
    fig, axes = plt.subplots(
        n,
        n_cols,
        figsize=(figsize_cell[0] * n_cols, figsize_cell[1] * n),
        dpi=dpi,
        squeeze=False,
    )

    for i, path in enumerate(paths):
        data = tifffile.imread(path, mode="r")
        for col, j in enumerate(p_idx):
            ch_name = labels[col]
            thr = col_thresholds[col]
            mode = channel_threshold_mode(ch_name)
            mip = prediction_channel_mip(
                data,
                j,
                channel_name=ch_name,
                threshold=thr,
            )
            draw_prediction_subplot(
                axes[i, col],
                mip,
                title=format_channel_title(ch_name, thr, mode) if i == 0 else None,
                ylabel=path.stem if col == 0 else None,
                font_siz=font_siz,
                cmap=auto_prediction_cmap(mip, cmap),
            )
        del data

    base_title = f"{config_stem}_{run} / {sub_folder}"
    thresh_txt = build_threshold_suptitle(labels, col_thresholds)
    fig.suptitle(
        f"{base_title}, {thresh_txt}" if thresh_txt else base_title,
        fontsize=font_siz + 2,
    )
    # Leave headroom so multi-line column titles (e.g. "F\nthr≥0.5") do not
    # collide with the figure suptitle.
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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
    threshold: float | Mapping[str, float] | Sequence[float] | None = None,
    figsize_cell=(3, 3),
    dpi=100,
    channel_names=None,
    font_siz=6,
    cmap=None,
):
    """Plot BiaPy instance-channel GT MIPs (F/Dc/Dn/P targets the network learns).

    Reads TIFF files or Zarr arrays from ``gt_path``
    (e.g. ``fisbe/.../train/label_F.…_Dn.…``). Each image is ``(Z, P, Y, X)``
    — same layout as ``per_image`` predictions.

    Parameters
    ----------
    gt_path : str or Path
        Directory of multi-channel GT volumes (TIFF files or .zarr directories).
    n_samples : int or None
        If set, select that many unique FISBe volumes (any one augmentation each).
        ``None`` lists all volumes in sorted order.
    seed : int
        RNG seed for unique-volume / augmentation selection when ``n_samples`` is set.
    p : int, sequence of int, or None
        Channel index/indices to show. ``None`` = all channels.
    threshold : float, mapping, sequence, or None
        Optional per-channel cutoff after max-Z MIP. ``None`` (default) shows the
        raw continuous MIP with no masking. A float applies the same cutoff to
        every column; a dict keys by ``channel_names``; a sequence aligns with
        plotted columns. Polarity follows BiaPy: F/P/Db/D keep ``value ≥ thr``;
        Dc/Dn/C/T/B keep ``value ≤ thr``.
    channel_names : sequence of str or None
        BiaPy ``DATA_CHANNELS`` codes. Required when ``threshold`` is set or when
        plotting more than one channel. If ``None`` and ``P==4``, defaults to
        ``['F', 'Dc', 'Dn', 'P']``.
    cmap : str or None
        Override colormap for all subplots. ``None`` auto-picks gray for binary
        MIPs (≤2 unique values) else viridis.
    """
    paths = resolve_gt_channel_paths(gt_path, n_samples, seed=seed)

    first = _as_zpyx(_load_volume(paths[0]))
    if first.ndim != 4:
        raise ValueError(f"Expected (Z, P, Y, X), got shape {first.shape} for {paths[0].name}")
    n_p_all = int(first.shape[1])
    print(f"Shape: {first.shape}")
    print(f"dtype: {first.dtype}")
    print(f"min: {first.min()}, max: {first.max()}, mean: {first.mean()}")

    del first

    if channel_names is None and n_p_all == 4:
        channel_names = ["F", "Dc", "Dn", "P"]

    p_idx, labels = resolve_prediction_channels(
        n_p_all, p, channel_names, threshold=threshold
    )
    col_thresholds = resolve_channel_thresholds(threshold, labels)
    n, n_cols = len(paths), len(p_idx)
    fig, axes = plt.subplots(
        n,
        n_cols,
        figsize=(figsize_cell[0] * n_cols, figsize_cell[1] * n),
        dpi=dpi,
        squeeze=False,
    )

    for i, path in enumerate(paths):
        data = _as_zpyx(_load_volume(path))
        for col, j in enumerate(p_idx):
            ch_name = labels[col]
            thr = col_thresholds[col]
            mode = channel_threshold_mode(ch_name)
            mip = prediction_channel_mip(
                data,
                j,
                channel_name=ch_name,
                threshold=thr,
            )
            draw_prediction_subplot(
                axes[i, col],
                mip,
                title=format_channel_title(ch_name, thr, mode) if i == 0 else None,
                ylabel=path.stem if col == 0 else None,
                font_siz=font_siz,
                cmap=auto_prediction_cmap(mip, cmap),
            )
        del data

    base_title = f"GT channels: {Path(gt_path).name}"
    thresh_txt = build_threshold_suptitle(labels, col_thresholds)
    fig.suptitle(
        f"{base_title}, {thresh_txt}" if thresh_txt else base_title,
        fontsize=font_siz + 2,
    )
    # Leave headroom so multi-line column titles do not collide with the
    # figure suptitle.
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    return fig, axes


def get_gt_paths(full_path: str):
    """Full path is mean to be .../label_F.erosion..."""
    # 1. Point at a BiaPy GT label directory (label_F.erosion… / etc.)
    result_path = Path(full_path)
    # 2. Collect all TIFF volumes in that folder
    image_paths = list(result_path.glob('*.tif'))
    return image_paths
