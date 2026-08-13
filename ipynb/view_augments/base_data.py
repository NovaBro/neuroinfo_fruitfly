"""Base FISBe imaging and per-channel stats helpers for augmentation notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORIGINAL_DIR = Path("fisbe/completely/test")
ORIGINAL_MIP_DIR = Path("fisbe/mips/completely/test")

PCT_LOW, PCT_HIGH, GAMMA = 1.0, 99.5, 0.72


def gen_mip(raw, axis) -> np.ndarray:
    """Max-intensity projection of RGB channels. For CZYX: axis=1 is the z projection."""
    mip = raw.astype(np.float32).max(axis=axis)
    mip = (mip - mip.min()) / (np.ptp(mip) + 1e-8)
    return mip


def enhance_display(data: np.ndarray) -> np.ndarray:
    """Shared contrast + gamma; same defaults as web/server/services/volume_pipeline.py."""
    v = data.astype(np.float32)
    if not np.any(v > 0):
        return np.zeros_like(v)
    sample = v[v > 0] if np.count_nonzero(v) > 256 else v.ravel()
    lo, hi = np.percentile(sample, [PCT_LOW, PCT_HIGH])
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return np.zeros_like(v)
    return np.clip((v - lo) / (hi - lo), 0, 1) ** GAMMA


def fisbe_rgb_mip(raw_czyx, z_axis=1) -> np.ndarray:
    """raw: CZYX. Returns Y×X×3 uint8, web-matched vibrance."""
    c = min(raw_czyx.shape[0], 3)
    mip_cyx = np.asarray(raw_czyx[:c], dtype=np.float32).max(axis=z_axis)  # C,Y,X
    rgb01 = enhance_display(mip_cyx)  # shared stretch across channels (keeps hue)
    return np.moveaxis((rgb01 * 255).astype(np.uint8), 0, -1)


def per_channel_stats(data: np.ndarray, axis: int = 0) -> pd.DataFrame:
    """Basic stats of values across channels (axis 0 for CZYX).

    Expects shape like (C, Z, Y, X), e.g. (3, 390, 680, 680) uint16.
    """
    data = np.asarray(data)
    n_ch = data.shape[axis]
    rows = []
    for c in range(n_ch):
        ch = np.take(data, c, axis=axis).ravel()
        nonzero = ch[ch > 0]
        rows.append({
            "channel": c,
            "dtype": str(data.dtype),
            "n": ch.size,
            "n_nonzero": int(nonzero.size),
            "min": int(ch.min()) if ch.size else np.nan,
            "max": int(ch.max()) if ch.size else np.nan,
            "mean": float(ch.mean()) if ch.size else np.nan,
            "std": float(ch.std()) if ch.size else np.nan,
            "median": float(np.median(ch)) if ch.size else np.nan,
            "p1": float(np.percentile(ch, 1)) if ch.size else np.nan,
            "p99": float(np.percentile(ch, 99)) if ch.size else np.nan,
            "mean_nonzero": float(nonzero.mean()) if nonzero.size else 0.0,
            "median_nonzero": float(np.median(nonzero)) if nonzero.size else 0.0,
        })
    return pd.DataFrame(rows).set_index("channel")


def per_channel_histogram(
    data: np.ndarray,
    sample_name: str,
    axis: int = 0,
    bins: int = 256,
    range_=None,
    skip_zeros: bool = True,
    log_y: bool = True,
    figsize=(12, 3.5),
):
    """Basic histogram of value distribution across channels (axis 0 for CZYX).

    Plots one subplot per channel. Returns (fig, axes, counts).
    """
    data = np.asarray(data)
    n_ch = data.shape[axis]
    if range_ is None:
        range_ = (int(data.min()), int(data.max()) + 1)

    fig, axes = plt.subplots(1, n_ch, figsize=figsize, sharey=True)
    if n_ch == 1:
        axes = [axes]

    counts = []
    for c, ax in enumerate(axes):
        ch = np.take(data, c, axis=axis).ravel()
        if skip_zeros:
            ch = ch[ch > 0]
        hist, edges = np.histogram(ch, bins=bins, range=range_)
        counts.append(hist)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, hist, width=np.diff(edges), align="center", alpha=0.8)
        ax.set_title(f"channel {c}")
        ax.set_xlabel("value")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlim(range_)

    axes[0].set_ylabel("count" + (" (log)" if log_y else ""))
    fig.suptitle(f"Per-channel value histogram {sample_name}", y=1.02)
    fig.tight_layout()
    return fig, axes, counts
