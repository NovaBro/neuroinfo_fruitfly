"""Shared 3D volume downsampling and encoding for API responses."""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import numpy as np

VolumeEncoding = Literal["raw", "labels", "labels_rgb"]

# Display tuning: percentile clip + gamma (< 1 brightens mid-tones)
_DISPLAY_PCT_LOW = 1.0
_DISPLAY_PCT_HIGH = 99.5
_DISPLAY_GAMMA = 0.72
_LABEL_COLOR_SEED = 42


def enhance_display_values(data: np.ndarray) -> np.ndarray:
    """Contrast-stretch with percentile clipping and gamma brighten."""
    values = data.astype(np.float32)
    if not np.any(values > 0):
        return np.zeros_like(values)

    sample = values[values > 0] if np.count_nonzero(values) > 256 else values.ravel()
    p_low, p_high = np.percentile(sample, [_DISPLAY_PCT_LOW, _DISPLAY_PCT_HIGH])
    if p_high <= p_low:
        p_low, p_high = float(values.min()), float(values.max())
    if p_high <= p_low:
        return np.zeros_like(values)

    scaled = np.clip((values - p_low) / (p_high - p_low), 0.0, 1.0)
    return np.power(scaled, _DISPLAY_GAMMA)


def to_display_uint8(data: np.ndarray) -> np.ndarray:
    return (enhance_display_values(data) * 255).astype(np.uint8)


def to_display_rgb_from_channels(data_cyx: np.ndarray) -> np.ndarray:
    """Enhance C,Y,X data with shared contrast and return Y,X,3 uint8."""
    enhanced = enhance_display_values(data_cyx)
    uint8 = (enhanced * 255).astype(np.uint8)
    return np.moveaxis(uint8, 0, -1)


class VolumeBytesResult(NamedTuple):
    data: bytes
    shape: tuple[int, int, int]
    original_shape: tuple[int, int, int]
    downsample_factor: int
    components: int = 1


def downsample_max_pool(volume: np.ndarray, factor: int) -> np.ndarray:
    """Block max-pool a Z,Y,X volume by an integer factor."""
    if factor <= 1:
        return volume
    z, y, x = volume.shape
    nz, ny, nx = z // factor, y // factor, x // factor
    if nz == 0 or ny == 0 or nx == 0:
        raise ValueError(f"Volume too small to downsample by factor {factor}")
    trimmed = volume[: nz * factor, : ny * factor, : nx * factor]
    return trimmed.reshape(nz, factor, ny, factor, nx, factor).max(axis=(1, 3, 5))


def normalize_raw_volume(volume: np.ndarray) -> np.ndarray:
    """Contrast-enhanced uint8 volume for display."""
    return to_display_uint8(volume)


def normalize_rgb_stack(stacked_czyx: np.ndarray) -> np.ndarray:
    """Normalize C,Z,Y,X float stack to Z,Y,X,3 uint8 with shared contrast."""
    n_ch = min(stacked_czyx.shape[0], 3)
    enhanced = enhance_display_values(stacked_czyx[:n_ch])
    uint8 = (enhanced * 255).astype(np.uint8)
    return np.moveaxis(uint8, 0, -1)


def _label_color(label_id: int) -> tuple[int, int, int]:
    """Deterministic pseudo-random vivid RGB for an instance label."""
    rng = np.random.default_rng(int(label_id) * 2_654_435_761 + _LABEL_COLOR_SEED)
    return tuple(int(v) for v in rng.integers(72, 256, size=3))


def encode_label_volume_rgb(volume: np.ndarray) -> np.ndarray:
    """Map instance label IDs to Z,Y,X,3 RGB (background black)."""
    labels = volume.astype(np.int32)
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        rgb[labels == label_id] = _label_color(label_id)
    return rgb


def encode_label_volume(volume: np.ndarray) -> np.ndarray:
    """Encode instance labels as uint8 mask (non-zero voxels scaled for MIP visibility)."""
    labels = volume.astype(np.float32)
    mask = labels > 0
    if not mask.any():
        return np.zeros(labels.shape, dtype=np.uint8)
    encoded = np.zeros(labels.shape, dtype=np.uint8)
    max_label = float(labels[mask].max())
    encoded[mask] = np.clip((labels[mask] / max_label) * 255, 64, 255).astype(np.uint8)
    return encoded


def compute_downsample_factor(shape: tuple[int, int, int], max_size: int) -> int:
    z, y, x = shape
    return max(1, math.ceil(max(z, y, x) / max_size))


def volume_array_to_bytes(
    volume: np.ndarray,
    *,
    max_size: int,
    encoding: VolumeEncoding = "raw",
) -> VolumeBytesResult:
    """Downsample and encode a Z,Y,X volume for browser-side 3D MIP rendering."""
    if volume.ndim != 3:
        raise ValueError(f"Expected Z,Y,X volume, got shape {volume.shape}")

    original_shape = tuple(int(s) for s in volume.shape)
    factor = compute_downsample_factor(original_shape, max_size)
    downsampled = downsample_max_pool(np.asarray(volume), factor)
    if encoding == "raw":
        normalized = normalize_raw_volume(downsampled)
        components = 1
    elif encoding == "labels_rgb":
        normalized = encode_label_volume_rgb(downsampled)
        components = 3
    else:
        normalized = encode_label_volume(downsampled)
        components = 1

    shape = (
        tuple(int(s) for s in normalized.shape[:3])
        if normalized.ndim == 4
        else tuple(int(s) for s in normalized.shape)
    )

    return VolumeBytesResult(
        data=normalized.tobytes(),
        shape=shape,
        original_shape=original_shape,
        downsample_factor=factor,
        components=components,
    )
