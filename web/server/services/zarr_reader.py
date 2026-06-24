"""Zarr volume slice and MIP extraction for API responses."""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import zarr
from PIL import Image

from services.volume_pipeline import (
    VolumeBytesResult,
    compute_downsample_factor,
    encode_label_volume_rgb,
    normalize_raw_volume,
    normalize_rgb_stack,
    to_display_rgb_from_channels,
    to_display_uint8,
)

VolumeKind = Literal["raw", "gt", "predicted"]
AxisKind = Literal["z", "y", "x"]
ChannelParam = int | Literal["all"]

RAW_KEY = "volumes/raw"
GT_KEY = "volumes/gt_instances"

# gt_instances stores each neuron mask in a separate channel (CZYX, uint8 labels).
# When building the 3D overlay we merge all channels into one colored volume; this
# stride keeps per-channel label ids in disjoint ranges so distinct neurons across
# channels get distinct colors (uint8 labels are <= 255, so 1000 leaves no overlap).
_GT_CHANNEL_LABEL_STRIDE = 1000

# Simple label colors for gt_instances display (R, G, B)
_GT_COLORS = np.array(
    [
        [0, 0, 0],
        [255, 80, 80],
        [80, 255, 80],
        [80, 80, 255],
        [255, 255, 80],
        [255, 80, 255],
        [80, 255, 255],
    ],
    dtype=np.uint8,
)


def _open_array(zarr_path: Path, volume: VolumeKind) -> zarr.Array:
    key = RAW_KEY if volume == "raw" else GT_KEY
    return zarr.open(str(zarr_path), mode="r", path=key)


def get_volume_meta(zarr_path: Path) -> dict:
    """Return shapes and dtypes for raw and gt_instances arrays."""
    meta: dict = {}
    for vol, key in (("raw", RAW_KEY), ("gt_instances", GT_KEY)):
        arr = zarr.open(str(zarr_path), mode="r", path=key)
        meta[vol] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
    return meta


def _normalize_raw_slice(slice_2d: np.ndarray) -> np.ndarray:
    """Contrast-enhanced 2D slice to uint8."""
    return to_display_uint8(slice_2d)


def _normalize_rgb_slab(slab_cyx: np.ndarray) -> np.ndarray:
    """Enhance a C,Y,X slab to Y,X,3 uint8 with shared contrast."""
    return to_display_rgb_from_channels(slab_cyx)


def _gt_slice_to_rgb(slice_2d: np.ndarray) -> np.ndarray:
    """Map integer label IDs to RGB for display."""
    labels = slice_2d.astype(np.int32)
    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    unique = np.unique(labels)
    for label_id in unique:
        if label_id == 0:
            continue
        color = _GT_COLORS[label_id % len(_GT_COLORS)]
        mask = labels == label_id
        rgb[mask] = color
    return rgb


def _extract_slice(
    arr: zarr.Array,
    volume: VolumeKind,
    channel: int,
    axis: AxisKind,
    index: int,
) -> np.ndarray:
    """Extract a 2D slice from a CZYX array."""
    c, z, y, x = arr.shape
    axis_sizes = {"z": z, "y": y, "x": x}
    size = axis_sizes[axis]
    index = max(0, min(index, size - 1))

    if volume == "raw":
        if channel < 0 or channel >= c:
            raise ValueError(f"channel must be 0..{c - 1}, got {channel}")
    else:
        gt_c = arr.shape[0]
        if channel < 0 or channel >= gt_c:
            raise ValueError(f"channel must be 0..{gt_c - 1}, got {channel}")

    if axis == "z":
        slice_2d = np.array(arr[channel, index, :, :])
    elif axis == "y":
        slice_2d = np.array(arr[channel, :, index, :])
    else:
        slice_2d = np.array(arr[channel, :, :, index])

    return slice_2d


def _extract_slice_rgb(
    arr: zarr.Array,
    axis: AxisKind,
    index: int,
) -> np.ndarray:
    """Extract an RGB slice (Y,X,3) from the first three CZYX channels."""
    c, z, y, x = arr.shape
    n_ch = min(c, 3)
    axis_sizes = {"z": z, "y": y, "x": x}
    size = axis_sizes[axis]
    index = max(0, min(index, size - 1))

    if axis == "z":
        slab = np.array(arr[:n_ch, index, :, :])
    elif axis == "y":
        slab = np.array(arr[:n_ch, :, index, :])
    else:
        slab = np.array(arr[:n_ch, :, :, index])

    return _normalize_rgb_slab(slab)


def _extract_mip_rgb(arr: zarr.Array) -> np.ndarray:
    """Maximum-intensity projection of RGB channels along Z."""
    n_ch = min(arr.shape[0], 3)
    mips = [np.array(arr[ch]).max(axis=0) for ch in range(n_ch)]
    return _normalize_rgb_slab(np.stack(mips, axis=0))


def _extract_mip(
    arr: zarr.Array,
    volume: VolumeKind,
    channel: int,
) -> np.ndarray:
    """Maximum-intensity projection along Z for a single channel."""
    if volume == "raw":
        c = arr.shape[0]
        if channel < 0 or channel >= c:
            raise ValueError(f"channel must be 0..{c - 1}, got {channel}")
        slab = arr[channel]
    else:
        gt_c = arr.shape[0]
        if channel < 0 or channel >= gt_c:
            raise ValueError(f"channel must be 0..{gt_c - 1}, got {channel}")
        slab = arr[channel]

    return np.array(slab).max(axis=0)


@lru_cache(maxsize=512)
def slice_to_png(
    zarr_path: str,
    volume: VolumeKind,
    channel: str,
    axis: AxisKind,
    index: int,
) -> bytes:
    zarr_path = Path(zarr_path)
    arr = _open_array(zarr_path, volume)

    if channel == "all":
        if volume != "raw":
            raise ValueError("channel=all is only supported for raw volumes")
        rgb = _extract_slice_rgb(arr, axis, index)
        img = Image.fromarray(rgb, mode="RGB")
    else:
        ch = int(channel)
        slice_2d = _extract_slice(arr, volume, ch, axis, index)
        if volume == "raw":
            gray = _normalize_raw_slice(slice_2d)
            img = Image.fromarray(gray, mode="L")
        else:
            rgb = _gt_slice_to_rgb(slice_2d)
            img = Image.fromarray(rgb, mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@lru_cache(maxsize=128)
def mip_to_png(
    zarr_path: str,
    volume: VolumeKind,
    channel: str,
) -> bytes:
    zarr_path = Path(zarr_path)
    arr = _open_array(zarr_path, volume)

    if channel == "all":
        if volume != "raw":
            raise ValueError("channel=all is only supported for raw volumes")
        rgb = _extract_mip_rgb(arr)
        img = Image.fromarray(rgb, mode="RGB")
    else:
        ch = int(channel)
        mip = _extract_mip(arr, volume, ch)
        if volume == "raw":
            gray = _normalize_raw_slice(mip)
            img = Image.fromarray(gray, mode="L")
        else:
            rgb = _gt_slice_to_rgb(mip)
            img = Image.fromarray(rgb, mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def downsample_max_pool_zarr(channel: zarr.Array, factor: int) -> np.ndarray:
    """Max-pool a Z,Y,X zarr channel without loading the full volume at once."""
    if factor <= 1:
        return np.array(channel)

    z, y, x = channel.shape
    nz, ny, nx = z // factor, y // factor, x // factor
    ty, tx = ny * factor, nx * factor
    if nz == 0 or ny == 0 or nx == 0:
        raise ValueError(f"Volume too small to downsample by factor {factor}")

    out = np.empty((nz, ny, nx), dtype=np.float32)
    for oz in range(nz):
        z_block = np.array(channel[oz * factor : (oz + 1) * factor, :ty, :tx])
        z_max = z_block.max(axis=0)
        out[oz] = z_max.reshape(ny, factor, nx, factor).max(axis=(1, 3))
    return out


def _validate_channel(arr: zarr.Array, channel: int) -> None:
    c = arr.shape[0]
    if channel < 0 or channel >= c:
        raise ValueError(f"channel must be 0..{c - 1}, got {channel}")


@lru_cache(maxsize=16)
def volume_to_bytes(
    zarr_path: str,
    volume: VolumeKind,
    channel: str,
    max_size: int,
) -> VolumeBytesResult:
    """Downsample a volume and return uint8 bytes for 3D rendering."""
    zarr_path = Path(zarr_path)
    if volume == "predicted":
        raise ValueError("predicted volume is loaded via biapy_loader, not zarr")

    if volume == "gt":
        # The ground-truth segmentation spans all channels (one neuron per
        # channel); merge them into a single colored instance volume regardless
        # of the requested channel.
        return _gt_volume_to_bytes(zarr_path, max_size)

    if channel == "all":
        return _volume_rgb_to_bytes(zarr_path, volume, max_size)

    ch = int(channel)
    arr = _open_array(zarr_path, volume)
    _validate_channel(arr, ch)
    channel_arr = arr[ch]
    original_shape = tuple(int(s) for s in channel_arr.shape)
    factor = compute_downsample_factor(original_shape, max_size)
    downsampled = downsample_max_pool_zarr(channel_arr, factor)
    normalized = normalize_raw_volume(downsampled)

    return VolumeBytesResult(
        data=normalized.tobytes(),
        shape=tuple(int(s) for s in normalized.shape),
        original_shape=original_shape,
        downsample_factor=factor,
        components=1,
    )


def _gt_volume_to_bytes(zarr_path: Path, max_size: int) -> VolumeBytesResult:
    """Merge all gt_instances channels into one colored Z,Y,X,3 instance volume."""
    arr = _open_array(zarr_path, "gt")
    n_ch = arr.shape[0]
    original_shape = tuple(int(s) for s in arr[0].shape)
    factor = compute_downsample_factor(original_shape, max_size)

    combined: np.ndarray | None = None
    for ch in range(n_ch):
        downsampled = downsample_max_pool_zarr(arr[ch], factor).astype(np.int64)
        if combined is None:
            combined = np.zeros(downsampled.shape, dtype=np.int64)
        # Give each (channel, label) a unique id so distinct neurons keep
        # distinct colors; background (0) stays 0. On overlap keep the larger
        # id so a voxel always renders some instance's color.
        encoded = downsampled + ch * _GT_CHANNEL_LABEL_STRIDE
        labelled = downsampled > 0
        combined = np.where(labelled & (encoded > combined), encoded, combined)

    if combined is None:
        raise ValueError("gt_instances has no channels")

    rgb = encode_label_volume_rgb(combined)
    return VolumeBytesResult(
        data=rgb.tobytes(),
        shape=tuple(int(s) for s in rgb.shape[:3]),
        original_shape=original_shape,
        downsample_factor=factor,
        components=3,
    )


def _volume_rgb_to_bytes(
    zarr_path: Path,
    volume: VolumeKind,
    max_size: int,
) -> VolumeBytesResult:
    if volume != "raw":
        raise ValueError("channel=all is only supported for raw volumes")

    arr = _open_array(zarr_path, volume)
    n_ch = min(arr.shape[0], 3)
    original_shape = tuple(int(s) for s in arr[0].shape)
    factor = compute_downsample_factor(original_shape, max_size)
    channels = [
        downsample_max_pool_zarr(arr[ch], factor) for ch in range(n_ch)
    ]
    stacked = np.stack(channels, axis=0)
    rgb = normalize_rgb_stack(stacked)

    return VolumeBytesResult(
        data=rgb.tobytes(),
        shape=tuple(int(s) for s in rgb.shape[:3]),
        original_shape=original_shape,
        downsample_factor=factor,
        components=3,
    )
