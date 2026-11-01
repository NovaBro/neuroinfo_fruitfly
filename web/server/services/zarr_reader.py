"""Zarr volume slice and MIP extraction for API responses."""

from __future__ import annotations

import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Literal, Sequence

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
from services.zarr_store import open_array as open_cached_array

VolumeKind = Literal["raw", "gt", "predicted"]
AxisKind = Literal["z", "y", "x"]
ChannelParam = int | Literal["all"]

RAW_KEY = "volumes/raw"
GT_KEY = "volumes/gt_instances"

# Default voxel size [vx, vy, vz] µm (X, Y, Z) when zarr root attrs lack
# ``voxel_size_um``. FISBe / synth are 0.44 µm isotropic; Gen1 MCFO (himanshu)
# defaults match read_h5j when the h5j attr is missing.
_DEFAULT_VOXEL_SIZE_UM: dict[str, tuple[float, float, float]] = {
    "completely": (0.44, 0.44, 0.44),
    "partly": (0.44, 0.44, 0.44),
    "synth_vh": (0.44, 0.44, 0.44),
    "synth_vh_u8": (0.44, 0.44, 0.44),
    "1k_v1": (0.44, 0.44, 0.44),
    "1k_v1_u8": (0.44, 0.44, 0.44),
    "himanshu": (0.461122, 0.461122, 0.7),
}
_FALLBACK_VOXEL_SIZE_UM = (0.44, 0.44, 0.44)

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

_logger = logging.getLogger("uvicorn.error")


def allocated_cpus() -> tuple[int, str]:
    """CPUs available to this process for the downsample thread pool.

    Prefers SLURM ``--cpus-per-task`` (``SLURM_CPUS_PER_TASK``), then the
    process CPU affinity set (cgroup under SLURM), then ``os.cpu_count()``.
    """
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        try:
            n = int(slurm)
            if n >= 1:
                return n, "SLURM_CPUS_PER_TASK"
        except ValueError:
            pass
    try:
        n = len(os.sched_getaffinity(0))
        if n >= 1:
            return n, "affinity"
    except (AttributeError, OSError):
        pass
    return max(1, os.cpu_count() or 1), "cpu_count"


_POOL_WORKERS, _POOL_SOURCE = allocated_cpus()
_logger.info(
    "downsample pool workers=%s (from %s)",
    _POOL_WORKERS,
    _POOL_SOURCE,
)


def _open_array(zarr_path: Path, volume: VolumeKind) -> zarr.Array:
    key = RAW_KEY if volume == "raw" else GT_KEY
    return open_cached_array(zarr_path, key)


def resolve_voxel_size_um(
    zarr_path: Path | str,
    dataset: str,
) -> tuple[float, float, float]:
    """Return (vx, vy, vz) microns from zarr root attrs or dataset default.

    Attr key is ``voxel_size_um`` with X, Y, Z order (matches Gen1 h5j /
    synth_vh). Falls back to dataset defaults when missing or invalid.
    """
    path = Path(zarr_path)
    try:
        root = zarr.open(str(path), mode="r")
        raw = root.attrs.get("voxel_size_um")
        if raw is not None:
            vals = [float(v) for v in list(raw)[:3]]
            if len(vals) == 3 and all(v > 0 and np.isfinite(v) for v in vals):
                return (vals[0], vals[1], vals[2])
    except Exception:
        _logger.debug("voxel_size_um unread for %s", path, exc_info=True)

    return _DEFAULT_VOXEL_SIZE_UM.get(dataset, _FALLBACK_VOXEL_SIZE_UM)


def get_volume_meta(zarr_path: Path, dataset: str = "completely") -> dict:
    """Return shapes, dtypes, and voxel size for raw and gt_instances arrays.

    ``gt_instances`` is ``None`` when the array is absent (e.g. himanshu raw-only).
    ``voxel_size_um`` is always ``[vx, vy, vz]`` in microns (X, Y, Z).
    """
    raw = open_cached_array(zarr_path, RAW_KEY)
    vx, vy, vz = resolve_voxel_size_um(zarr_path, dataset)
    meta: dict = {
        "raw": {
            "shape": list(raw.shape),
            "dtype": str(raw.dtype),
        },
        "gt_instances": None,
        "voxel_size_um": [vx, vy, vz],
    }
    try:
        gt = open_cached_array(zarr_path, GT_KEY)
        meta["gt_instances"] = {
            "shape": list(gt.shape),
            "dtype": str(gt.dtype),
        }
    except Exception:
        pass
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


# Z-planes read per block when max-projecting. Full FISBe channels are ~GB, so
# loading a whole channel to MIP it can OOM the host; project in bounded slabs.
_MIP_Z_BLOCK = 32


def _channel_selector(
    channels: slice | Sequence[int] | int,
) -> slice | list[int] | int:
    if isinstance(channels, slice) or isinstance(channels, int):
        return channels
    return list(channels)


def _max_project_channels(
    arr: zarr.Array,
    channels: slice | Sequence[int] | int,
) -> np.ndarray:
    """MIP along Z for one or more CZYX channels, reading all selected C together.

    Returns ``(Y, X)`` for a single channel index, or ``(C, Y, X)`` otherwise.
    """
    z = arr.shape[1]
    sel = _channel_selector(channels)
    out: np.ndarray | None = None
    for zi in range(0, z, _MIP_Z_BLOCK):
        block = np.asarray(arr[sel, zi : zi + _MIP_Z_BLOCK, :, :])
        # block: (C, zb, Y, X) or (zb, Y, X) when sel is an int
        if block.ndim == 3:
            reduced = block.max(axis=0)
        else:
            reduced = block.max(axis=1)
        out = reduced if out is None else np.maximum(out, reduced)
    if out is None:
        raise ValueError("volume has no Z planes to project")
    return out


def _max_project_channel(arr: zarr.Array, channel: int) -> np.ndarray:
    """MIP along Z for one CZYX channel, reading Z in blocks to bound memory."""
    return _max_project_channels(arr, channel)


def _extract_mip_rgb(arr: zarr.Array) -> np.ndarray:
    """Maximum-intensity projection of RGB channels along Z."""
    n_ch = min(arr.shape[0], 3)
    mips = _max_project_channels(arr, slice(0, n_ch))
    return _normalize_rgb_slab(mips)


def _extract_mip(
    arr: zarr.Array,
    volume: VolumeKind,
    channel: int,
) -> np.ndarray:
    """Maximum-intensity projection along Z for a single channel."""
    c = arr.shape[0]
    if channel < 0 or channel >= c:
        raise ValueError(f"channel must be 0..{c - 1}, got {channel}")

    return _max_project_channel(arr, channel)


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


def _merged_gt_mip_rgb(arr: zarr.Array) -> np.ndarray:
    """MIP of *all* gt_instances channels merged into one colored (Y,X,3) image.

    Mirrors the merge/coloring in `_gt_volume_to_bytes` so the 2D GT MIP matches
    the 3D GT overlay. Reads every channel in one Z-blocked pass.
    """
    n_ch = arr.shape[0]
    if n_ch == 0:
        raise ValueError("gt_instances has no channels")

    mips = _max_project_channels(arr, slice(0, n_ch)).astype(np.int64)  # (C, Y, X)
    combined = np.zeros(mips.shape[1:], dtype=np.int64)
    for ch in range(n_ch):
        mip = mips[ch]
        encoded = mip + ch * _GT_CHANNEL_LABEL_STRIDE
        labelled = mip > 0
        combined = np.where(labelled & (encoded > combined), encoded, combined)

    # encode_label_volume_rgb works on Z,Y,X → Z,Y,X,3; feed a single-plane stack.
    return encode_label_volume_rgb(combined[np.newaxis, ...])[0]


@lru_cache(maxsize=128)
def mip_to_png(
    zarr_path: str,
    volume: VolumeKind,
    channel: str,
) -> bytes:
    zarr_path = Path(zarr_path)
    arr = _open_array(zarr_path, volume)

    if channel == "all":
        if volume == "raw":
            rgb = _extract_mip_rgb(arr)
        elif volume == "gt":
            # Merge every instance channel into one colored MIP (see helper).
            rgb = _merged_gt_mip_rgb(arr)
        else:
            raise ValueError("channel=all is only supported for raw and gt volumes")
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


def _pool_z_range(
    arr: zarr.Array,
    channels: slice | Sequence[int],
    factor: int,
    oz_start: int,
    oz_end: int,
    ty: int,
    tx: int,
    ny: int,
    nx: int,
    out: np.ndarray,
) -> None:
    """Max-pool output-Z slabs ``[oz_start, oz_end)`` into ``out`` (C, nz, ny, nx)."""
    sel = _channel_selector(channels)
    for oz in range(oz_start, oz_end):
        z0 = oz * factor
        z1 = z0 + factor
        block = np.asarray(arr[sel, z0:z1, :ty, :tx])  # (C, factor, ty, tx)
        z_max = block.max(axis=1)  # (C, ty, tx)
        out[:, oz] = z_max.reshape(z_max.shape[0], ny, factor, nx, factor).max(
            axis=(2, 4)
        )


def downsample_max_pool_czyx(
    arr: zarr.Array,
    channels: slice | Sequence[int],
    factor: int,
) -> np.ndarray:
    """Max-pool selected CZYX channels, reading all selected C once per Z slab.

    Returns ``(C, nz, ny, nx)`` float32. Z-slabs are processed in parallel with
    up to ``_POOL_WORKERS`` threads (sized from SLURM ``--cpus-per-task``).
    """
    if isinstance(channels, slice):
        ch_sel: slice | list[int] = channels
        n_ch = len(range(*channels.indices(arr.shape[0])))
    else:
        ch_sel = list(channels)
        n_ch = len(ch_sel)

    if n_ch == 0:
        raise ValueError("no channels to downsample")

    _, z, y, x = arr.shape
    if factor <= 1:
        data = np.asarray(arr[ch_sel, :, :, :], dtype=np.float32)
        return data

    nz, ny, nx = z // factor, y // factor, x // factor
    ty, tx = ny * factor, nx * factor
    if nz == 0 or ny == 0 or nx == 0:
        raise ValueError(f"Volume too small to downsample by factor {factor}")

    out = np.empty((n_ch, nz, ny, nx), dtype=np.float32)
    workers = min(_POOL_WORKERS, nz)
    if workers <= 1:
        _pool_z_range(arr, ch_sel, factor, 0, nz, ty, tx, ny, nx, out)
        return out

    # Split output-Z into contiguous ranges for each worker.
    bounds = [i * nz // workers for i in range(workers + 1)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _pool_z_range,
                arr,
                ch_sel,
                factor,
                bounds[i],
                bounds[i + 1],
                ty,
                tx,
                ny,
                nx,
                out,
            )
            for i in range(workers)
            if bounds[i] < bounds[i + 1]
        ]
        for fut in futures:
            fut.result()
    return out


def downsample_max_pool_zarr(channel: zarr.Array, factor: int) -> np.ndarray:
    """Max-pool a Z,Y,X zarr channel without loading the full volume at once.

    Prefer :func:`downsample_max_pool_czyx` when the parent CZYX array is
    available so multi-channel reads share Blosc chunks.
    """
    if factor <= 1:
        return np.array(channel)

    z, y, x = channel.shape
    nz, ny, nx = z // factor, y // factor, x // factor
    ty, tx = ny * factor, nx * factor
    if nz == 0 or ny == 0 or nx == 0:
        raise ValueError(f"Volume too small to downsample by factor {factor}")

    out = np.empty((nz, ny, nx), dtype=np.float32)

    def _one_range(oz_start: int, oz_end: int) -> None:
        for oz in range(oz_start, oz_end):
            z_block = np.array(channel[oz * factor : (oz + 1) * factor, :ty, :tx])
            z_max = z_block.max(axis=0)
            out[oz] = z_max.reshape(ny, factor, nx, factor).max(axis=(1, 3))

    workers = min(_POOL_WORKERS, nz)
    if workers <= 1:
        _one_range(0, nz)
        return out

    bounds = [i * nz // workers for i in range(workers + 1)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_one_range, bounds[i], bounds[i + 1])
            for i in range(workers)
            if bounds[i] < bounds[i + 1]
        ]
        for fut in futures:
            fut.result()
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
    original_shape = tuple(int(s) for s in arr.shape[1:])
    factor = compute_downsample_factor(original_shape, max_size)
    downsampled = downsample_max_pool_czyx(arr, [ch], factor)[0]
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
    if n_ch == 0:
        raise ValueError("gt_instances has no channels")

    original_shape = tuple(int(s) for s in arr.shape[1:])
    factor = compute_downsample_factor(original_shape, max_size)
    downsampled = downsample_max_pool_czyx(arr, slice(0, n_ch), factor).astype(
        np.int64
    )  # (C, nz, ny, nx)

    combined = np.zeros(downsampled.shape[1:], dtype=np.int64)
    for ch in range(n_ch):
        # Give each (channel, label) a unique id so distinct neurons keep
        # distinct colors; background (0) stays 0. On overlap keep the larger
        # id so a voxel always renders some instance's color.
        encoded = downsampled[ch] + ch * _GT_CHANNEL_LABEL_STRIDE
        labelled = downsampled[ch] > 0
        combined = np.where(labelled & (encoded > combined), encoded, combined)

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
    original_shape = tuple(int(s) for s in arr.shape[1:])
    factor = compute_downsample_factor(original_shape, max_size)
    stacked = downsample_max_pool_czyx(arr, slice(0, n_ch), factor)
    rgb = normalize_rgb_stack(stacked)

    return VolumeBytesResult(
        data=rgb.tobytes(),
        shape=tuple(int(s) for s in rgb.shape[:3]),
        original_shape=original_shape,
        downsample_factor=factor,
        components=3,
    )
