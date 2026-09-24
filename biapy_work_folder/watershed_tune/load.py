"""Load channel predictions (ZYXC) and GT instance labels (ZYX)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from .channels import fisbe_sample_id


def _open_zarr_array(path: Path):
    """Open a Zarr v3 array directory (BiaPy per_image / label layout)."""
    return zarr.open(path.as_posix(), mode="r", zarr_format=3)


def load_channel_pred(per_image_dir: Path, stem: str) -> np.ndarray:
    """Load channel prediction volume as ``(Z, Y, X, C)`` float32."""
    path = Path(per_image_dir) / f"{stem}.zarr"
    if not path.is_dir():
        raise FileNotFoundError(f"Missing channel pred zarr: {path}")
    arr = np.asarray(_open_zarr_array(path), dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected ZYXC channel pred, got shape {arr.shape} for {path}")
    return arr


def load_gt_instances(gt_dir: Path, stem: str) -> np.ndarray:
    """Load instance-label GT as ``(Z, Y, X)`` (uint)."""
    sample = fisbe_sample_id(stem)
    path = Path(gt_dir) / f"{sample}.zarr"
    if not path.is_dir():
        raise FileNotFoundError(f"Missing GT volume for {stem!r}: {path}")
    arr = np.asarray(_open_zarr_array(path))
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D instance labels, got shape {arr.shape} for {path}")
    return arr


def channel_index(channel_names: list[str], name: str) -> int:
    """Index of ``name`` in YAML ``DATA_CHANNELS`` order."""
    try:
        return list(channel_names).index(name)
    except ValueError as exc:
        raise KeyError(
            f"Channel {name!r} not in DATA_CHANNELS {list(channel_names)}"
        ) from exc


def count_instances(labels: np.ndarray) -> int:
    """Count unique positive instance IDs in a label volume."""
    ids = np.unique(labels)
    return int(np.count_nonzero(ids > 0))
