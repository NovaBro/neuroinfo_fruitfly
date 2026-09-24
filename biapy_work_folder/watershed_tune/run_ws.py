"""Thin wrapper around BiaPy ``watershed_by_channels``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BIAPY_ROOT = _REPO_ROOT / "BiaPy-novabro"
if str(_BIAPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_BIAPY_ROOT))

from biapy.data.post_processing.post_processing import watershed_by_channels  # noqa: E402


def parse_thresh(value: str | float | int) -> float | str:
    """Parse a CLI / YAML threshold: ``\"auto\"`` or a float."""
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return float(value)
    s = str(value).strip().lower()
    if s == "auto":
        return "auto"
    return float(s)


def _as_thresh_list(
    values: list[Any] | Any,
    n: int,
    *,
    label: str,
) -> list[Any]:
    """Broadcast a single thresh to ``n`` channels or validate list length."""
    if not isinstance(values, list):
        values = [values]
    if len(values) == 1 and n > 1:
        values = list(values) * n
    if len(values) != n:
        raise ValueError(
            f"{label}: expected {n} threshold(s) for {n} channel(s), got {len(values)}"
        )
    return [parse_thresh(v) for v in values]


def run_watershed(
    data_zyxc: np.ndarray,
    instance_seg: dict[str, Any],
    *,
    seed_ths: list[Any] | Any,
    growth_ths: list[Any] | Any,
    verbose: bool = False,
) -> np.ndarray:
    """Run BiaPy watershed using YAML INSTANCE_SEG channel / WATERSHED fields.

    Parameters
    ----------
    data_zyxc
        Channel predictions ``(Z, Y, X, C)``.
    instance_seg
        ``PROBLEM.INSTANCE_SEG`` dict from the run YAML.
    seed_ths / growth_ths
        Per-seed / per-growth thresholds (float or ``\"auto\"``). A single value
        is broadcast to all channels in the corresponding list.
    """
    channels = list(instance_seg["DATA_CHANNELS"])
    ws = instance_seg["WATERSHED"]
    seed_channels = list(ws["SEED_CHANNELS"])
    growth_channels = list(ws["GROWTH_MASK_CHANNELS"])
    topo = ws["TOPOGRAPHIC_SURFACE_CHANNEL"]

    seed_channel_ths = _as_thresh_list(seed_ths, len(seed_channels), label="seed_ths")
    growth_mask_channel_ths = _as_thresh_list(
        growth_ths, len(growth_channels), label="growth_ths"
    )

    remove_before = bool(ws.get("DATA_REMOVE_SMALL_OBJ_BEFORE", False))
    thres_small_before = int(ws.get("DATA_REMOVE_SMALL_OBJ_BEFORE_SIZE", 10) or 10)
    seed_morph_sequence = list(ws.get("SEED_MORPH_SEQUENCE") or [])
    seed_morph_radius = list(ws.get("SEED_MORPH_RADIUS") or [])
    erode_and_dilate = bool(ws.get("ERODE_AND_DILATE_GROWTH_MASK", False))
    fore_erosion_radius = int(ws.get("FORE_EROSION_RADIUS", 5))
    fore_dilation_radius = int(ws.get("FORE_DILATION_RADIUS", 5))

    extra_opts = instance_seg.get("DATA_CHANNELS_EXTRA_OPTS") or []
    db_discretize = False
    if "Db" in channels and extra_opts:
        # BiaPy stores a list of per-channel option dicts; Db may be first entry.
        for opts in extra_opts:
            if isinstance(opts, dict) and "Db" in opts:
                db_discretize = opts["Db"].get("val_type", "norm") == "discretize"
                break

    labels = watershed_by_channels(
        data=data_zyxc,
        channels=channels,
        seed_channels=seed_channels,
        seed_channel_ths=seed_channel_ths,
        topo_surface_channel=topo,
        growth_mask_channels=growth_channels,
        growth_mask_channel_ths=growth_mask_channel_ths,
        remove_before=remove_before,
        thres_small_before=thres_small_before,
        seed_morph_sequence=seed_morph_sequence,
        seed_morph_radius=seed_morph_radius,
        erode_and_dilate_growth_mask=erode_and_dilate,
        fore_erosion_radius=fore_erosion_radius,
        fore_dilation_radius=fore_dilation_radius,
        verbose=verbose,
        db_discretize=db_discretize,
    )
    labels = np.asarray(labels)
    if labels.ndim == 4 and labels.shape[-1] == 1:
        labels = labels[..., 0]
    return labels
