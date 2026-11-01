"""Per-sample predicted-instance size histograms for the Instance sizes tab.

Loads full-resolution label volumes for the selected prediction set, counts
voxels per instance id, and bins those sizes into shared log-spaced edges
(1 … 1e6). Samples are partitioned like Aggregate (fisbe_split × dataset).
Numinst sets are not applicable (no per-body IDs).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from services import biapy_loader, ppp_loader
from services.aggregate_metrics import (
    GROUP_KEYS,
    _group_for_stem,
    _lookup_by_namespaced,
    _lookup_unique_basename,
)
from services.local_stage import ensure_local
from services.predictions import stems_with_predictions

# Shared edges for every sample (25 edges → 24 bins).
_LOG_EDGES = np.logspace(0, 6, 25)


def _bin_edges_list() -> list[float]:
    return [float(x) for x in _LOG_EDGES]


def _histogram_sizes(sizes: np.ndarray) -> tuple[list[int], int]:
    """Bin instance sizes into ``_LOG_EDGES``; clip outliers into end bins."""
    if sizes.size == 0:
        return [0] * (len(_LOG_EDGES) - 1), 0
    clipped = np.clip(sizes.astype(np.float64), _LOG_EDGES[0], _LOG_EDGES[-1])
    hist, _ = np.histogram(clipped, bins=_LOG_EDGES)
    return [int(c) for c in hist], int(sizes.size)


def _load_biapy_labels(stem: str, set_id: str) -> np.ndarray | None:
    try:
        root = biapy_loader.biapy.resolve_prediction_set_root(set_id)
        tiff = biapy_loader.biapy.resolve_tiff_path(
            root / "per_image_instances", stem
        )
        return np.asarray(biapy_loader.biapy.load_zarr_tiff(ensure_local(tiff)))
    except (FileNotFoundError, ValueError, OSError):
        return None


def _load_ppp_instance_labels(stem: str, set_id: str) -> np.ndarray | None:
    try:
        kind, roots = ppp_loader._resolve_roots(set_id)
    except (FileNotFoundError, ValueError):
        return None
    if kind != "instances":
        return None
    root = ppp_loader._root_for_stem(kind, roots, stem)
    if root is None:
        return None
    if not ppp_loader._h5py_available():
        return None
    hdf = ppp_loader._inst_hdf(root, stem)
    if not hdf.is_file():
        return None
    import h5py

    try:
        with h5py.File(ensure_local(hdf), "r") as f:
            return np.asarray(f[ppp_loader._VOTE_DATASET])
    except (OSError, KeyError):
        return None


def _instance_sizes_from_labels(volume: np.ndarray) -> np.ndarray:
    ids, counts = np.unique(volume, return_counts=True)
    mask = ids != 0
    return counts[mask]


def _sample_histogram(
    stem: str, set_id: str, *, is_ppp: bool
) -> dict | None:
    if is_ppp:
        volume = _load_ppp_instance_labels(stem, set_id)
    else:
        volume = _load_biapy_labels(stem, set_id)
    if volume is None:
        return None
    sizes = _instance_sizes_from_labels(volume)
    counts, n_instances = _histogram_sizes(sizes)
    if n_instances == 0:
        return None
    return {
        "sample": stem.rsplit("/", 1)[-1],
        "counts": counts,
        "n_instances": n_instances,
    }


def _empty_payload(set_id: str | None, *, applicable: bool, reason: str | None) -> dict:
    return {
        "prediction_set": set_id,
        "applicable": applicable,
        "reason": reason,
        "bin_edges": _bin_edges_list(),
        "groups": {k: {"samples": []} for k in GROUP_KEYS},
        "groupOrder": list(GROUP_KEYS),
    }


@lru_cache(maxsize=8)
def get_instance_size_histograms(prediction_set: str | None) -> dict:
    """Build grouped per-sample size histograms for ``prediction_set``."""
    if not prediction_set:
        return _empty_payload(
            None, applicable=False, reason="Select a prediction set."
        )

    if ppp_loader.is_ppp_set(prediction_set):
        try:
            kind, _roots = ppp_loader._resolve_roots(prediction_set)
        except (FileNotFoundError, ValueError) as exc:
            return _empty_payload(
                prediction_set, applicable=False, reason=str(exc)
            )
        if kind == "numinst":
            return _empty_payload(
                prediction_set,
                applicable=False,
                reason="Numinst overlays have no per-instance body sizes.",
            )
        is_ppp = True
    else:
        is_ppp = False

    stems = sorted(stems_with_predictions(prediction_set))
    if not stems:
        return _empty_payload(
            prediction_set,
            applicable=True,
            reason="No predicted instances found for this set.",
        )

    by_namespaced = _lookup_by_namespaced()
    by_unique = _lookup_unique_basename()
    buckets: dict[str, list[dict]] = {k: [] for k in GROUP_KEYS}

    for stem in stems:
        mapped = _group_for_stem(stem, by_namespaced, by_unique)
        if mapped is None:
            continue
        group_key, row_label = mapped
        load_stem = stem if is_ppp else stem.rsplit("/", 1)[-1]
        hist = _sample_histogram(load_stem, prediction_set, is_ppp=is_ppp)
        if hist is None and is_ppp and "/" in stem:
            hist = _sample_histogram(
                stem.rsplit("/", 1)[-1], prediction_set, is_ppp=is_ppp
            )
        if hist is None:
            continue
        hist = {**hist, "sample": row_label}
        buckets[group_key].append(hist)

    for key in buckets:
        buckets[key].sort(key=lambda s: s["sample"])

    return {
        "prediction_set": prediction_set,
        "applicable": True,
        "reason": None,
        "bin_edges": _bin_edges_list(),
        "groups": {k: {"samples": buckets[k]} for k in GROUP_KEYS},
        "groupOrder": list(GROUP_KEYS),
    }
