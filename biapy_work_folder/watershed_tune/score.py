"""Multi-metric instance matching + composite score for watershed_tune."""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BIAPY_ROOT = _REPO_ROOT / "BiaPy-novabro"
if str(_BIAPY_ROOT) not in sys.path:
    sys.path.insert(0, str(_BIAPY_ROOT))

from biapy.utils.matching import matching, wrapper_matching_dataset_lazy  # noqa: E402

IOU_THRESHOLDS: tuple[float, ...] = (0.3, 0.5, 0.75)

# Fixed keys for Ray / CSV / CLI (Stage 2 schema).
METRIC_KEYS: tuple[str, ...] = (
    "pq_0.3",
    "pq_0.5",
    "pq_0.75",
    "f1_0.3",
    "f1_0.5",
    "f1_0.75",
    "precision_0.3",
    "recall_0.3",
    "precision_0.5",
    "recall_0.5",
    "mean_matched_0.3",
    "mean_matched_0.5",
    "n_pred",
    "n_true",
    "mean_n_pred",
    "mean_n_true",
    "count_penalty",
    "score",
)


@dataclass(frozen=True)
class MetricWeights:
    """Weights for the Stage-2 composite ``score`` (maximize)."""

    w_pq03: float = 1.0
    w_pq05: float = 0.5
    w_f103: float = 0.5
    w_f105: float = 0.25
    w_count: float = 0.25


DEFAULT_WEIGHTS = MetricWeights()

_WEIGHT_ALIASES = {
    "pq03": "w_pq03",
    "pq05": "w_pq05",
    "f103": "w_f103",
    "f105": "w_f105",
    "count": "w_count",
    "w_pq03": "w_pq03",
    "w_pq05": "w_pq05",
    "w_f103": "w_f103",
    "w_f105": "w_f105",
    "w_count": "w_count",
}


def parse_metric_weights(spec: str | None, base: MetricWeights = DEFAULT_WEIGHTS) -> MetricWeights:
    """Parse ``pq03=1,pq05=0.5,...`` into ``MetricWeights`` (unlisted keys keep defaults)."""
    if not spec or not str(spec).strip():
        return base
    vals = asdict(base)
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad weight token {part!r}; expected key=value")
        key, raw = part.split("=", 1)
        key = key.strip().lower()
        if key not in _WEIGHT_ALIASES:
            raise ValueError(
                f"Unknown weight key {key!r}; expected one of {sorted(_WEIGHT_ALIASES)}"
            )
        vals[_WEIGHT_ALIASES[key]] = float(raw.strip())
    return MetricWeights(**vals)


def count_penalty(n_pred: float, n_true: float, eps: float = 1e-6) -> float:
    """Soft count regularizer: ``|log((n_pred+ε)/(n_true+ε))|``."""
    return float(abs(math.log((float(n_pred) + eps) / (float(n_true) + eps))))


def composite_score(
    metrics: Mapping[str, Any],
    weights: MetricWeights = DEFAULT_WEIGHTS,
) -> float:
    """Weighted composite (maximize) from a metric dict with schema keys."""
    return float(
        weights.w_pq03 * float(metrics["pq_0.3"])
        + weights.w_pq05 * float(metrics["pq_0.5"])
        + weights.w_f103 * float(metrics["f1_0.3"])
        + weights.w_f105 * float(metrics["f1_0.5"])
        - weights.w_count * float(metrics["count_penalty"])
    )


def score_volume(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict, ...]:
    """Per-volume ``matching`` at all IoU thresholds (tuple of dicts, one per thresh)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    result = matching(y_true, y_pred, thresh=list(IOU_THRESHOLDS), criterion="iou")
    if isinstance(result, dict):
        return (result,)
    return tuple(result)


def _stats_to_mapping(stat: Any) -> dict[str, Any]:
    if isinstance(stat, Mapping):
        return dict(stat)
    if hasattr(stat, "_asdict"):
        return dict(stat._asdict())
    return {k: getattr(stat, k) for k in METRIC_KEYS if hasattr(stat, k)}


def _by_thresh(stats_tuple: Sequence[Any]) -> dict[float, dict[str, Any]]:
    out: dict[float, dict[str, Any]] = {}
    for s in stats_tuple:
        m = _stats_to_mapping(s)
        out[float(m["thresh"])] = m
    return out


def aggregate_volume_stats(
    stats_list: Sequence[tuple[dict, ...]],
    *,
    n_images: int | None = None,
) -> dict[str, Any]:
    """Aggregate per-volume matching tuples into the Stage-2 metric schema (no score yet).

    Quality metrics use BiaPy ``by_image=True`` (macro-average). ``n_pred`` / ``n_true``
    are summed across volumes; ``mean_n_*`` are per-volume means.
    """
    if not stats_list:
        raise ValueError("stats_list is empty")
    n_images = n_images if n_images is not None else len(stats_list)
    if n_images <= 0:
        raise ValueError("n_images must be positive")

    threshs = list(IOU_THRESHOLDS)
    agg = wrapper_matching_dataset_lazy(
        list(stats_list),
        thresh=threshs,
        criterion="iou",
        by_image=True,
    )
    if isinstance(agg, tuple):
        by_th = _by_thresh(agg)
    else:
        by_th = _by_thresh((agg,))

    s03 = by_th[0.3]
    s05 = by_th[0.5]
    s075 = by_th[0.75]

    n_pred = int(s03["n_pred"])
    n_true = int(s03["n_true"])
    metrics: dict[str, Any] = {
        "pq_0.3": float(s03["panoptic_quality"]),
        "pq_0.5": float(s05["panoptic_quality"]),
        "pq_0.75": float(s075["panoptic_quality"]),
        "f1_0.3": float(s03["f1"]),
        "f1_0.5": float(s05["f1"]),
        "f1_0.75": float(s075["f1"]),
        "precision_0.3": float(s03["precision"]),
        "recall_0.3": float(s03["recall"]),
        "precision_0.5": float(s05["precision"]),
        "recall_0.5": float(s05["recall"]),
        "mean_matched_0.3": float(s03["mean_matched_score"]),
        "mean_matched_0.5": float(s05["mean_matched_score"]),
        "n_pred": n_pred,
        "n_true": n_true,
        "mean_n_pred": float(n_pred) / float(n_images),
        "mean_n_true": float(n_true) / float(n_images),
        "count_penalty": count_penalty(n_pred, n_true),
    }
    return metrics


def finalize_metrics(
    metrics: Mapping[str, Any],
    weights: MetricWeights = DEFAULT_WEIGHTS,
) -> dict[str, Any]:
    """Attach ``score`` and return a dict with all ``METRIC_KEYS``."""
    out = {k: metrics[k] for k in METRIC_KEYS if k != "score"}
    # Ensure count_penalty present if caller only passed raw quality fields.
    if "count_penalty" not in out:
        out["count_penalty"] = count_penalty(out["n_pred"], out["n_true"])
    out["score"] = composite_score(out, weights)
    missing = [k for k in METRIC_KEYS if k not in out]
    if missing:
        raise KeyError(f"Metric schema missing keys: {missing}")
    return out


def evaluate_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: MetricWeights = DEFAULT_WEIGHTS,
) -> dict[str, Any]:
    """Full schema for a single volume (treated as a 1-image dataset)."""
    vol_stats = score_volume(y_true, y_pred)
    return finalize_metrics(aggregate_volume_stats([vol_stats], n_images=1), weights)


def evaluate_dataset(
    stats_list: Sequence[tuple[dict, ...]],
    weights: MetricWeights = DEFAULT_WEIGHTS,
) -> dict[str, Any]:
    """Full schema over a list of per-volume ``score_volume`` results."""
    return finalize_metrics(aggregate_volume_stats(stats_list), weights)
