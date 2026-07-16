"""Aggregate per-sample BiaPy scoring metrics into a samples x metrics matrix.

The per-sample metrics come from the same two sources ``services.metrics``
reads for a single sample (``test_results_metrics.csv`` and the per-stem
``tests/metrics/<stem>.zarr.toml``). Here they are collected across *every*
sample in the split list so the client can render one heatmap of
samples (rows) x metrics (columns).

Only a curated "core quality" set of metrics is exposed. The threshold-free
scalars come straight through; the threshold-dependent CSV metrics are flattened
at a caller-chosen detection threshold (default ``0.5``).

Alongside the matrix, each column is summarised by its mean and median across
samples (``summary``). Those are derived rows, not samples, so they are returned
separately from ``samples``/``values``.
"""

from __future__ import annotations

from statistics import fmean, median

from services.metrics import _csv_row_keys, _load_toml_metrics, load_csv_table
from services.biapy_loader import biapy
from services.sample_list import parse_sample_list

# Detection thresholds the CSV carries a full column group for.
THRESHOLD_CHOICES = ("0.3", "0.5", "0.75")
DEFAULT_THRESHOLD = "0.5"

# Curated heatmap columns. Each entry describes where the value comes from:
#   source="csv-scalar"   -> csv["scalars"][field]
#   source="csv-th"       -> csv["thresholds"][threshold][field]
#   source="toml-summary" -> toml["summary"][field]   (confusion_matrix avg)
#   source="toml-general" -> toml["general"][field]
# ``higher_is_better`` documents metric orientation for the client legend.
METRIC_SPECS: tuple[dict, ...] = (
    {"key": "iou_f", "label": "IoU (f channel)", "source": "csv-scalar",
     "field": "iou (f channel)", "higher_is_better": True},
    {"key": "iou_c", "label": "IoU (c channel)", "source": "csv-scalar",
     "field": "iou (c channel)", "higher_is_better": True},
    {"key": "avAP", "label": "avAP", "source": "toml-summary",
     "field": "avAP", "higher_is_better": True},
    {"key": "avFscore", "label": "avFscore", "source": "toml-summary",
     "field": "avFscore", "higher_is_better": True},
    {"key": "cldice_05", "label": "clDice @0.5", "source": "toml-general",
     "field": "avg_TP_05_cldice", "higher_is_better": True},
    {"key": "tp05_rel", "label": "TP@0.5 (rel)", "source": "toml-general",
     "field": "TP_05_rel", "higher_is_better": True},
    {"key": "precision", "label": "precision", "source": "csv-th",
     "field": "precision", "higher_is_better": True},
    {"key": "recall", "label": "recall", "source": "csv-th",
     "field": "recall", "higher_is_better": True},
    {"key": "f1", "label": "f1", "source": "csv-th",
     "field": "f1", "higher_is_better": True},
    {"key": "panoptic_quality", "label": "panoptic quality", "source": "csv-th",
     "field": "panoptic_quality", "higher_is_better": True},
)


def _as_number(value) -> float | None:
    """Coerce a parsed metric cell to a float, or ``None`` when not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract(spec: dict, csv_data: dict | None, toml_data: dict | None,
             threshold: str) -> float | None:
    field = spec["field"]
    source = spec["source"]
    if source == "csv-scalar":
        if not csv_data:
            return None
        return _as_number(csv_data.get("scalars", {}).get(field))
    if source == "csv-th":
        if not csv_data:
            return None
        return _as_number(csv_data.get("thresholds", {}).get(threshold, {}).get(field))
    if source == "toml-summary":
        if not toml_data:
            return None
        return _as_number(toml_data.get("summary", {}).get(field))
    if source == "toml-general":
        if not toml_data:
            return None
        return _as_number(toml_data.get("general", {}).get(field))
    return None


def _column_summary(samples: list[str],
                    values: dict[str, dict[str, float | None]]) -> dict:
    """Per-column mean/median across samples, over the cells that have a value.

    ``n`` reports how many samples actually contributed to each column, so a
    mean taken over a subset is never mistaken for one over every sample. A
    column with no values at all summarises to ``None``.
    """
    means: dict[str, float | None] = {}
    medians: dict[str, float | None] = {}
    counts: dict[str, int] = {}

    for spec in METRIC_SPECS:
        key = spec["key"]
        column = [
            v
            for s in samples
            if isinstance(v := values[s].get(key), (int, float))
            and not isinstance(v, bool)
        ]
        counts[key] = len(column)
        means[key] = fmean(column) if column else None
        medians[key] = median(column) if column else None

    return {"mean": means, "median": medians, "n": counts}


def get_aggregate_metrics(set_id: str | None = None,
                          threshold: str | None = None) -> dict:
    """Build the samples x metrics matrix for the given prediction set.

    Samples with no metric source at all (neither CSV row nor TOML) are omitted
    so empty rows never clutter the heatmap. Individual missing cells come back
    as ``None``.
    """
    threshold = threshold if threshold in THRESHOLD_CHOICES else DEFAULT_THRESHOLD

    try:
        result_root = biapy.resolve_prediction_set_root(set_id)
    except (FileNotFoundError, ValueError):
        result_root = None

    metrics_meta = [
        {"key": s["key"], "label": s["label"], "source": s["source"],
         "higherIsBetter": s["higher_is_better"]}
        for s in METRIC_SPECS
    ]

    samples: list[str] = []
    values: dict[str, dict[str, float | None]] = {}

    if result_root is not None:
        # Read each source once for the whole sweep: the CSV is parsed a single
        # time, and the metrics dir is listed once. Most sample-list stems have
        # no scores in a given set, so this also lets us skip them without any
        # per-stem filesystem probing.
        csv_table = load_csv_table(result_root)
        metrics_dir = result_root / "tests" / "metrics"
        toml_names = (
            {p.name for p in metrics_dir.iterdir() if p.suffix == ".toml"}
            if metrics_dir.is_dir()
            else set()
        )

        seen: set[str] = set()
        for entry in parse_sample_list():
            stem = entry.name
            if stem in seen:
                continue
            seen.add(stem)

            csv_data = next(
                (csv_table[k] for k in _csv_row_keys(stem) if k in csv_table), None
            )
            has_toml = any(n.startswith(stem) and n.endswith(".toml") for n in toml_names)
            if csv_data is None and not has_toml:
                continue

            toml_data = _load_toml_metrics(result_root, stem) if has_toml else None
            if csv_data is None and toml_data is None:
                continue

            values[stem] = {
                spec["key"]: _extract(spec, csv_data, toml_data, threshold)
                for spec in METRIC_SPECS
            }
            samples.append(stem)

    return {
        "prediction_set": set_id,
        "threshold": threshold,
        "thresholdChoices": list(THRESHOLD_CHOICES),
        "metrics": metrics_meta,
        # Real samples only — the client normalises each column against these,
        # so the derived summary rows below are deliberately kept out of it.
        "samples": samples,
        "values": values,
        "summary": _column_summary(samples, values),
    }
