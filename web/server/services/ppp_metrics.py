"""Load per-sample PatchPerPix scoring metrics for the web viewer.

Two on-disk sources (already produced by PPP eval / predict jobs):

1. **instances** — ``summary.csv`` under the parallel evaluated param dir:
   ``{split}/evaluated/<ckpt>/<params...>/summary.csv`` matching the
   ``{split}/instanced/<ckpt>/<params...>`` prediction-set root. Semicolon-
   separated; columns like ``general Num GT``, ``th_0_5 fscore``.
2. **numinst** — ``{stem}_pred_metrics.csv`` beside the processed zarr:
   ``{split}/processed/<ckpt>/<stem>_pred_metrics.csv``. Comma-separated
   rows of ``metric,thresh,f1,precision,recall,...``.

Both are reshaped into the same ``toml`` / ``csv`` payload the BiaPy metrics
endpoint and MetricsPanel already understand, so the client needs no PPP-only
rendering path.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from services import ppp_loader

_TH_COL = re.compile(r"^th_(\d+(?:_\d+)?)\s+(.+)$")
_GENERAL_COL = re.compile(r"^general\s+(.+)$")
_CONFUSION_COL = re.compile(r"^confusion_matrix\s+(.+)$")


def _num(value: str):
    if value is None or value == "":
        return None
    try:
        if value.lstrip("-").isdigit():
            return int(value)
        return float(value)
    except ValueError:
        return value


def _threshold_label(raw: str) -> str:
    """``0_5`` / ``0_55`` → ``0.5`` / ``0.55``."""
    return raw.replace("_", ".", 1)


def _evaluated_summary_path(instanced_root: Path) -> Path | None:
    """Map ``.../instanced/<ckpt>/<params...>`` → matching ``evaluated/.../summary.csv``."""
    parts = list(instanced_root.parts)
    try:
        idx = parts.index("instanced")
    except ValueError:
        return None
    evaluated = Path(*parts[:idx], "evaluated", *parts[idx + 1 :]) / "summary.csv"
    return evaluated if evaluated.is_file() else None


def _load_instances_summary(stem: str, set_id: str) -> dict | None:
    try:
        kind, root = ppp_loader._resolve_root(set_id)
    except (FileNotFoundError, ValueError):
        return None
    if kind != "instances":
        return None
    path = _evaluated_summary_path(root)
    if path is None:
        return None

    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        row = None
        for record in reader:
            sample = (record.get("sample") or "").strip()
            if sample == stem:
                row = record
                break
        if row is None:
            return None

    general: dict[str, object] = {}
    summary: dict[str, object] = {}
    thresholds: dict[str, dict[str, object]] = {}
    for col, raw in row.items():
        if col is None or col == "sample":
            continue
        val = _num(raw)
        if (m := _GENERAL_COL.match(col)):
            general[m.group(1)] = val
        elif (m := _CONFUSION_COL.match(col)):
            summary[m.group(1)] = val
        elif (m := _TH_COL.match(col)):
            th = _threshold_label(m.group(1))
            thresholds.setdefault(th, {})[m.group(2)] = val
        else:
            summary[col] = val

    thresholds = dict(sorted(thresholds.items(), key=lambda kv: float(kv[0])))
    try:
        source = str(path.relative_to(ppp_loader._base()))
    except ValueError:
        source = str(path)

    return {
        "source": source,
        "general": general,
        "summary": summary,
        "thresholds": thresholds,
    }


def _load_numinst_pred_metrics(stem: str, set_id: str) -> dict | None:
    try:
        kind, root = ppp_loader._resolve_root(set_id)
    except (FileNotFoundError, ValueError):
        return None
    if kind != "numinst":
        return None
    path = root / f"{stem}_pred_metrics.csv"
    if not path.is_file():
        return None

    scalars: dict[str, object] = {}
    thresholds: dict[str, dict[str, object]] = {}
    with path.open(newline="") as fh:
        for record in csv.DictReader(fh):
            metric = (record.get("metric") or "").strip()
            if not metric:
                continue
            thresh_raw = (record.get("thresh") or "").strip()
            metrics = {
                k: _num(v)
                for k, v in record.items()
                if k not in ("metric", "thresh")
            }
            if thresh_raw:
                th = thresh_raw
                bucket = thresholds.setdefault(th, {})
                for k, v in metrics.items():
                    bucket[f"{metric}_{k}"] = v
            else:
                for k, v in metrics.items():
                    scalars[f"{metric}_{k}"] = v

    thresholds = dict(sorted(thresholds.items(), key=lambda kv: float(kv[0])))
    try:
        source = str(path.relative_to(ppp_loader._base()))
    except ValueError:
        source = str(path)

    return {
        "source": source,
        "file": f"{stem}_pred_metrics.csv",
        "scalars": scalars,
        "thresholds": thresholds,
    }


def get_ppp_sample_metrics(stem: str, set_id: str | None) -> dict:
    """Return ``{toml, csv}`` for a PatchPerPix set (missing → ``None``)."""
    if not set_id or not ppp_loader.is_ppp_set(set_id):
        return {"toml": None, "csv": None}
    if set_id.startswith(ppp_loader.INST_PREFIX):
        return {"toml": _load_instances_summary(stem, set_id), "csv": None}
    return {"toml": None, "csv": _load_numinst_pred_metrics(stem, set_id)}


# Aggregate heatmap columns for PPP instances summary.csv.
PPP_INSTANCE_METRIC_SPECS: tuple[dict, ...] = (
    {
        "key": "num_gt",
        "label": "Num GT",
        "source": "toml-general",
        "field": "Num GT",
        "higher_is_better": True,
    },
    {
        "key": "num_pred",
        "label": "Num Pred",
        "source": "toml-general",
        "field": "Num Pred",
        "higher_is_better": False,
    },
    {
        "key": "avg_f1_cov",
        "label": "avg F1 cov",
        "source": "toml-general",
        "field": "avg_f1_cov_score",
        "higher_is_better": True,
    },
    {
        "key": "avFscore",
        "label": "avFscore",
        "source": "toml-summary",
        "field": "avFscore",
        "higher_is_better": True,
    },
    {
        "key": "gt_skel_cov",
        "label": "GT skel cov",
        "source": "toml-general",
        "field": "avg_gt_skel_coverage",
        "higher_is_better": True,
    },
    {
        "key": "fscore",
        "label": "fscore",
        "source": "toml-th",
        "field": "fscore",
        "higher_is_better": True,
    },
)


def load_ppp_instances_summary_table(set_id: str) -> dict[str, dict]:
    """Parse evaluated ``summary.csv`` once, keyed by sample stem.

    Used by the aggregate endpoint so multi-sample heatmaps do not re-read the
    CSV per stem.
    """
    try:
        kind, root = ppp_loader._resolve_root(set_id)
    except (FileNotFoundError, ValueError):
        return {}
    if kind != "instances":
        return {}
    path = _evaluated_summary_path(root)
    if path is None:
        return {}

    table: dict[str, dict] = {}
    with path.open(newline="") as fh:
        for record in csv.DictReader(fh, delimiter=";"):
            sample = (record.get("sample") or "").strip()
            if not sample or sample in ("mean", "sum"):
                continue
            # Reuse the single-row parser by reconstructing via get path.
            # Inline parse to avoid a second file open:
            general: dict[str, object] = {}
            summary: dict[str, object] = {}
            thresholds: dict[str, dict[str, object]] = {}
            for col, raw in record.items():
                if col is None or col == "sample":
                    continue
                val = _num(raw)
                if (m := _GENERAL_COL.match(col)):
                    general[m.group(1)] = val
                elif (m := _CONFUSION_COL.match(col)):
                    summary[m.group(1)] = val
                elif (m := _TH_COL.match(col)):
                    th = _threshold_label(m.group(1))
                    thresholds.setdefault(th, {})[m.group(2)] = val
                else:
                    summary[col] = val
            thresholds = dict(
                sorted(thresholds.items(), key=lambda kv: float(kv[0]))
            )
            try:
                source = str(path.relative_to(ppp_loader._base()))
            except ValueError:
                source = str(path)
            table[sample] = {
                "source": source,
                "general": general,
                "summary": summary,
                "thresholds": thresholds,
            }
    return table
