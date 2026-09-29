"""Load per-sample PatchPerPix scoring metrics for the web viewer.

Two on-disk sources (already produced by PPP eval / predict jobs):

1. **instances** — per-sample TOMLs under the parallel evaluated param dir
   (primary)::

       {split}/evaluated/<ckpt>/<params...>/<stem>_vote_instances_cldice_greedy.toml

   matching ``{split}/instanced/<ckpt>/<params...>``. Same ``[general]`` +
   ``[confusion_matrix.th_*]`` shape as BiaPy. Falls back to a matching row in
   ``summary.csv`` when the TOML is missing (semicolon-separated columns like
   ``general Num GT``, ``th_0_5 fscore``).
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
import tomllib
from pathlib import Path

from services import ppp_loader
from services.metrics import _clean_value

_TH_COL = re.compile(r"^th_(\d+(?:_\d+)?)\s+(.+)$")
_GENERAL_COL = re.compile(r"^general\s+(.+)$")
_CONFUSION_COL = re.compile(r"^confusion_matrix\s+(.+)$")
_VOTE_TOML_SUFFIX = "_vote_instances"


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
    """``0_5`` / ``th_0_5`` / ``0_55`` → ``0.5`` / ``0.55``."""
    if raw.startswith("th_"):
        raw = raw[len("th_") :]
    return raw.replace("_", ".", 1)


def _evaluated_dir(instanced_root: Path) -> Path | None:
    """Map ``.../instanced/<ckpt>/<params...>`` → matching ``evaluated/...`` dir."""
    parts = list(instanced_root.parts)
    try:
        idx = parts.index("instanced")
    except ValueError:
        return None
    evaluated = Path(*parts[:idx], "evaluated", *parts[idx + 1 :])
    return evaluated if evaluated.is_dir() else None


def _evaluated_summary_path(instanced_root: Path) -> Path | None:
    """Map ``.../instanced/<ckpt>/<params...>`` → matching ``evaluated/.../summary.csv``."""
    evaluated = _evaluated_dir(instanced_root)
    if evaluated is None:
        return None
    path = evaluated / "summary.csv"
    return path if path.is_file() else None


def _instances_toml_path(evaluated_dir: Path, stem: str) -> Path | None:
    """Locate ``{stem}_vote_instances*.toml`` (prefer cldice_greedy)."""
    preferred = evaluated_dir / f"{stem}_vote_instances_cldice_greedy.toml"
    if preferred.is_file():
        return preferred
    matches = sorted(evaluated_dir.glob(f"{stem}_vote_instances*.toml"))
    return matches[0] if matches else None


def _parse_instances_toml(path: Path) -> dict:
    """Parse a PPP/BiaPy-shaped metrics TOML into the MetricsPanel payload."""
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    confusion = data.get("confusion_matrix", {})
    summary = {
        k: _clean_value(v) for k, v in confusion.items() if not isinstance(v, dict)
    }
    thresholds = {
        _threshold_label(k): _clean_value(v)
        for k, v in confusion.items()
        if isinstance(v, dict)
    }
    thresholds = dict(sorted(thresholds.items(), key=lambda kv: float(kv[0])))

    try:
        source = str(path.relative_to(ppp_loader._base()))
    except ValueError:
        source = str(path)

    return {
        "source": source,
        "general": _clean_value(data.get("general", {})),
        "summary": summary,
        "thresholds": thresholds,
    }


def _load_instances_toml(stem: str, set_id: str) -> dict | None:
    try:
        kind, root = ppp_loader._resolve_root(set_id)
    except (FileNotFoundError, ValueError):
        return None
    if kind != "instances":
        return None
    evaluated = _evaluated_dir(root)
    if evaluated is None:
        return None
    path = _instances_toml_path(evaluated, stem)
    if path is None:
        return None
    return _parse_instances_toml(path)


def _parse_summary_row(row: dict, source: str) -> dict:
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
    return {
        "source": source,
        "general": general,
        "summary": summary,
        "thresholds": thresholds,
    }


def _load_instances_summary_csv(stem: str, set_id: str) -> dict | None:
    """Fallback: one row from evaluated ``summary.csv``."""
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

    try:
        source = str(path.relative_to(ppp_loader._base()))
    except ValueError:
        source = str(path)
    return _parse_summary_row(row, source)


def _load_instances_metrics(stem: str, set_id: str) -> dict | None:
    """Prefer per-sample TOML; fall back to ``summary.csv`` row."""
    return _load_instances_toml(stem, set_id) or _load_instances_summary_csv(
        stem, set_id
    )


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
        return {"toml": _load_instances_metrics(stem, set_id), "csv": None}
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


def _stem_from_vote_toml(path: Path) -> str | None:
    """``R22C03-…_vote_instances_cldice_greedy.toml`` → ``R22C03-…``."""
    name = path.name
    idx = name.find(_VOTE_TOML_SUFFIX)
    if idx <= 0:
        return None
    return name[:idx]


def load_ppp_instances_summary_table(set_id: str) -> dict[str, dict]:
    """Load per-sample metrics for the aggregate heatmap, keyed by stem.

    Prefers ``*_vote_instances*.toml`` in the evaluated leaf (complete even when
    ``summary.csv`` was overwritten by a partial eval). Merges any CSV-only
    stems as a fallback.
    """
    try:
        kind, root = ppp_loader._resolve_root(set_id)
    except (FileNotFoundError, ValueError):
        return {}
    if kind != "instances":
        return {}

    table: dict[str, dict] = {}
    evaluated = _evaluated_dir(root)
    if evaluated is not None:
        for path in sorted(evaluated.glob("*_vote_instances*.toml")):
            stem = _stem_from_vote_toml(path)
            if not stem or stem in table:
                continue
            table[stem] = _parse_instances_toml(path)

    # CSV fallback for stems without a TOML (rare once eval wrote both).
    path = _evaluated_summary_path(root)
    if path is not None:
        try:
            source = str(path.relative_to(ppp_loader._base()))
        except ValueError:
            source = str(path)
        with path.open(newline="") as fh:
            for record in csv.DictReader(fh, delimiter=";"):
                sample = (record.get("sample") or "").strip()
                if not sample or sample in ("mean", "sum") or sample in table:
                    continue
                table[sample] = _parse_summary_row(record, source)

    return table
