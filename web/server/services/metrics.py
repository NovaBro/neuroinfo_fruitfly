"""Load per-sample BiaPy scoring metrics for the web viewer.

Two sources live under each prediction set's result root:

1. ``tests/metrics/<stem>.zarr.toml`` — the BiaPy instance-segmentation
   metrics TOML (``[general]`` + per-threshold ``[confusion_matrix.*]``).
2. ``test_results_metrics.csv`` — a one-row-per-sample CSV keyed by the
   ``<stem>.zarr.tiff`` filename, with IoU and per-threshold columns.

Both are grouped by detection threshold here so the client can render them as
compact tables.
"""

from __future__ import annotations

import csv
import re
import tomllib
from pathlib import Path

from services.biapy_loader import biapy  # reuse the biapy scripts module

# BiaPy writes some metrics as numpy ``repr`` strings, e.g.
# ``"np.float64(0.0014492753623188404)"`` or ``"np.int32(3)"``. Unwrap the
# inner literal back to a plain number so the client never sees the dtype.
_NUMPY_REPR = re.compile(r"^np\.\w+\((.*)\)$")


def _clean_value(value):
    """Recursively strip numpy ``repr`` wrappers from a parsed-TOML value."""
    if isinstance(value, str):
        match = _NUMPY_REPR.match(value.strip())
        if match:
            return _num(match.group(1))
        return value
    if isinstance(value, list):
        return [_clean_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean_value(v) for k, v in value.items()}
    return value


def _metrics_toml_path(result_root: Path, stem: str) -> Path | None:
    """Locate the ``tests/metrics`` TOML for ``stem`` (prefer ``<stem>.zarr.toml``)."""
    metrics_dir = result_root / "tests" / "metrics"
    for candidate in (f"{stem}.zarr.toml", f"{stem}.toml"):
        path = metrics_dir / candidate
        if path.is_file():
            return path
    # Fall back to any TOML that starts with the stem (e.g. the cldice variant).
    if metrics_dir.is_dir():
        for path in sorted(metrics_dir.glob(f"{stem}*.toml")):
            return path
    return None


def _threshold_label(key: str) -> str:
    """``th_0_55`` -> ``0.55`` (BiaPy replaces the decimal point with ``_``)."""
    if key.startswith("th_"):
        key = key[len("th_") :]
    return key.replace("_", ".", 1)


def _load_toml_metrics(result_root: Path, stem: str) -> dict | None:
    path = _metrics_toml_path(result_root, stem)
    if path is None:
        return None
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

    return {
        "source": str(path.relative_to(result_root)),
        "general": _clean_value(data.get("general", {})),
        "summary": summary,
        "thresholds": thresholds,
    }


def _num(value: str):
    """Parse a CSV cell to int/float when possible, else return the raw string."""
    if value == "":
        return None
    try:
        if value.lstrip("-").isdigit():
            return int(value)
        return float(value)
    except ValueError:
        return value


def _load_csv_metrics(result_root: Path, stem: str) -> dict | None:
    path = result_root / "test_results_metrics.csv"
    if not path.is_file():
        return None

    wanted = {f"{stem}.zarr.tiff", f"{stem}.zarr.tif", f"{stem}.zarr"}
    row: dict[str, str] | None = None
    with path.open(newline="") as fh:
        for record in csv.DictReader(fh):
            if (record.get("file") or "").strip() in wanted:
                row = record
                break
    if row is None:
        return None

    scalars: dict[str, object] = {}
    thresholds: dict[str, dict[str, object]] = {}
    for col, val in row.items():
        if col == "file" or col is None:
            continue
        if " TH " in col:
            th, _, metric = col.partition(" TH ")
            thresholds.setdefault(th.strip(), {})[metric.strip()] = _num(val)
        else:
            scalars[col] = _num(val)
    thresholds = dict(sorted(thresholds.items(), key=lambda kv: float(kv[0])))

    return {
        "source": "test_results_metrics.csv",
        "file": row.get("file"),
        "scalars": scalars,
        "thresholds": thresholds,
    }


def get_sample_metrics(stem: str, set_id: str | None = None) -> dict:
    """Return both metric sources for ``stem`` in the given prediction set.

    Missing sources come back as ``None`` (rather than raising) so the client
    can render whatever is available.
    """
    try:
        result_root = biapy.resolve_prediction_set_root(set_id)
    except (FileNotFoundError, ValueError):
        return {"toml": None, "csv": None}
    return {
        "toml": _load_toml_metrics(result_root, stem),
        "csv": _load_csv_metrics(result_root, stem),
    }
