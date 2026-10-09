"""Load per-sample PatchPerPix scoring metrics for the web viewer.

Two on-disk sources (already produced by PPP eval / predict jobs):

1. **instances** — per-sample TOMLs under the parallel evaluated param dir
   (primary)::

       {split}/evaluated/<ckpt>/<params...>/<dataset>/<stem>_vote_instances_cldice_greedy.toml

   matching ``{split}/instanced/<ckpt>/<params...>/<dataset>/``. Same
   ``[general]`` + ``[confusion_matrix.th_*]`` shape as BiaPy. Falls back to a
   matching row in ``summary.csv`` (sample column ``dataset/stem``) when the
   TOML is missing.
2. **numinst** — ``{stem}_pred_metrics.csv`` beside the processed zarr:
   ``{split}/processed/<ckpt>/<dataset>/<stem>_pred_metrics.csv``.

Both are reshaped into the same ``toml`` / ``csv`` payload the BiaPy metrics
endpoint and MetricsPanel already understand, so the client needs no PPP-only
rendering path. Instance scores also expose a ``leaderboard`` list with FISBe
website names (see ``services.fisbe_metrics``).
"""

from __future__ import annotations

import csv
import re
import tomllib
from pathlib import Path

from services import ppp_loader
from services.fisbe_metrics import leaderboard_from_toml
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
    """Map ``.../instanced/<ckpt>/<params...>`` → matching ``evaluated/...`` dir.

    Instanced HDFs nest under ``{dataset}/``; the evaluated *params* leaf is the
    parent of those dataset dirs (tomls live in ``evaluated/.../<dataset>/``).
    If ``instanced_root`` itself ends in a dataset name, strip it first.
    """
    from services.ppp_loader import _DATASET_DIRS

    parts = list(instanced_root.parts)
    try:
        idx = parts.index("instanced")
    except ValueError:
        return None
    rem = parts[idx + 1 :]
    if rem and rem[-1] in _DATASET_DIRS:
        rem = rem[:-1]
    evaluated = Path(*parts[:idx], "evaluated", *rem)
    return evaluated if evaluated.is_dir() else None


def _evaluated_summary_path(instanced_root: Path) -> Path | None:
    """Map ``.../instanced/<ckpt>/<params...>`` → matching ``evaluated/.../summary.csv``."""
    evaluated = _evaluated_dir(instanced_root)
    if evaluated is None:
        return None
    path = evaluated / "summary.csv"
    return path if path.is_file() else None


def _instances_toml_path(evaluated_dir: Path, stem: str) -> Path | None:
    """Locate ``{stem}_vote_instances*.toml`` (namespaced or flat).

    ``stem`` may be ``dataset/name`` (preferred) or a bare basename.
    """
    from services.ppp_loader import _DATASET_DIRS

    if "/" in stem:
        dataset, base = stem.split("/", 1)
        nested = evaluated_dir / dataset
        preferred = nested / f"{base}_vote_instances_cldice_greedy.toml"
        if preferred.is_file():
            return preferred
        matches = sorted(nested.glob(f"{base}_vote_instances*.toml"))
        if matches:
            return matches[0]
    preferred = evaluated_dir / f"{stem}_vote_instances_cldice_greedy.toml"
    if preferred.is_file():
        return preferred
    matches = sorted(evaluated_dir.glob(f"{stem}_vote_instances*.toml"))
    if matches:
        return matches[0]
    # Bare stem under any dataset subdir.
    base = stem.rsplit("/", 1)[-1]
    for dataset in sorted(_DATASET_DIRS):
        nested = evaluated_dir / dataset
        preferred = nested / f"{base}_vote_instances_cldice_greedy.toml"
        if preferred.is_file():
            return preferred
        matches = sorted(nested.glob(f"{base}_vote_instances*.toml"))
        if matches:
            return matches[0]
    return None


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


def _root_for_instances_stem(set_id: str, stem: str) -> Path | None:
    """Resolve the instanced leaf that holds ``stem`` for a (possibly merged) set."""
    try:
        kind, roots = ppp_loader._resolve_roots(set_id)
    except (FileNotFoundError, ValueError):
        return None
    if kind != "instances":
        return None
    return ppp_loader._root_for_stem(kind, roots, stem)


def _load_instances_toml(stem: str, set_id: str) -> dict | None:
    root = _root_for_instances_stem(set_id, stem)
    if root is None:
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
    root = _root_for_instances_stem(set_id, stem)
    if root is None:
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
        kind, roots = ppp_loader._resolve_roots(set_id)
    except (FileNotFoundError, ValueError):
        return None
    if kind != "numinst":
        return None
    root = ppp_loader._root_for_stem(kind, roots, stem)
    if root is None:
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
    """Return ``{toml, csv, leaderboard}`` for a PatchPerPix set.

    ``leaderboard`` uses FISBe website names (S · avF1 · C · …); see
    ``services.fisbe_metrics``. Missing sources are ``None`` / null values.
    """
    if not set_id or not ppp_loader.is_ppp_set(set_id):
        return {"toml": None, "csv": None, "leaderboard": leaderboard_from_toml(None)}
    if set_id.startswith(ppp_loader.INST_PREFIX):
        toml = _load_instances_metrics(stem, set_id)
        return {
            "toml": toml,
            "csv": None,
            "leaderboard": leaderboard_from_toml(toml),
        }
    return {
        "toml": None,
        "csv": _load_numinst_pred_metrics(stem, set_id),
        "leaderboard": leaderboard_from_toml(None),
    }


def _stem_from_vote_toml(path: Path) -> str | None:
    """``R22C03-…_vote_instances_cldice_greedy.toml`` → ``R22C03-…``."""
    name = path.name
    idx = name.find(_VOTE_TOML_SUFFIX)
    if idx <= 0:
        return None
    return name[:idx]


def _sample_id_from_vote_toml(path: Path, evaluated_root: Path) -> str | None:
    """Return ``dataset/stem`` when nested, else bare stem (legacy flat)."""
    from services.ppp_loader import _DATASET_DIRS

    base = _stem_from_vote_toml(path)
    if not base:
        return None
    parent = path.parent
    if parent.name in _DATASET_DIRS and parent.parent == evaluated_root:
        return f"{parent.name}/{base}"
    return base


def _iter_vote_tomls(evaluated: Path):
    """Yield vote-instances tomls under nested dataset dirs, then flat legacy."""
    from services.ppp_loader import _DATASET_DIRS

    for dataset in sorted(_DATASET_DIRS):
        nested = evaluated / dataset
        if nested.is_dir():
            yield from sorted(nested.glob("*_vote_instances*.toml"))
    yield from sorted(evaluated.glob("*_vote_instances*.toml"))


def load_ppp_instances_summary_table(set_id: str) -> dict[str, dict]:
    """Load per-sample metrics for the aggregate heatmap, keyed by sample ID.

    Keys are namespaced ``{dataset}/{stem}`` when tomls live under
    ``evaluated/.../<dataset>/``. Prefers TOMLs over ``summary.csv`` rows.
    Spans all roots of a merged val+test set; when an ID appears in more than
    one leaf, prefers the FISBe sample-list split, then TOML over CSV.
    """
    try:
        kind, roots = ppp_loader._resolve_roots(set_id)
    except (FileNotFoundError, ValueError):
        return {}
    if kind != "instances":
        return {}

    stem_split = ppp_loader._stem_to_fisbe_split()
    table: dict[str, dict] = {}
    # sample_id → (root_split, from_toml)
    provenance: dict[str, tuple[str | None, bool]] = {}

    def _fisbe_preferred(sample_id: str) -> str | None:
        base = sample_id.rsplit("/", 1)[-1]
        return stem_split.get(sample_id) or stem_split.get(base)

    def _take(sample_id: str, root: Path, from_toml: bool) -> bool:
        new_split = ppp_loader._split_of_root(root)
        if sample_id not in table:
            return True
        old_split, old_toml = provenance[sample_id]
        preferred = _fisbe_preferred(sample_id)
        if preferred is not None:
            old_ok = old_split == preferred
            new_ok = new_split == preferred
            if new_ok and not old_ok:
                return True
            if old_ok and not new_ok:
                return False
        if old_toml and not from_toml:
            return False
        if from_toml and not old_toml:
            return True
        return False

    for root in roots:
        evaluated = _evaluated_dir(root)
        if evaluated is not None:
            for path in _iter_vote_tomls(evaluated):
                sample_id = _sample_id_from_vote_toml(path, evaluated)
                if not sample_id or not _take(sample_id, root, True):
                    continue
                table[sample_id] = _parse_instances_toml(path)
                provenance[sample_id] = (ppp_loader._split_of_root(root), True)

        path = _evaluated_summary_path(root)
        if path is None:
            continue
        try:
            source = str(path.relative_to(ppp_loader._base()))
        except ValueError:
            source = str(path)
        with path.open(newline="") as fh:
            for record in csv.DictReader(fh, delimiter=";"):
                sample = (record.get("sample") or "").strip()
                if (
                    not sample
                    or sample in ("mean", "sum")
                    or not _take(sample, root, False)
                ):
                    continue
                table[sample] = _parse_summary_row(record, source)
                provenance[sample] = (ppp_loader._split_of_root(root), False)

    return table
