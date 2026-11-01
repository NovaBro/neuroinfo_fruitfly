"""Aggregate per-sample scoring metrics into grouped samples x metrics matrices.

BiaPy sets use ``test_results_metrics.csv`` + TOML. PatchPerPix *instances*
sets use evaluated ``summary.csv`` / vote tomls (via ``ppp_metrics``). Numinst
sets have no aggregate matrix (voxel-level CSV only) and return empty groups.

Scored samples are partitioned into one table per ``(fisbe_split × dataset)``:

* Keys like ``test/completely``, ``val/completely``, ``test/synth_vh``.
* Sample row labels inside a table are **stem only** (dataset is in the key).
* Sample IDs on disk / in PPP metrics are namespaced ``{dataset}/{stem}``.
"""

from __future__ import annotations

from statistics import fmean, median

from services.metrics import _csv_row_keys, _load_toml_metrics, load_csv_table
from services.biapy_loader import biapy
from services import ppp_loader
from services.fisbe_metrics import FISBE_LEADERBOARD_SPECS, _as_number
from services.ppp_metrics import load_ppp_instances_summary_table
from services.sample_list import KNOWN_DATASETS, parse_sample_list

# Detection thresholds the CSV carries a full column group for.
THRESHOLD_CHOICES = ("0.3", "0.5", "0.75")
DEFAULT_THRESHOLD = "0.5"

_DATASETS = KNOWN_DATASETS
_SPLITS = ("test", "val")
# Stable table order: test×datasets then val×datasets.
GROUP_KEYS: tuple[str, ...] = tuple(
    f"{split}/{dataset}" for split in _SPLITS for dataset in _DATASETS
)

# Alias: Aggregate heatmap columns = FISBe website leaderboard names.
METRIC_SPECS: tuple[dict, ...] = FISBE_LEADERBOARD_SPECS
PPP_INSTANCE_METRIC_SPECS: tuple[dict, ...] = FISBE_LEADERBOARD_SPECS


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
    if source == "toml-th":
        if not toml_data:
            return None
        return _as_number(
            toml_data.get("thresholds", {}).get(threshold, {}).get(field)
        )
    return None


def _column_summary(
    samples: list[str],
    values: dict[str, dict[str, float | None]],
    specs: tuple[dict, ...],
) -> dict:
    """Per-column mean/median across samples, over the cells that have a value."""
    means: dict[str, float | None] = {}
    medians: dict[str, float | None] = {}
    counts: dict[str, int] = {}

    for spec in specs:
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


def _basename_stem(stem: str) -> str:
    """Strip optional ``{dataset}/`` prefix from list-mode sample IDs."""
    return stem.rsplit("/", 1)[-1]


def _dataset_of(stem: str) -> str | None:
    """Return dataset prefix when ``stem`` is ``{dataset}/{name}``."""
    if "/" not in stem:
        return None
    dataset, _ = stem.split("/", 1)
    return dataset if dataset in _DATASETS else None


def _lookup_by_namespaced() -> dict[str, tuple[str, str]]:
    """``{dataset}/{name}`` → (fisbe_split, dataset)."""
    out: dict[str, tuple[str, str]] = {}
    for entry in parse_sample_list():
        out[f"{entry.dataset}/{entry.name}"] = (entry.split, entry.dataset)
    return out


def _lookup_unique_basename() -> dict[str, tuple[str, str]]:
    """basename → (fisbe_split, dataset) only when the basename is unique."""
    counts: dict[str, int] = {}
    first: dict[str, tuple[str, str]] = {}
    for entry in parse_sample_list():
        counts[entry.name] = counts.get(entry.name, 0) + 1
        first.setdefault(entry.name, (entry.split, entry.dataset))
    return {name: first[name] for name, n in counts.items() if n == 1}


def _group_for_stem(
    stem: str,
    by_namespaced: dict[str, tuple[str, str]],
    by_unique_basename: dict[str, tuple[str, str]] | None = None,
) -> tuple[str, str] | None:
    """Return ``(group_key, row_label)`` or None to omit.

    ``group_key`` is ``{fisbe_split}/{dataset}``; ``row_label`` is stem only.
    """
    base = _basename_stem(stem)
    dataset = _dataset_of(stem)
    by_unique_basename = by_unique_basename or {}

    if dataset is not None:
        listed = by_namespaced.get(f"{dataset}/{base}")
        if listed is None:
            return None
        fisbe_split, dataset = listed
    else:
        listed = by_unique_basename.get(base)
        if listed is None:
            return None
        fisbe_split, dataset = listed

    if fisbe_split not in _SPLITS:
        return None
    if dataset not in _DATASETS:
        return None
    return f"{fisbe_split}/{dataset}", base


def _empty_group(specs: tuple[dict, ...]) -> dict:
    return {
        "samples": [],
        "values": {},
        "summary": _column_summary([], {}, specs),
    }


def _partition_aggregate(
    samples: list[str],
    values: dict[str, dict[str, float | None]],
    specs: tuple[dict, ...],
    by_namespaced: dict[str, tuple[str, str]] | None = None,
    by_unique_basename: dict[str, tuple[str, str]] | None = None,
) -> dict[str, dict]:
    """Split scored samples into split×dataset groups with summaries."""
    by_namespaced = (
        by_namespaced if by_namespaced is not None else _lookup_by_namespaced()
    )
    by_unique_basename = (
        by_unique_basename
        if by_unique_basename is not None
        else _lookup_unique_basename()
    )
    buckets: dict[str, list[str]] = {k: [] for k in GROUP_KEYS}
    # row_label → original key in ``values`` (first wins).
    value_key_for: dict[str, dict[str, str]] = {k: {} for k in GROUP_KEYS}

    for stem in samples:
        mapped = _group_for_stem(stem, by_namespaced, by_unique_basename)
        if mapped is None:
            continue
        group_key, row = mapped
        if row not in value_key_for[group_key]:
            value_key_for[group_key][row] = stem
            buckets[group_key].append(row)

    groups: dict[str, dict] = {}
    for key in GROUP_KEYS:
        group_samples = buckets[key]
        group_values = {
            s: values[value_key_for[key][s]] for s in group_samples
        }
        groups[key] = {
            "samples": group_samples,
            "values": group_values,
            "summary": _column_summary(group_samples, group_values, specs),
        }
    return groups


def _metrics_meta(specs: tuple[dict, ...]) -> list[dict]:
    return [
        {
            "key": s["key"],
            "label": s["label"],
            "source": s["source"],
            "higherIsBetter": s["higher_is_better"],
            "colorScale": s.get("color_scale", True),
        }
        for s in specs
    ]


def _empty_aggregate(set_id: str | None, threshold: str, specs: tuple[dict, ...]) -> dict:
    return {
        "prediction_set": set_id,
        "threshold": threshold,
        "thresholdChoices": list(THRESHOLD_CHOICES),
        "metrics": _metrics_meta(specs),
        "groups": {k: _empty_group(specs) for k in GROUP_KEYS},
        "groupOrder": list(GROUP_KEYS),
    }


def _payload(
    set_id: str | None,
    threshold: str,
    specs: tuple[dict, ...],
    samples: list[str],
    values: dict[str, dict[str, float | None]],
    by_namespaced: dict[str, tuple[str, str]] | None = None,
    by_unique_basename: dict[str, tuple[str, str]] | None = None,
) -> dict:
    return {
        "prediction_set": set_id,
        "threshold": threshold,
        "thresholdChoices": list(THRESHOLD_CHOICES),
        "metrics": _metrics_meta(specs),
        "groups": _partition_aggregate(
            samples, values, specs, by_namespaced, by_unique_basename
        ),
        "groupOrder": list(GROUP_KEYS),
    }


def _aggregate_ppp_instances(set_id: str, threshold: str) -> dict:
    specs = PPP_INSTANCE_METRIC_SPECS
    table = load_ppp_instances_summary_table(set_id)
    if not table:
        return _empty_aggregate(set_id, threshold, specs)

    by_namespaced = _lookup_by_namespaced()
    by_unique = _lookup_unique_basename()
    samples: list[str] = []
    values: dict[str, dict[str, float | None]] = {}
    for stem, toml_data in table.items():
        if _group_for_stem(stem, by_namespaced, by_unique) is None:
            continue
        values[stem] = {
            spec["key"]: _extract(spec, None, toml_data, threshold)
            for spec in specs
        }
        samples.append(stem)

    return _payload(
        set_id, threshold, specs, samples, values, by_namespaced, by_unique
    )


def get_aggregate_metrics(set_id: str | None = None,
                          threshold: str | None = None) -> dict:
    """Build grouped samples x metrics matrices for the given prediction set.

    Samples with no metric source at all (neither CSV row nor TOML) are omitted.
    Individual missing cells come back as ``None``. Response ``groups`` always
    includes every ``{test,val}/{completely,partly,synth_vh,…}`` key; empty
    tables have zero samples. ``groupOrder`` lists keys for the client.
    """
    threshold = threshold if threshold in THRESHOLD_CHOICES else DEFAULT_THRESHOLD

    if ppp_loader.is_ppp_set(set_id):
        if set_id and set_id.startswith(ppp_loader.INST_PREFIX):
            return _aggregate_ppp_instances(set_id, threshold)
        # numinst: voxel-level CSV only — no instance aggregate matrix.
        return _empty_aggregate(set_id, threshold, PPP_INSTANCE_METRIC_SPECS)

    try:
        result_root = biapy.resolve_prediction_set_root(set_id)
    except (FileNotFoundError, ValueError):
        result_root = None

    samples: list[str] = []
    values: dict[str, dict[str, float | None]] = {}
    by_namespaced = _lookup_by_namespaced()
    by_unique = _lookup_unique_basename()

    if result_root is not None:
        csv_table = load_csv_table(result_root)
        metrics_dir = result_root / "tests" / "metrics"
        toml_names = (
            {p.name for p in metrics_dir.iterdir() if p.suffix == ".toml"}
            if metrics_dir.is_dir()
            else set()
        )

        seen: set[str] = set()
        for entry in parse_sample_list():
            # BiaPy rows are flat stems; namespace for grouping.
            stem = f"{entry.dataset}/{entry.name}"
            if stem in seen:
                continue
            seen.add(stem)
            if _group_for_stem(stem, by_namespaced, by_unique) is None:
                continue

            csv_data = next(
                (csv_table[k] for k in _csv_row_keys(entry.name) if k in csv_table),
                None,
            )
            has_toml = any(
                n.startswith(entry.name) and n.endswith(".toml") for n in toml_names
            )
            if csv_data is None and not has_toml:
                continue

            toml_data = (
                _load_toml_metrics(result_root, entry.name) if has_toml else None
            )
            if csv_data is None and toml_data is None:
                continue

            values[stem] = {
                spec["key"]: _extract(spec, csv_data, toml_data, threshold)
                for spec in METRIC_SPECS
            }
            samples.append(stem)

    return _payload(
        set_id, threshold, METRIC_SPECS, samples, values, by_namespaced, by_unique
    )
