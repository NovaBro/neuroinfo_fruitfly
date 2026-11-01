"""Unified prediction-set dispatcher over multiple model sources.

The web viewer overlays a sample's *predicted* instances from a selected
"prediction set". Sets come from two sources:

* **BiaPy** — run dirs under ``BIAPY_RESULTS_BASES`` (``services/biapy_loader``;
  default ``biapy_work_folder/results`` + ``metrics/biapy``).
* **PatchPerPix** — experiment dirs under ``PPP_EXPERIMENTS_BASE``
  (``services/ppp_loader``), exposing ``numinst`` and ``instances`` overlays.

Each set carries a ``source`` (``"biapy"`` | ``"ppp"``) and a globally-unique
``id``. BiaPy ids are repo-relative paths (so ``biapy_work_folder/results/...`` and
``metrics/biapy/...`` stay distinct; bare paths relative to
``BIAPY_RESULTS_BASE`` still resolve for backward compatibility). The
PatchPerPix loader prefixes its ids (``ppp-numinst:`` / ``ppp-inst:``). Routing
is by that prefix — this module is the single interface ``main.py`` imports, so
the individual endpoints stay source-agnostic.
"""

from __future__ import annotations

from services import biapy_loader, ppp_loader
from services.ttl_cache import ttl_cache
from services.volume_pipeline import VolumeBytesResult


def _bare_stem(stem: str) -> str:
    return stem.rsplit("/", 1)[-1]


def list_prediction_sets() -> list[dict]:
    """All prediction sets from every source, each tagged with ``source``."""
    biapy_sets = [
        {**s, "source": "biapy", "kind": "instances"}
        for s in biapy_loader.list_prediction_sets()
    ]
    return biapy_sets + ppp_loader.discover_prediction_sets()


def stems_with_predictions_any() -> set[str]:
    """Stems with predicted output in *any* set across all sources."""
    return (
        biapy_loader.stems_with_predictions_any()
        | ppp_loader.stems_with_predictions_any()
    )


@ttl_cache(60)
def stems_with_predictions(set_id: str | None = None) -> set[str]:
    """Stems that have predicted output in ``set_id``, or any set if unset.

    PatchPerPix list-mode stems are namespaced ``{dataset}/{name}``; BiaPy and
    flat PPP dumps use bare names.
    """
    if not set_id:
        return stems_with_predictions_any()
    if ppp_loader.is_ppp_set(set_id):
        return ppp_loader.stems_for_set(set_id)
    try:
        root = biapy_loader.biapy.resolve_prediction_set_root(set_id)
    except (FileNotFoundError, ValueError):
        return set()
    inst_dir = root / "per_image_instances"
    if not inst_dir.is_dir():
        return set()
    return {biapy_loader.biapy.sample_stem(p) for p in inst_dir.glob("*.tif*")}


def sample_has_prediction(
    name: str,
    dataset: str,
    set_id: str | None = None,
) -> bool:
    """True when ``(dataset, name)`` has predicted output in the given set.

    Prefers namespaced ``dataset/name``. A bare-stem hit counts only when no
    other dataset's namespaced form for the same basename is present (avoids
    synth_vh vs synth_vh_u8 collisions).
    """
    predicted = stems_with_predictions(set_id)
    namespaced = f"{dataset}/{name}"
    if namespaced in predicted:
        return True
    if name not in predicted:
        return False
    for stem in predicted:
        if "/" not in stem:
            continue
        other_ds, other_name = stem.split("/", 1)
        if other_name == name and other_ds != dataset:
            return False
    return True


def has_predicted_instances(stem: str, set_id: str | None = None) -> bool:
    """True when ``stem`` has predicted output in the given set (or any set)."""
    if "/" in stem:
        dataset, name = stem.split("/", 1)
        return sample_has_prediction(name, dataset, set_id)
    if not set_id:
        return stem in stems_with_predictions_any()
    if ppp_loader.is_ppp_set(set_id):
        return ppp_loader.has_predicted_instances(stem, set_id)
    return biapy_loader.has_predicted_instances(stem, set_id)


def get_predicted_instances_meta(
    stem: str, set_id: str | None = None
) -> dict | None:
    """Predicted-instances metadata for ``stem`` in the given set."""
    if ppp_loader.is_ppp_set(set_id):
        return ppp_loader.get_predicted_instances_meta(stem, set_id)
    return biapy_loader.get_predicted_instances_meta(_bare_stem(stem), set_id)


def predicted_instances_to_bytes(
    stem: str,
    max_size: int,
    set_id: str | None = None,
    min_size: int = 0,
) -> VolumeBytesResult:
    """Load and downsample the selected set's predicted overlay for ``stem``."""
    if ppp_loader.is_ppp_set(set_id):
        return ppp_loader.predicted_instances_to_bytes(
            stem, max_size, set_id, min_size
        )
    return biapy_loader.predicted_instances_to_bytes(
        _bare_stem(stem), max_size, set_id, min_size
    )
