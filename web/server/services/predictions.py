"""Unified prediction-set dispatcher over multiple model sources.

The web viewer overlays a sample's *predicted* instances from a selected
"prediction set". Sets come from two sources:

* **BiaPy** — run dirs under ``BIAPY_RESULTS_BASE`` (``services/biapy_loader``).
* **PatchPerPix** — experiment dirs under ``PPP_EXPERIMENTS_BASE``
  (``services/ppp_loader``), exposing ``numinst`` and ``instances`` overlays.

Each set carries a ``source`` (``"biapy"`` | ``"ppp"``) and a globally-unique
``id``. BiaPy ids stay bare (relative paths, for backward compatibility); the
PatchPerPix loader prefixes its ids (``ppp-numinst:`` / ``ppp-inst:``). Routing
is by that prefix — this module is the single interface ``main.py`` imports, so
the individual endpoints stay source-agnostic.
"""

from __future__ import annotations

from services import biapy_loader, ppp_loader
from services.volume_pipeline import VolumeBytesResult


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


def get_predicted_instances_meta(
    stem: str, set_id: str | None = None
) -> dict | None:
    """Predicted-instances metadata for ``stem`` in the given set."""
    if ppp_loader.is_ppp_set(set_id):
        return ppp_loader.get_predicted_instances_meta(stem, set_id)
    return biapy_loader.get_predicted_instances_meta(stem, set_id)


def predicted_instances_to_bytes(
    stem: str, max_size: int, set_id: str | None = None
) -> VolumeBytesResult:
    """Load and downsample the selected set's predicted overlay for ``stem``."""
    if ppp_loader.is_ppp_set(set_id):
        return ppp_loader.predicted_instances_to_bytes(stem, max_size, set_id)
    return biapy_loader.predicted_instances_to_bytes(stem, max_size, set_id)
