"""Load BiaPy prediction volumes via ipynb/scripts/biapy.py.

A "prediction set" is one BiaPy run directory (containing ``per_image_instances``)
under ``BiaPy/results``. Callers select one by its ``id`` (path relative to the
results base); ``None`` falls back to the default set.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from config import REPO_ROOT
from services.volume_pipeline import VolumeBytesResult, volume_array_to_bytes

_SCRIPTS_DIR = REPO_ROOT / "ipynb" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import biapy  # noqa: E402


def list_prediction_sets() -> list[dict]:
    """List available prediction sets discovered under the BiaPy results base."""
    return biapy.discover_prediction_sets()


def has_predicted_instances(stem: str, set_id: str | None = None) -> bool:
    """Return True when the given prediction set has output for ``stem``."""
    try:
        root = biapy.resolve_prediction_set_root(set_id)
    except (FileNotFoundError, ValueError):
        return False
    return biapy.has_biapy_instances(stem, result_root=root)


def has_predicted_instances_any(stem: str) -> bool:
    """Return True when *any* prediction set has output for ``stem``."""
    return any(
        biapy.has_biapy_instances(stem, result_root=Path(s["path"]))
        for s in biapy.discover_prediction_sets()
    )


def get_predicted_instances_meta(
    stem: str, set_id: str | None = None
) -> dict | None:
    """Return predicted-instances metadata for ``stem`` in the given set."""
    try:
        root = biapy.resolve_prediction_set_root(set_id)
    except (FileNotFoundError, ValueError):
        return None
    return biapy.get_biapy_instances_meta(stem, result_root=root)


@lru_cache(maxsize=16)
def predicted_instances_to_bytes(
    stem: str, max_size: int, set_id: str | None = None
) -> VolumeBytesResult:
    """Load and downsample a prediction set's per_image_instances for rendering."""
    root = biapy.resolve_prediction_set_root(set_id)
    volume = biapy.load_biapy_per_image_instances(stem, result_root=root)
    return volume_array_to_bytes(
        volume, max_size=max_size, encoding="labels_rgb"
    )
