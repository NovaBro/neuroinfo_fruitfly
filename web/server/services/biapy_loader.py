"""Load BiaPy prediction volumes via ipynb/scripts/biapy.py."""

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


def has_predicted_instances(stem: str) -> bool:
    """Return True when BiaPy per_image_instances output exists for ``stem``."""
    return biapy.has_biapy_instances(stem)


def get_predicted_instances_meta(stem: str) -> dict | None:
    """Return predicted-instances metadata when BiaPy output exists."""
    return biapy.get_biapy_instances_meta(stem)


@lru_cache(maxsize=8)
def predicted_instances_to_bytes(stem: str, max_size: int) -> VolumeBytesResult:
    """Load and downsample BiaPy per_image_instances for 3D rendering."""
    volume = biapy.load_biapy_per_image_instances(stem)
    return volume_array_to_bytes(
        volume, max_size=max_size, encoding="labels_rgb"
    )
