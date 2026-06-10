"""Reusable PyImageJ / Fiji helpers.

Import the high-level entry points directly::

    from imagej_helper import skeletonize_h5j, z_project_and_save

or reach for the lower-level building blocks::

    from imagej_helper import get_ij, open_h5j, binarize
"""

from __future__ import annotations

from .conversion import array_to_imageplus, imp_to_numpy, to_uint8_for_display
from .gateway import DEFAULT_FIJI_APP, DEFAULT_OUTPUT_DIR, get_ij
from .io import open_h5j
from .processing import binarize, duplicate_channel
from .projection import PROJECTION_METHODS, z_project_and_save
from .skeleton import SkeletonResult, analyze_skeleton, skeletonize_h5j

__all__ = [
    "DEFAULT_FIJI_APP",
    "DEFAULT_OUTPUT_DIR",
    "PROJECTION_METHODS",
    "SkeletonResult",
    "analyze_skeleton",
    "array_to_imageplus",
    "binarize",
    "duplicate_channel",
    "get_ij",
    "imp_to_numpy",
    "open_h5j",
    "skeletonize_h5j",
    "to_uint8_for_display",
    "z_project_and_save",
]
