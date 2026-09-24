"""Channel polarity and FISBe stem helpers.

Polarity sets match ``ipynb/view_augments/metric_review.py`` (BiaPy watershed
seed / growth threshold direction). Kept here so watershed_tune does not import
notebook deps (matplotlib / nibabel).
"""

from __future__ import annotations

import re
from typing import Literal

# BiaPy watershed seed polarity (see instance_segmentation workflow docs).
THRESH_ABOVE = frozenset({"F", "P", "Db", "D"})
THRESH_BELOW = frozenset({"C", "B", "T", "Dn", "Dc"})
KNOWN_CHANNELS = THRESH_ABOVE | THRESH_BELOW

# FISBe volume id: {line}-{YYYYMMDD}_{nn}_{well}; trailing BiaPy aug_id is ignored.
_FISBE_SAMPLE_RE = re.compile(r"^(.+-\d{8}_\d+_[A-Z]\d+)")


def fisbe_sample_id(stem: str) -> str:
    """Strip trailing BiaPy aug_id; keep the FISBe volume id.

    Stems are ``{sample}{aug_id}``, e.g. ``R38F04-20181005_63_G3_c021_r0_k0``:
    sample ``R38F04-20181005_63_G3``, aug ``_c021_r0_k0``. No-aug stems and
    unmatched names are returned unchanged.
    """
    match = _FISBE_SAMPLE_RE.match(stem)
    return match.group(1) if match else stem


def channel_threshold_mode(name: str) -> Literal["above", "below"]:
    """Return BiaPy watershed threshold polarity for a channel code."""
    if name in THRESH_ABOVE:
        return "above"
    if name in THRESH_BELOW:
        return "below"
    raise ValueError(
        f"Unknown BiaPy channel {name!r}; expected one of {sorted(KNOWN_CHANNELS)}"
    )
