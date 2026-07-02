"""Serve the pre-generated FISBe MIP PNGs (no on-the-fly zarr projection).

FISBe ships one MIP PNG per sample (fisbe_v1.0_mips.zip). Each PNG is a
side-by-side composite: the left half is the original colored raw MIP and the
right half is the ground-truth instance-segmentation MIP (one color per neuron).
We locate the file on disk and, depending on the requested ``half``, hand back
the whole image, just the raw half, or just the GT half — cropping only, never
regenerating a projection.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

from PIL import Image

from config import FISBE_MIPS_ROOT
from services.sample_list import SampleEntry

MipHalf = Literal["full", "raw", "gt"]


def fisbe_mip_path(entry: SampleEntry) -> Path:
    """On-disk location of a sample's pre-generated MIP PNG."""
    return FISBE_MIPS_ROOT / entry.dataset / entry.split / f"{entry.name}.png"


def fisbe_mip_png(entry: SampleEntry, half: MipHalf = "full") -> bytes:
    """Return the pre-generated MIP PNG bytes, optionally cropped to one half.

    Raises FileNotFoundError if no MIP PNG exists for the sample.
    """
    path = fisbe_mip_path(entry)
    if not path.is_file():
        raise FileNotFoundError(f"FISBe MIP not found on disk: {path}")

    with Image.open(path) as img:
        img = img.convert("RGB")
        if half == "full":
            cropped = img
        else:
            mid = img.width // 2
            box = (0, 0, mid, img.height) if half == "raw" else (mid, 0, img.width, img.height)
            cropped = img.crop(box)

        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()
