"""Conversions between Fiji ImagePlus objects and NumPy arrays."""

from __future__ import annotations

from typing import Any

import numpy as np
import scyjava


def imp_to_numpy(ij: Any, imp_c: Any) -> np.ndarray:
    """Convert a single-channel binary ImagePlus to ``uint8`` NumPy ``(Z, Y, X)``."""

    arr = ij.py.from_java(imp_c)
    if hasattr(arr, "values"):
        arr = arr.values
    arr = np.asarray(arr)

    # Drop trailing singleton channel dims that PyImageJ can introduce for
    # single-channel stacks: (Z, Y, X, 1) -> (Z, Y, X).
    while arr.ndim > 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def to_uint8_for_display(a: np.ndarray) -> np.ndarray:
    """Contrast-stretch ``a`` to 8-bit for PNG display."""

    if a.dtype == np.uint8:
        return a
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    scaled = (a - lo) / (hi - lo)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def array_to_imageplus(ij: Any, arr_2d: np.ndarray, title: str) -> Any:
    """Convert a 2-D NumPy array to a Fiji ``ImagePlus``.

    Tries the modern ``ij.py.to_imageplus`` first; falls back to a manual
    construction via ``ImageProcessor`` if that helper isn't available on the
    installed PyImageJ.
    """

    to_imageplus = getattr(ij.py, "to_imageplus", None)
    if to_imageplus is not None:
        imp = to_imageplus(arr_2d)
        imp.setTitle(title)
        return imp

    ImagePlus = scyjava.jimport("ij.ImagePlus")
    height, width = int(arr_2d.shape[0]), int(arr_2d.shape[1])

    if arr_2d.dtype == np.uint8:
        ByteProcessor = scyjava.jimport("ij.process.ByteProcessor")
        proc = ByteProcessor(width, height)
        proc.setPixels(arr_2d.tobytes())
    elif arr_2d.dtype in (np.uint16, np.int16):
        ShortProcessor = scyjava.jimport("ij.process.ShortProcessor")
        proc = ShortProcessor(width, height)
        proc.setPixels(arr_2d.astype(np.uint16).tolist())
    else:
        FloatProcessor = scyjava.jimport("ij.process.FloatProcessor")
        proc = FloatProcessor(width, height)
        proc.setPixels(arr_2d.astype(np.float32).flatten().tolist())

    return ImagePlus(title, proc)
