"""Z-projection of 3-D stacks with TIFF + PNG output."""

from __future__ import annotations

import os
from typing import Any, Optional, Union

import numpy as np
import scyjava

from .conversion import array_to_imageplus, to_uint8_for_display
from .gateway import get_ij
from .skeleton import SkeletonResult

PROJECTION_METHODS = ("max", "min", "sum", "mean", "stddev", "median")


def z_project_and_save(
    data: Union[np.ndarray, SkeletonResult],
    out_path: Union[str, os.PathLike],
    *,
    method: str = "max",
    output_dir: Optional[Union[str, os.PathLike]] = None,
    ij_gateway: Any = None,
) -> np.ndarray:
    """Collapse a ``(Z, Y, X)`` stack along axis 0 and save TIFF + PNG.

    Parameters
    ----------
    data:
        Either a 3-D NumPy array shaped ``(Z, Y, X)`` or a :class:`SkeletonResult`
        (whose ``.skeleton`` attribute is used).
    out_path:
        Base path (or just a filename stem). Any extension is stripped, and the
        function writes ``{stem}.tif`` (dtype-preserving) and ``{stem}.png``
        (8-bit, contrast stretched) side by side. When ``output_dir`` is also
        given, only the basename of ``out_path`` is kept and the files land
        inside ``output_dir`` regardless of the directory in ``out_path``.
    method:
        One of ``"max"``, ``"min"``, ``"sum"``, ``"mean"``, ``"stddev"``,
        ``"median"``. Default ``"max"`` (right choice for binary skeletons).
    output_dir:
        Optional directory to write into. When given, it is created if it does
        not exist and takes precedence over the directory portion of
        ``out_path``. Useful for keeping outputs out of read-only source trees
        such as ``downloads/``.
    ij_gateway:
        Optional PyImageJ gateway; falls back to the module-level cache.

    Returns
    -------
    np.ndarray
        The 2-D projection as a NumPy array at full precision (not the 8-bit
        PNG copy).
    """

    if isinstance(data, SkeletonResult):
        arr = data.skeleton
    else:
        arr = np.asarray(data)

    if arr.ndim != 3:
        raise ValueError(
            f"z_project_and_save expects a 3-D (Z, Y, X) array; got shape {arr.shape}."
        )

    method_key = method.lower()
    if method_key not in PROJECTION_METHODS:
        raise ValueError(
            f"Unknown projection method {method!r}; valid options: {PROJECTION_METHODS}."
        )

    if method_key == "max":
        proj = np.max(arr, axis=0)
    elif method_key == "min":
        proj = np.min(arr, axis=0)
    elif method_key == "sum":
        # Promote integer accumulators to uint32 to avoid uint8 overflow.
        sum_dtype = np.uint32 if arr.dtype.kind in "ub" else np.float64
        proj = np.sum(arr, axis=0, dtype=sum_dtype)
    elif method_key == "mean":
        proj = np.mean(arr, axis=0)
    elif method_key == "stddev":
        proj = np.std(arr, axis=0)
    else:  # median
        proj = np.median(arr, axis=0)

    stem = os.path.splitext(os.fspath(out_path))[0]
    if output_dir is not None:
        output_dir = os.fspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, os.path.basename(stem))
    else:
        base = stem
    tif_path = base + ".tif"
    png_path = base + ".png"

    ij = get_ij(ij_gateway)
    IJ = scyjava.jimport("ij.IJ")

    title = os.path.basename(base)
    imp_tif = array_to_imageplus(ij, proj, f"{title}_{method_key}")
    IJ.saveAsTiff(imp_tif, tif_path)

    imp_png = array_to_imageplus(
        ij, to_uint8_for_display(proj), f"{title}_{method_key}_8bit"
    )
    IJ.saveAs(imp_png, "PNG", png_path)

    return proj
