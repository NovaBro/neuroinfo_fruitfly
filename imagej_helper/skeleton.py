"""Skeletonization of h5j volumes and per-tree skeleton analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
import scyjava

from .conversion import imp_to_numpy
from .gateway import get_ij
from .io import open_h5j
from .processing import binarize, duplicate_channel


@dataclass
class SkeletonResult:
    """Container for the outputs of :func:`skeletonize_h5j`."""

    skeleton: np.ndarray
    summary: pd.DataFrame
    threshold_used: Tuple[float, float]
    source_imp: Any
    skeleton_imp: Any


def analyze_skeleton(imp_skel: Any) -> pd.DataFrame:
    """Run AnalyzeSkeleton on ``imp_skel`` and return a per-tree DataFrame."""

    AnalyzeSkeleton_ = scyjava.jimport("sc.fiji.analyzeSkeleton.AnalyzeSkeleton_")
    analyzer = AnalyzeSkeleton_()
    analyzer.setup("", imp_skel)
    # AnalyzeSkeleton_.run has two overloads whose second arg is either
    # `boolean pruneEnds` or `double pruneThreshold`; wrap booleans with
    # JBoolean so JPype selects the boolean overload unambiguously.
    # Args: (pruneIndex, pruneEnds, shortPath, origIP, silent, verbose)
    from jpype import JBoolean

    result = analyzer.run(
        AnalyzeSkeleton_.NONE,
        JBoolean(False),
        JBoolean(False),
        None,
        JBoolean(True),
        JBoolean(False),
    )

    def _as_list(java_array: Any) -> list:
        return list(java_array) if java_array is not None else []

    columns = {
        "branches": _as_list(result.getBranches()),
        "junctions": _as_list(result.getJunctions()),
        "end_points": _as_list(result.getEndPoints()),
        "junction_voxels": _as_list(result.getJunctionVoxels()),
        "slabs": _as_list(result.getSlabs()),
        "triples": _as_list(result.getTriples()),
        "quadruples": _as_list(result.getQuadruples()),
        "avg_branch_length": _as_list(result.getAverageBranchLength()),
        "max_branch_length": _as_list(result.getMaximumBranchLength()),
    }

    # All per-tree arrays should have the same length; fall back gracefully if
    # AnalyzeSkeleton ever returns ragged results.
    n_trees = max((len(v) for v in columns.values()), default=0)
    for key, values in columns.items():
        if len(values) < n_trees:
            columns[key] = list(values) + [np.nan] * (n_trees - len(values))

    df = pd.DataFrame(columns)
    df.insert(0, "tree_id", np.arange(1, len(df) + 1))

    if len(df) > 1:
        totals = {
            "tree_id": "total",
            "branches": df["branches"].sum(),
            "junctions": df["junctions"].sum(),
            "end_points": df["end_points"].sum(),
            "junction_voxels": df["junction_voxels"].sum(),
            "slabs": df["slabs"].sum(),
            "triples": df["triples"].sum(),
            "quadruples": df["quadruples"].sum(),
            "avg_branch_length": np.average(
                df["avg_branch_length"],
                weights=df["branches"].clip(lower=1),
            )
            if df["branches"].sum() > 0
            else np.nan,
            "max_branch_length": df["max_branch_length"].max(),
        }
        df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    return df


def skeletonize_h5j(
    file_path: Union[str, os.PathLike],
    *,
    channel: int = 1,
    threshold: Union[int, Tuple[int, int], None] = None,
    threshold_method: str = "Otsu",
    ij_gateway: Any = None,
) -> SkeletonResult:
    """Skeletonize a single channel of an ``.h5j`` file using Fiji.

    Parameters
    ----------
    file_path:
        Path to the ``.h5j`` file.
    channel:
        1-indexed channel to skeletonize (matches Fiji's ``setC()`` convention).
    threshold:
        - ``None`` (default): auto-threshold using ``threshold_method``.
        - ``int``: manual lower cutoff, upper cutoff fixed at 255.
        - ``(lo, hi)`` tuple: explicit threshold band.
    threshold_method:
        Fiji auto-threshold method name (``"Otsu"``, ``"Li"``, ``"Triangle"``,
        ``"Huang"``, ``"MaxEntropy"`` ...). Only used when ``threshold`` is
        ``None``.
    ij_gateway:
        Optional existing PyImageJ gateway to reuse. When ``None``, a cached
        module-level gateway is created on demand.
    """

    file_path = os.fspath(file_path)
    ij = get_ij(ij_gateway)

    print(f"Opening {file_path}...")
    imp = open_h5j(ij, file_path)
    print(
        f"Opened ImagePlus: title={imp.getTitle()}, "
        f"dims={imp.getWidth()}x{imp.getHeight()}x{imp.getNSlices()} "
        f"channels={imp.getNChannels()} frames={imp.getNFrames()}"
    )

    print(f"Duplicating channel {channel}...")
    imp_c = duplicate_channel(imp, channel)

    print(
        "Binarizing ("
        + (f"manual threshold={threshold}" if threshold is not None
           else f"auto method={threshold_method}")
        + ")..."
    )
    threshold_used = binarize(imp_c, threshold, threshold_method)
    print(f"Threshold range applied: {threshold_used}")

    IJ = scyjava.jimport("ij.IJ")
    print("Skeletonizing (2D/3D)...")
    IJ.run(imp_c, "Skeletonize (2D/3D)", "")

    print("Running AnalyzeSkeleton...")
    summary = analyze_skeleton(imp_c)

    print("Converting skeleton to NumPy...")
    skeleton = imp_to_numpy(ij, imp_c)

    return SkeletonResult(
        skeleton=skeleton,
        summary=summary,
        threshold_used=threshold_used,
        source_imp=imp,
        skeleton_imp=imp_c,
    )
