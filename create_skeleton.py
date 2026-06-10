"""Skeletonize Janelia h5j volumes via Fiji / PyImageJ.

This module is now a thin compatibility shim. The reusable helpers live in the
:mod:`imagej_helper` package; import them directly for new code::

    from imagej_helper import skeletonize_h5j, z_project_and_save

``main()`` below is kept as a runnable demo and ``from create_skeleton import
skeletonize_h5j`` continues to work via the re-exports here.
"""

from __future__ import annotations

import os
from typing import Union

import pandas as pd
import scyjava

from imagej_helper import (
    DEFAULT_FIJI_APP,
    DEFAULT_OUTPUT_DIR,
    PROJECTION_METHODS,
    SkeletonResult,
    analyze_skeleton,
    array_to_imageplus,
    binarize,
    duplicate_channel,
    get_ij,
    imp_to_numpy,
    open_h5j,
    skeletonize_h5j,
    to_uint8_for_display,
    z_project_and_save,
)

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
    "main",
]


def main(output_dir: Union[str, os.PathLike, None] = None) -> None:
    """Demo: skeletonize the bundled sample h5j and save outputs in ``output_dir``.

    Outputs (3D skeleton TIFF + 2D max-projection TIFF/PNG) are written to
    ``output_dir`` (default :data:`DEFAULT_OUTPUT_DIR`, i.e. ``./skeletons/``)
    so the source ``downloads/`` tree is left untouched.
    """

    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "downloads/Descending Neurons 2018/SS01076/SS01076-20140623_33_J1-f-20x-ventral_nerve_cord-Split_GAL4-unaligned_stack.h5j",
    )

    output_dir = os.fspath(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    result = skeletonize_h5j(file_path, channel=2, threshold_method="Otsu")

    print("\n=== Skeleton ===")
    print(f"shape={result.skeleton.shape} dtype={result.skeleton.dtype}")
    print(f"skeleton voxels: {int((result.skeleton > 0).sum())}")
    print(f"threshold used (lo, hi): {result.threshold_used}")

    print("\n=== AnalyzeSkeleton summary ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(result.summary)

    stem = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(output_dir, f"{stem}_skeleton.tif")
    IJ = scyjava.jimport("ij.IJ")
    IJ.saveAsTiff(result.skeleton_imp, out_path)
    print(f"\nSaved skeleton TIFF to {out_path}")

    proj = z_project_and_save(
        result, f"{stem}_skeleton_z", method="max", output_dir=output_dir
    )
    print(
        f"\nZ-projection: shape={proj.shape} dtype={proj.dtype} "
        f"nonzero={int((proj > 0).sum())}"
    )
    print(
        f"Saved {os.path.join(output_dir, stem)}_skeleton_z.tif and "
        f"{os.path.join(output_dir, stem)}_skeleton_z.png"
    )


if __name__ == "__main__":
    main()
