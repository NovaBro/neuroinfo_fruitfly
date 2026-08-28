"""Skeletonize Janelia h5j volumes via Fiji / PyImageJ.

The public entry point is :func:`skeletonize_h5j`, which opens an ``.h5j`` file
through Fiji's ``H5J_Loader_Plugin``, isolates a single channel, binarizes it,
runs ``Skeletonize3D``, and then reports per-tree statistics from
``AnalyzeSkeleton`` as a :class:`pandas.DataFrame`.
"""

from __future__ import annotations

import os

# Force PyImageJ to use Conda's Java, ignoring system defaults. This must be
# set before scyjava / imagej are imported, because those trigger JVM startup.
if "CONDA_PREFIX" in os.environ:
    os.environ.setdefault("JAVA_HOME", os.environ["CONDA_PREFIX"])

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scyjava

# Ensure the JVM comes up headless with the IJ1 legacy layer active so that
# IJ.openImage (and thus the H5J_Loader_Plugin) work from notebook kernels.
scyjava.config.add_option("-Djava.awt.headless=true")

import imagej


# DEFAULT_FIJI_APP = "/home/william-zheng/Downloads/Fiji.app"
DEFAULT_FIJI_APP = "/Users/vuhepola/Desktop/Fiji"

# Default directory for skeleton/projection outputs. Lives next to this script
# so the source `downloads/` tree (populated by s3_download_fast.sh) is never
# modified by the skeletonization pipeline.
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "skeletons"
)

# Module-level cache so repeated calls in the same kernel reuse one JVM.
_IJ_GATEWAY: Any = None


@dataclass
class SkeletonResult:
    """Container for the outputs of :func:`skeletonize_h5j`."""

    skeleton: np.ndarray
    summary: pd.DataFrame
    threshold_used: Tuple[float, float]
    source_imp: Any
    skeleton_imp: Any


def _get_ij(ij_gateway: Any = None, fiji_app: str = DEFAULT_FIJI_APP) -> Any:
    """Return a cached PyImageJ gateway, initializing one if necessary."""

    global _IJ_GATEWAY
    if ij_gateway is not None:
        return ij_gateway
    if _IJ_GATEWAY is not None:
        return _IJ_GATEWAY

    print("Initializing headless Fiji environment...")
    ij = imagej.init(fiji_app, mode="headless", add_legacy=True)
    version = str(ij.getVersion())
    print(f"ImageJ version: {version}")
    if version.endswith("/Inactive"):
        raise RuntimeError(
            "Legacy ImageJ1 is Inactive in this kernel. Restart the Python "
            "process so the JVM can be started fresh with the IJ1 legacy "
            "layer active (required for IJ.openImage and the H5J plugin)."
        )
    _IJ_GATEWAY = ij
    return ij


def _open_h5j(ij: Any, file_path: str) -> Any:
    """Open an h5j file via the IJ1 legacy ``H5J_Loader_Plugin``."""

    IJ = scyjava.jimport("ij.IJ")
    imp = IJ.openImage(file_path)
    if imp is None:
        raise RuntimeError(
            f"IJ.openImage returned null for {file_path}. "
            "Check that H5J_Loader_Plugin is present in Fiji.app/plugins."
        )
    return imp


def _duplicate_channel(imp: Any, channel: int) -> Any:
    """Return a new ImagePlus containing just the requested 1-indexed channel."""

    n_channels = int(imp.getNChannels())
    n_slices = int(imp.getNSlices())
    n_frames = int(imp.getNFrames())
    if not 1 <= channel <= n_channels:
        raise ValueError(
            f"channel={channel} out of range; image has {n_channels} channel(s)."
        )

    Duplicator = scyjava.jimport("ij.plugin.Duplicator")
    imp_c = Duplicator().run(imp, channel, channel, 1, n_slices, 1, n_frames)
    imp_c.setTitle(f"{imp.getTitle()}__C{channel}")
    return imp_c


def _binarize(
    imp_c: Any,
    threshold: Union[int, Tuple[int, int], None],
    threshold_method: str,
) -> Tuple[float, float]:
    """Convert ``imp_c`` to an 8-bit binary mask in place.

    Returns the ``(lo, hi)`` intensity cutoffs that were actually applied.
    """

    IJ = scyjava.jimport("ij.IJ")

    # Skeletonize3D requires an 8-bit image; collapse first so threshold values
    # are interpreted against the same 0-255 range the user sees in Fiji.
    if imp_c.getBitDepth() != 8:
        IJ.run(imp_c, "8-bit", "")

    if threshold is None:
        IJ.setAutoThreshold(imp_c, f"{threshold_method} dark stack")
        proc = imp_c.getProcessor()
        lo = float(proc.getMinThreshold())
        hi = float(proc.getMaxThreshold())
        IJ.run(
            imp_c,
            "Convert to Mask",
            f"method={threshold_method} background=Dark black",
        )
    else:
        if isinstance(threshold, tuple):
            lo, hi = float(threshold[0]), float(threshold[1])
        else:
            lo, hi = float(threshold), 255.0
        imp_c.setThreshold(lo, hi)
        IJ.run(imp_c, "Convert to Mask", "method=Default background=Dark black")

    return lo, hi


def _analyze_skeleton(imp_skel: Any) -> pd.DataFrame:
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


def _imp_to_numpy(ij: Any, imp_c: Any) -> np.ndarray:
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


_PROJECTION_METHODS = ("max", "min", "sum", "mean", "stddev", "median")


def _to_uint8_for_display(a: np.ndarray) -> np.ndarray:
    """Contrast-stretch ``a`` to 8-bit for PNG display."""

    if a.dtype == np.uint8:
        return a
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    scaled = (a - lo) / (hi - lo)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def _array_to_imageplus(ij: Any, arr_2d: np.ndarray, title: str) -> Any:
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
    if method_key not in _PROJECTION_METHODS:
        raise ValueError(
            f"Unknown projection method {method!r}; valid options: {_PROJECTION_METHODS}."
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

    ij = _get_ij(ij_gateway)
    IJ = scyjava.jimport("ij.IJ")

    title = os.path.basename(base)
    imp_tif = _array_to_imageplus(ij, proj, f"{title}_{method_key}")
    IJ.saveAsTiff(imp_tif, tif_path)

    imp_png = _array_to_imageplus(
        ij, _to_uint8_for_display(proj), f"{title}_{method_key}_8bit"
    )
    IJ.saveAs(imp_png, "PNG", png_path)

    return proj


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
    ij = _get_ij(ij_gateway)

    print(f"Opening {file_path}...")
    imp = _open_h5j(ij, file_path)
    print(
        f"Opened ImagePlus: title={imp.getTitle()}, "
        f"dims={imp.getWidth()}x{imp.getHeight()}x{imp.getNSlices()} "
        f"channels={imp.getNChannels()} frames={imp.getNFrames()}"
    )

    print(f"Duplicating channel {channel}...")
    imp_c = _duplicate_channel(imp, channel)

    print(
        "Binarizing ("
        + (f"manual threshold={threshold}" if threshold is not None
           else f"auto method={threshold_method}")
        + ")..."
    )
    threshold_used = _binarize(imp_c, threshold, threshold_method)
    print(f"Threshold range applied: {threshold_used}")

    IJ = scyjava.jimport("ij.IJ")
    print("Skeletonizing (2D/3D)...")
    IJ.run(imp_c, "Skeletonize (2D/3D)", "")

    print("Running AnalyzeSkeleton...")
    summary = _analyze_skeleton(imp_c)

    print("Converting skeleton to NumPy...")
    skeleton = _imp_to_numpy(ij, imp_c)

    return SkeletonResult(
        skeleton=skeleton,
        summary=summary,
        threshold_used=threshold_used,
        source_imp=imp,
        skeleton_imp=imp_c,
    )



def main(output_dir: Union[str, os.PathLike, None] = None) -> None:
    """Demo: skeletonize the bundled sample h5j and save outputs in ``output_dir``.

    Outputs (3D skeleton TIFF + 2D max-projection TIFF/PNG) are written to
    ``output_dir`` (default :data:`DEFAULT_OUTPUT_DIR`, i.e. ``./skeletons/``)
    so the source ``downloads/`` tree is left untouched.
    """

    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        # "downloads/Descending Neurons 2018/SS01076/SS01076-20140623_33_J1-f-20x-ventral_nerve_cord-Split_GAL4-unaligned_stack.h5j",
        "/Users/vuhepola/GitHub/Repos/neuroinfo_fruitfly/data/SS00771-20131227_32_H2-f-20x-ventral_nerve_cord-Split_GAL4-unaligned_stack.h5j",
        
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
