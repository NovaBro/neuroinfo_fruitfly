"""Channel isolation and binarization helpers."""

from __future__ import annotations

from typing import Any, Tuple, Union

import scyjava


def duplicate_channel(imp: Any, channel: int) -> Any:
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


def binarize(
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
