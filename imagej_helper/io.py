"""File IO helpers for h5j volumes."""

from __future__ import annotations

from typing import Any

import scyjava


def open_h5j(ij: Any, file_path: str) -> Any:
    """Open an h5j file via the IJ1 legacy ``H5J_Loader_Plugin``."""

    IJ = scyjava.jimport("ij.IJ")
    imp = IJ.openImage(file_path)
    if imp is None:
        raise RuntimeError(
            f"IJ.openImage returned null for {file_path}. "
            "Check that H5J_Loader_Plugin is present in Fiji.app/plugins."
        )
    return imp
