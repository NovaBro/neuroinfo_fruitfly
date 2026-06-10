"""JVM bootstrap and cached PyImageJ gateway.

Importing this module configures the JVM environment (Conda Java, headless
mode, IJ1 legacy layer) *before* ``imagej`` is imported, which is required so
that ``IJ.openImage`` and the ``H5J_Loader_Plugin`` work from notebook kernels.
"""

from __future__ import annotations

import os

# Force PyImageJ to use Conda's Java, ignoring system defaults. This must be
# set before scyjava / imagej are imported, because those trigger JVM startup.
if "CONDA_PREFIX" in os.environ:
    os.environ.setdefault("JAVA_HOME", os.environ["CONDA_PREFIX"])

from typing import Any

import scyjava

# Ensure the JVM comes up headless with the IJ1 legacy layer active so that
# IJ.openImage (and thus the H5J_Loader_Plugin) work from notebook kernels.
scyjava.config.add_option("-Djava.awt.headless=true")

import imagej


DEFAULT_FIJI_APP = "/home/william-zheng/Downloads/Fiji.app"

# Default directory for skeleton/projection outputs. Lives at the project root
# (the parent of this package) so the source `downloads/` tree (populated by
# s3_download_fast.sh) is never modified by the skeletonization pipeline.
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skeletons"
)

# Module-level cache so repeated calls in the same kernel reuse one JVM.
_IJ_GATEWAY: Any = None


def get_ij(ij_gateway: Any = None, fiji_app: str = DEFAULT_FIJI_APP) -> Any:
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
