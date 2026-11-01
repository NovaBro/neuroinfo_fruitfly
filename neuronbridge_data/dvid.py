"""DVID sparsevol fetch for neuron-only binary segmentation."""

from __future__ import annotations

import urllib.error
import urllib.request

import numpy as np

from neuronbridge_data.datasets import MAX_VOXELS, TIGHT_PAD


def fetch_sparsevol_srles(
    server: str,
    uuid: str,
    instance: str,
    body_id: int,
    scale: int = 0,
    timeout: int = 300,
) -> np.ndarray:
    """Return Nx4 int32 array of (x, y, z, run) spans."""
    base = server.rstrip("/")
    url = (
        f"{base}/api/node/{uuid}/{instance}/sparsevol/{body_id}"
        f"?format=srles&scale={int(scale)}"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"DVID sparsevol failed for bodyId={body_id} scale={scale}: "
            f"HTTP {exc.code} {url}"
        ) from exc
    if len(raw) == 0:
        raise RuntimeError(f"Empty sparsevol for bodyId={body_id}")
    if len(raw) % 16 != 0:
        raise RuntimeError(
            f"Unexpected sparsevol size {len(raw)} for bodyId={body_id} "
            "(expected multiple of 16 for srles)"
        )
    return np.frombuffer(raw, dtype="<i4").reshape(-1, 4)


def spans_to_binary(spans: np.ndarray, pad: int = TIGHT_PAD) -> np.ndarray:
    """Paint srles spans into a tight uint8 binary volume."""
    if spans.size == 0:
        raise RuntimeError("No sparsevol spans to paint")
    x0 = int(spans[:, 0].min())
    y0 = int(spans[:, 1].min())
    z0 = int(spans[:, 2].min())
    x1 = int((spans[:, 0] + spans[:, 3]).max())
    y1 = int(spans[:, 1].max()) + 1
    z1 = int(spans[:, 2].max()) + 1

    # Expand by pad (no volume bounds available here)
    x0p, y0p, z0p = x0 - pad, y0 - pad, z0 - pad
    x1p, y1p, z1p = x1 + pad, y1 + pad, z1 + pad
    shape = (x1p - x0p, y1p - y0p, z1p - z0p)
    n_vox = int(np.prod(shape))
    if n_vox > MAX_VOXELS:
        raise RuntimeError(
            f"Sparsevol tight bbox {shape} = {n_vox:,} voxels exceeds "
            f"limit {MAX_VOXELS:,}. Raise --mip (DVID scale)."
        )
    vol = np.zeros(shape, dtype=np.uint8)
    for x, y, z, run in spans:
        xa = int(x) - x0p
        ya = int(y) - y0p
        za = int(z) - z0p
        vol[xa : xa + int(run), ya, za] = 1
    return vol
