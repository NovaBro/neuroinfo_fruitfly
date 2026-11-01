"""Download neuron-only binary segmentation (no EM tissue)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from cloudvolume import CloudVolume
from neuprint import Client

from neuronbridge_data.catalog import skeleton_bbox_mip0, skeleton_points_mip0
from neuronbridge_data.datasets import MAX_VOXELS, TIGHT_PAD
from neuronbridge_data.dvid import fetch_sparsevol_srles, spans_to_binary


def open_volume(cloudpath: str, mip: int) -> CloudVolume:
    return CloudVolume(
        cloudpath,
        mip=mip,
        use_https=True,
        fill_missing=True,
        progress=True,
    )


def mip0_bbox_to_mip(
    vol: CloudVolume, lo: tuple[int, int, int], hi: tuple[int, int, int]
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    fx, fy, fz = (int(x) for x in vol.downsample_ratio)
    lo_m = (lo[0] // fx, lo[1] // fy, lo[2] // fz)
    hi_m = (
        (hi[0] + fx - 1) // fx,
        (hi[1] + fy - 1) // fy,
        (hi[2] + fz - 1) // fz,
    )
    bmin = vol.bounds.minpt
    bmax = vol.bounds.maxpt
    lo_m = (
        max(lo_m[0], int(bmin.x)),
        max(lo_m[1], int(bmin.y)),
        max(lo_m[2], int(bmin.z)),
    )
    hi_m = (
        min(hi_m[0], int(bmax.x)),
        min(hi_m[1], int(bmax.y)),
        min(hi_m[2], int(bmax.z)),
    )
    return lo_m, hi_m


def fetch_bbox(
    vol: CloudVolume, lo: tuple[int, int, int], hi: tuple[int, int, int]
) -> np.ndarray:
    cut = np.asarray(vol[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]])
    return np.squeeze(cut)


def tight_crop_binary(mask: np.ndarray, pad: int = TIGHT_PAD) -> np.ndarray:
    """Crop uint8/bool mask to nonzero bbox (+ pad)."""
    nz = np.argwhere(mask > 0)
    if nz.size == 0:
        raise RuntimeError("Cannot tight-crop: mask has no nonzero voxels")
    lo = nz.min(axis=0) - pad
    hi = nz.max(axis=0) + 1 + pad
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, mask.shape)
    return np.ascontiguousarray(
        mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].astype(np.uint8)
    )


def _save_seg(
    out_dir: Path,
    seg: np.ndarray,
    body_id: int,
    meta: dict[str, Any],
    neuron_meta: dict[str, Any] | None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Drop legacy tissue EM if present
    em_path = out_dir / "em.npy"
    if em_path.is_file():
        em_path.unlink()
    np.save(out_dir / "seg.npy", seg.astype(np.uint8))
    if neuron_meta:
        meta.update(neuron_meta)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"Wrote {out_dir}/seg.npy shape={seg.shape} dtype=uint8 "
        f"n_voxels={meta['n_voxels']}"
    )
    return out_dir


def download_via_dvid(
    cfg: dict[str, Any],
    body_id: int,
    scale: int,
    out_root: Path,
    neuron_meta: dict[str, Any] | None,
) -> Path:
    spans = fetch_sparsevol_srles(
        cfg["dvid_server"],
        cfg["dvid_uuid"],
        cfg["dvid_instance"],
        body_id,
        scale=scale,
    )
    seg = spans_to_binary(spans, pad=TIGHT_PAD)
    n_vox = int(np.count_nonzero(seg))
    if n_vox == 0:
        raise RuntimeError(f"bodyId={body_id}: DVID sparsevol painted zero voxels")
    meta = {
        "bodyId": body_id,
        "seg_mode": "dvid_sparsevol",
        "mip": scale,
        "scale": scale,
        "seg_shape": list(seg.shape),
        "n_voxels": n_vox,
        "source": (
            f"{cfg['dvid_server']}/api/node/{cfg['dvid_uuid']}/"
            f"{cfg['dvid_instance']}/sparsevol/{body_id}"
        ),
    }
    return _save_seg(out_root / str(body_id), seg, body_id, meta, neuron_meta)


def _sample_labels_along_skeleton(
    seg_vol: CloudVolume,
    points_mip0: np.ndarray,
    max_samples: int = 800,
) -> set[int]:
    fx, fy, fz = (int(x) for x in seg_vol.downsample_ratio)
    pts = (points_mip0 / np.array([fx, fy, fz], dtype=np.float64)).astype(np.int64)
    if len(pts) > max_samples:
        idx = np.linspace(0, len(pts) - 1, num=max_samples, dtype=int)
        pts = pts[idx]
    bmin = seg_vol.bounds.minpt
    bmax = seg_vol.bounds.maxpt
    labels: set[int] = set()
    for x, y, z in pts:
        if x < bmin.x or y < bmin.y or z < bmin.z:
            continue
        if x >= bmax.x or y >= bmax.y or z >= bmax.z:
            continue
        val = int(np.asarray(seg_vol[x : x + 1, y : y + 1, z : z + 1]).ravel()[0])
        if val != 0:
            labels.add(val)
    return labels


def download_via_cloudvolume(
    client: Client,
    cfg: dict[str, Any],
    body_id: int,
    mip: int,
    pad: int,
    out_root: Path,
    neuron_meta: dict[str, Any] | None,
    *,
    mask_mode: str,
) -> Path:
    seg_vol = open_volume(cfg["seg"], mip)
    lo0, hi0 = skeleton_bbox_mip0(client, body_id, seg_vol, pad)
    lo_m, hi_m = mip0_bbox_to_mip(seg_vol, lo0, hi0)
    shape = tuple(hi_m[i] - lo_m[i] for i in range(3))
    n_box = int(np.prod(shape))
    if n_box > MAX_VOXELS:
        raise RuntimeError(
            f"bodyId={body_id} crop at mip={mip} is {shape} = {n_box:,} voxels "
            f"(limit {MAX_VOXELS:,}). Raise --mip or pick a smaller neuron."
        )

    print(f"bodyId={body_id} mip0_bbox={lo0}->{hi0} mip{mip}_bbox={lo_m}->{hi_m}")
    seg = fetch_bbox(seg_vol, lo_m, hi_m)

    if mask_mode == "body":
        binary = (seg == body_id).astype(np.uint8)
    elif mask_mode == "skel_labels":
        pts = skeleton_points_mip0(client, body_id)
        labels = _sample_labels_along_skeleton(seg_vol, pts)
        if not labels:
            raise RuntimeError(
                f"bodyId={body_id}: no nonzero labels along skeleton at mip={mip}"
            )
        print(f"bodyId={body_id}: {len(labels)} supervoxel labels from skeleton")
        binary = np.isin(seg, list(labels)).astype(np.uint8)
    else:
        raise ValueError(f"Unknown mask_mode={mask_mode}")

    n_before = int(np.count_nonzero(binary))
    if n_before == 0:
        raise RuntimeError(
            f"bodyId={body_id}: zero voxels after masking at mip={mip}. "
            "Wrong seg source or body ID mismatch."
        )
    binary = tight_crop_binary(binary, pad=TIGHT_PAD)
    n_vox = int(np.count_nonzero(binary))
    meta = {
        "bodyId": body_id,
        "seg_mode": (
            "cloudvolume_body" if mask_mode == "body" else "cloudvolume_skel_labels"
        ),
        "mip": mip,
        "pad_mip0": pad,
        "bbox_mip0": {"lo": list(lo0), "hi": list(hi0)},
        "bbox_mip": {"lo": list(lo_m), "hi": list(hi_m)},
        "seg_shape": list(binary.shape),
        "n_voxels": n_vox,
        "source": cfg["seg"],
    }
    return _save_seg(out_root / str(body_id), binary, body_id, meta, neuron_meta)


def download_body(
    client: Client,
    cfg: dict[str, Any],
    body_id: int,
    mip: int,
    pad: int,
    out_root: Path,
    neuron_meta: dict[str, Any] | None = None,
) -> Path:
    mode = cfg.get("seg_mode", "cloudvolume_body")
    if mode == "dvid_sparsevol":
        return download_via_dvid(cfg, body_id, mip, out_root, neuron_meta)
    if mode == "cloudvolume_body":
        return download_via_cloudvolume(
            client,
            cfg,
            body_id,
            mip,
            pad,
            out_root,
            neuron_meta,
            mask_mode="body",
        )
    if mode == "cloudvolume_skel_labels":
        return download_via_cloudvolume(
            client,
            cfg,
            body_id,
            mip,
            pad,
            out_root,
            neuron_meta,
            mask_mode="skel_labels",
        )
    raise RuntimeError(f"Unknown seg_mode={mode} for bodyId={body_id}")
