"""Render single SWCs or whole 1k_v1 images with SWC2IMG, plus MIP / comparison PNGs."""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile

from .fiji import run_swc2img
from .swc import clip_to_box, merge, read_swc, with_anchors, write_swc


@dataclass
class RenderResult:
    volume: np.ndarray          # float32, ZYX
    tif: Path
    mip_png: Path
    log: str
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- PNG helpers

def _norm(a: np.ndarray, lo_pct=0.5, hi_pct=99.9) -> np.ndarray:
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


def save_mip_png(volume: np.ndarray, path: str | os.PathLike) -> Path:
    """Greyscale max projection along Z of a ZYX float volume."""
    from PIL import Image
    mip = _norm(volume.max(axis=0))
    Image.fromarray((mip * 255).astype(np.uint8)).save(path)
    return Path(path)


def v1_raw_mip_rgb(zarr_path: str | os.PathLike) -> np.ndarray:
    """RGB max projection of a v1 raw volume (channels HA, V5, FLAG -> R, G, B)."""
    import zarr
    raw = zarr.open(str(zarr_path), mode="r")["volumes/raw"]
    return np.stack([_norm(np.asarray(raw[c]).max(axis=0)) for c in range(raw.shape[0])], axis=-1)


def save_comparison_png(swc2img_vol: np.ndarray, v1_zarr: str | os.PathLike, path, title: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v1 = v1_raw_mip_rgb(v1_zarr)
    s = _norm(swc2img_vol.max(axis=0))
    fig, ax = plt.subplots(1, 2, figsize=(14, 7.4), constrained_layout=True)
    ax[0].imshow(v1)
    ax[0].set_title("synth_MCFO 1k_v1 (mcfo_sim), MIP")
    ax[1].imshow(s, cmap="gray")
    ax[1].set_title("SWC2IMG, same neurons and field of view, MIP")
    for a in ax:
        a.set_axis_off()
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return Path(path)


# ---------------------------------------------------------------- renders

def render_swc(swc_path, out_dir, **swc2img_kw) -> RenderResult:
    """Render one SWC file as-is (SWC2IMG picks the field of view: neuron extent + border)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(swc_path)
    local = out_dir / src.name
    if src.resolve() != local.resolve():
        shutil.copyfile(src, local)
    tif, corrected, log = run_swc2img(local, **swc2img_kw)
    vol = tifffile.imread(tif)
    mip = save_mip_png(vol, out_dir / f"{local.stem}_swc2img_mip.png")
    return RenderResult(vol, tif, mip, log)


def _v1_row(bundle: Path, image_id: str) -> dict:
    with open(bundle / "images.csv") as f:
        for row in csv.DictReader(f):
            if row["image_id"] == image_id:
                return row
    raise KeyError(f"{image_id} not in {bundle / 'images.csv'}")


TAGS = ("HA", "V5", "FLAG")  # v1 channel order -> R, G, B


def _load_neurons(bundle: Path, body_ids, radius_scale: float) -> dict:
    out = {}
    for b in body_ids:
        n = read_swc(bundle / "swc" / f"{b}.swc")
        n["r"] *= radius_scale
        out[b] = n
    return out


def _v1_box(bundle: Path, row: dict, image_id: str, voxel_um: float):
    lo = np.array(json.loads(row["crop_lo_um_xyz"]), float)
    hi = np.array(json.loads(row["crop_hi_um_xyz"]), float)
    ex_json = bundle / "examples" / image_id / f"{image_id}.json"
    if ex_json.exists():  # use the v1 render's exact shape when we have it
        shape_xyz = np.array(json.load(open(ex_json))["shape_zyx"][::-1])
    else:
        shape_xyz = np.round((hi - lo) / voxel_um).astype(int)
    return lo, hi, shape_xyz


def _render_in_box(nodes, name, out_dir: Path, lo, hi, shape_xyz, voxel_um, header, **swc2img_kw):
    """Render `nodes` with SWC2IMG and crop to the box [lo, hi] (shape_xyz voxels). Returns (vol, tif, log)."""
    if lo is not None:
        nodes = with_anchors(clip_to_box(nodes, lo, hi), lo, hi)
    swc = out_dir / f"{name}.swc"
    write_swc(swc, nodes, header=header)
    tif, corrected, log = run_swc2img(swc, voxel_um=voxel_um, **swc2img_kw)
    vol = tifffile.imread(tif)
    if lo is not None:
        anchor_lo = read_swc(corrected)[-2]
        start_xyz = np.round(np.array([anchor_lo["x"], anchor_lo["y"], anchor_lo["z"]]) / voxel_um).astype(int)
        (x0, y0, z0), (nx, ny, nz) = start_xyz, shape_xyz
        vol = np.ascontiguousarray(vol[z0:z0 + nz, y0:y0 + ny, x0:x0 + nx])
        tifffile.imwrite(tif, vol, imagej=True, resolution=(1 / voxel_um, 1 / voxel_um),
                         metadata={"spacing": voxel_um, "unit": "micron", "axes": "ZYX"})
    return vol, tif, log


def render_v1_image(
    image_id: str,
    bundle_dir,
    out_dir,
    *,
    crop: bool = True,
    radius_scale: float = 1.0,
    voxel_um: float = 0.44,
    **swc2img_kw,
) -> RenderResult:
    """Render all neurons of one 1k_v1 image into one greyscale volume.

    With ``crop=True`` (default) the volume covers exactly the v1 field of view
    (``crop_lo_um_xyz`` .. ``crop_hi_um_xyz`` in images.csv) at ``voxel_um``, so it aligns with
    the v1 render. v1 additionally applied small placement jitter and radius x0.75; pass
    ``radius_scale=0.75`` to match the radii.
    """
    bundle, out_dir = Path(bundle_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row = _v1_row(bundle, image_id)
    if row.get("integrity_ok", "True") != "True":
        print(f"warning: {image_id} has integrity_ok=False in images.csv")
    body_ids = row["body_ids"].split()
    nodes = merge(list(_load_neurons(bundle, body_ids, radius_scale).values()))
    meta = {"image_id": image_id, "body_ids": body_ids, "overlap_mode": row["overlap_mode"],
            "split": row["split"], "voxel_um": voxel_um, "radius_scale": radius_scale,
            "crop": crop, "swc2img": {k: v for k, v in swc2img_kw.items() if k != "fiji_dir"}}

    lo = hi = shape_xyz = None
    if crop:
        lo, hi, shape_xyz = _v1_box(bundle, row, image_id, voxel_um)
        meta.update(crop_lo_um_xyz=lo.tolist(), crop_hi_um_xyz=hi.tolist())
    vol, tif, log = _render_in_box(
        nodes, image_id, out_dir, lo, hi, shape_xyz, voxel_um,
        f"{image_id}: neurons {' '.join(body_ids)}\nmale CNS v1.0, JRC2018U, um", **swc2img_kw)
    meta["shape_zyx"] = list(vol.shape)
    json.dump(meta, open(out_dir / f"{image_id}.swc2img.json", "w"), indent=1)

    mip = save_mip_png(vol, out_dir / f"{image_id}_swc2img_mip.png")
    v1_zarr = bundle / "examples" / image_id / f"{image_id}.zarr"
    if crop and v1_zarr.exists():
        save_comparison_png(vol, v1_zarr, out_dir / f"{image_id}_compare.png",
                            f"{image_id} ({row['overlap_mode']}, {len(body_ids)} neurons)")
    return RenderResult(vol, tif, mip, log, meta)


def _random_mcfo_tags(n: int, rng: np.random.Generator, q: float = 0.45) -> list[str]:
    """MCFO-1 style: each tag excised independently with probability q, redrawn until >= 1 tag."""
    out = []
    for _ in range(n):
        on = np.zeros(3, bool)
        while not on.any():
            on = rng.random(3) < q
        out.append("+".join(t for t, o in zip(TAGS, on) if o))
    return out


def render_v1_image_rgb(
    image_id: str,
    bundle_dir,
    out_dir,
    *,
    tags: str = "v1",
    seed: int = 0,
    radius_scale: float = 1.0,
    voxel_um: float = 0.44,
    **swc2img_kw,
) -> RenderResult:
    """MCFO-style 3-channel render of one 1k_v1 image, cropped to the v1 field of view.

    One SWC2IMG render per tag channel (HA, V5, FLAG): channel c contains every neuron that
    expresses tag c, so neurons with several tags get mixed colours, and each channel has its
    own SWC2IMG noise. ``tags="v1"`` uses the tags from neurons.csv (same colours as the v1
    render); ``tags="random"`` draws new MCFO tags with ``seed``.
    Output volume: float32 (3, Z, Y, X).
    """
    bundle, out_dir = Path(bundle_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row = _v1_row(bundle, image_id)
    body_ids = row["body_ids"].split()
    if tags == "v1":
        with open(bundle / "neurons.csv") as f:
            tag_of = {r["bodyId"]: r["tags"] for r in csv.DictReader(f) if r["image_id"] == image_id}
    elif tags == "random":
        tag_of = dict(zip(body_ids, _random_mcfo_tags(len(body_ids), np.random.default_rng(seed))))
    else:
        raise ValueError("tags must be 'v1' or 'random'")
    neurons = _load_neurons(bundle, body_ids, radius_scale)
    lo, hi, shape_xyz = _v1_box(bundle, row, image_id, voxel_um)

    channels, channel_tifs, log = [], [], ""
    for tag in TAGS:
        members = [b for b in body_ids if tag in tag_of.get(b, "").split("+")]
        nodes = merge([neurons[b] for b in members]) if members else np.zeros(0, dtype=neurons[body_ids[0]].dtype)
        vol, tif, lg = _render_in_box(
            nodes, f"{image_id}_{tag}", out_dir, lo, hi, shape_xyz, voxel_um,
            f"{image_id} channel {tag}: neurons {' '.join(members) or '(none)'}", **swc2img_kw)
        channels.append(vol)
        log += lg
        channel_tifs.append(tif)
    rgb = np.stack(channels)
    del channels

    tif = out_dir / f"{image_id}.swc2img_rgb.tif"
    # ImageJ hyperstacks are stored Z, C, Y, X; open in Fiji as a 3-channel composite.
    tifffile.imwrite(tif, np.moveaxis(rgb, 0, 1), imagej=True, resolution=(1 / voxel_um, 1 / voxel_um),
                     metadata={"spacing": voxel_um, "unit": "micron", "axes": "ZCYX", "mode": "composite"})
    for t in channel_tifs:  # replaced by the combined TIFF
        t.unlink()
    meta = {"image_id": image_id, "body_ids": body_ids, "overlap_mode": row["overlap_mode"], "crop": True,
            "tags_source": tags, "seed": seed if tags == "random" else None,
            "channels": list(TAGS), "tags": tag_of, "voxel_um": voxel_um, "radius_scale": radius_scale,
            "crop_lo_um_xyz": lo.tolist(), "crop_hi_um_xyz": hi.tolist(), "shape_czyx": list(rgb.shape),
            "swc2img": {k: v for k, v in swc2img_kw.items() if k != "fiji_dir"}}
    json.dump(meta, open(out_dir / f"{image_id}.swc2img_rgb.json", "w"), indent=1)

    from PIL import Image
    mip_rgb = np.stack([_norm(c.max(axis=0)) for c in rgb], axis=-1)
    mip = out_dir / f"{image_id}_swc2img_rgb_mip.png"
    Image.fromarray((mip_rgb * 255).astype(np.uint8)).save(mip)

    v1_zarr = bundle / "examples" / image_id / f"{image_id}.zarr"
    if v1_zarr.exists():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 7.4), constrained_layout=True)
        ax[0].imshow(v1_raw_mip_rgb(v1_zarr))
        ax[0].set_title("synth_MCFO 1k_v1 (mcfo_sim), MIP")
        ax[1].imshow(mip_rgb)
        ax[1].set_title(f"SWC2IMG, MCFO colours ({'v1 tags' if tags == 'v1' else f'random tags, seed {seed}'}), MIP")
        for a in ax:
            a.set_axis_off()
        fig.suptitle(f"{image_id} ({row['overlap_mode']}, {len(body_ids)} neurons); channels HA, V5, FLAG = R, G, B")
        fig.savefig(out_dir / f"{image_id}_rgb_compare.png", dpi=110)
        plt.close(fig)
    return RenderResult(rgb, tif, mip, log, meta)
