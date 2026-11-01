from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from imaging_helpers_hpc.paths import AnalysisOutputPaths
from imaging_helpers_hpc.processing import axis_rotation

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def gen_mip(raw, axis):
    """Max-intensity projection of RGB channels along the stack axis. BiaPy, axis=0. Fisby, axis=1."""
    mip = raw.astype(np.float32).max(axis=axis)  # (C, Y, X)
    mip = (mip - mip.min()) / (np.ptp(mip) + 1e-8)
    return np.moveaxis(mip, 0, -1)

def gen_biapy_mip_4panel(raw, probs, instances, analysis_output_paths: AnalysisOutputPaths, title_prefix="", axis=0, ):
    """4-panel max-intensity projection along axis (default Z)."""
    prob_mip = probs.max(axis=axis)
    inst_mip = instances.max(axis=axis)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(gen_mip(raw, axis=axis))
    axes[0].set_title(f"{title_prefix}raw RGB MIP")
    axes[0].axis("off")

    axes[1].imshow(prob_mip[0], cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("F channel MIP")
    axes[1].axis("off")

    axes[2].imshow(prob_mip[1], cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("C channel MIP")
    axes[2].axis("off")

    axes[3].imshow(inst_mip, cmap="nipy_spectral")
    axes[3].set_title(f"instances MIP ({len(np.unique(inst_mip)) - 1} labels)")
    axes[3].axis("off")

    plt.suptitle(f"Maximum intensity projection (axis={axis}, depth={raw.shape[axis]} slices)", y=1.02)
    plt.tight_layout()
    # plt.savefig(f"fisbe/biapy/results/3d_instance_segmentation/results/3d_instance_segmentation_1/mip_{title_prefix}.png")
    logger.info(f"Saving to: {analysis_output_paths.output_images / f'mip_{title_prefix}.png'}")
    plt.savefig(analysis_output_paths.output_images / f"mip_{title_prefix}.png")


def _zcyx_to_zyx_mip(vol: np.ndarray, axis: int = 0) -> np.ndarray:
    """MIP a (Z, C, Y, X) or (Z, Y, X) volume to 2D (Y, X)."""
    vol = np.asarray(vol, dtype=np.float32)
    if vol.ndim == 4:
        # collapse channels first (argmax already applied by caller if needed)
        if vol.shape[1] == 1:
            vol = vol[:, 0]
        else:
            vol = vol.max(axis=1)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D ZYX after channel collapse, got {vol.shape}")
    return vol.max(axis=axis)


def gen_ppp_snapshot_mip(
    raw: np.ndarray | None,
    pred_numinst: np.ndarray | None,
    gt_numinst: np.ndarray | None,
    pred_affs: np.ndarray | None,
    gt_affs: np.ndarray | None,
    analysis_output_paths: AnalysisOutputPaths,
    title_prefix: str = "",
    axis: int = 0,
    output_path: Path | None = None,
) -> None:
    """4-panel MIP QA for a PPP training snapshot HDF.

    Arrays are (Z, C, Y, X) as returned by load_any_volume(..., reduce_wide_channels=False).
    Panels: raw RGB | pred_numinst argmax | gt_numinst | mean |pred_affs-gt_affs| L1.
    Missing arrays leave a blank panel with a warning.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 0: raw RGB MIP
    if raw is not None:
        axes[0].imshow(gen_mip(raw, axis=axis))
        axes[0].set_title(f"{title_prefix}raw RGB MIP")
    else:
        logger.warning("volumes/raw missing; leaving panel blank")
        axes[0].set_title("raw (missing)")
    axes[0].axis("off")

    # 1: pred_numinst argmax then MIP
    if pred_numinst is not None:
        # (Z, C, Y, X) -> argmax over C -> (Z, Y, X)
        labels = np.argmax(pred_numinst, axis=1)
        mip = labels.max(axis=axis)
        axes[1].imshow(mip, cmap="nipy_spectral")
        axes[1].set_title("pred_numinst argmax MIP")
    else:
        logger.warning("volumes/pred_numinst missing; leaving panel blank")
        axes[1].set_title("pred_numinst (missing)")
    axes[1].axis("off")

    # 2: gt_numinst MIP
    if gt_numinst is not None:
        mip = _zcyx_to_zyx_mip(gt_numinst, axis=axis)
        axes[2].imshow(mip, cmap="nipy_spectral")
        axes[2].set_title("gt_numinst MIP")
    else:
        logger.warning("volumes/gt_numinst missing; leaving panel blank")
        axes[2].set_title("gt_numinst (missing)")
    axes[2].axis("off")

    # 3: affinity L1 error MIP
    if pred_affs is not None and gt_affs is not None:
        if pred_affs.shape != gt_affs.shape:
            logger.warning(
                "pred_affs %s vs gt_affs %s shape mismatch; blanking aff panel",
                pred_affs.shape,
                gt_affs.shape,
            )
            axes[3].set_title("aff L1 (shape mismatch)")
        else:
            # mean over channels of |pred - gt|, then MIP over Z
            err = np.mean(
                np.abs(
                    pred_affs.astype(np.float32) - gt_affs.astype(np.float32)
                ),
                axis=1,
            )
            err_mip = err.max(axis=axis)
            im = axes[3].imshow(err_mip, cmap="magma")
            axes[3].set_title("aff L1 error MIP")
            fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    else:
        logger.warning(
            "volumes/pred_affs and/or volumes/gt_affs missing; leaving panel blank"
        )
        axes[3].set_title("aff L1 (missing)")
    axes[3].axis("off")

    depth = raw.shape[axis] if raw is not None else "?"
    plt.suptitle(
        f"PPP snapshot MIP (axis={axis}, depth={depth})",
        y=1.02,
    )
    plt.tight_layout()
    if output_path is None:
        out_name = f"ppp_snapshot_{title_prefix}".rstrip("_") + ".png"
        out_path = analysis_output_paths.output_images / out_name
    else:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving to: %s", out_path)
    plt.savefig(out_path)
    plt.close(fig)


def gen_basic_mip(
        input_np:np.ndarray, 
        output_file_name:str, 
        analysis_output_paths: AnalysisOutputPaths,
        axis:int=1,
        output_path: Path | None = None,
        dpi: int = 800,
):
    logger.info(f"gen_basic_mip:")
    logger.info(f"\toutput_file_name: {output_file_name}")
    logger.info(f"\tGEN MIP - Before Shape: {input_np.shape}")
    # MIP calculation
    # mip = input_np.max(axis=1)
    # mip = (mip - mip.min()) / (np.ptp(mip) + 1e-8)
    # rgb = np.moveaxis(mip, 0, -1) # For matplotlib plotting
    rgb = gen_mip(input_np, axis=axis) # For matplotlib plotting
    logger.info(f"\tGEN MIP - After Shape: {rgb.shape}")

    fig, axes = plt.subplots(1, 1, figsize=(8, 8), dpi=dpi)
    axes.imshow(rgb)
    axes.set_title(f"MIP of Zarr Image {output_file_name}")
    axes.axis("off")
    if output_path is None:
        out_path = analysis_output_paths.output_images / f"{output_file_name}.png"
    else:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    logger.info(f"\tSaving to: {out_path}")
    plt.close(fig)

def _hillshade(z: np.ndarray, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    """Simple Lambertian hillshade of a 2D height field, values in [0, 1]."""
    dy, dx = np.gradient(z.astype(np.float64))
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = np.radians(azimuth_deg)
    alt = np.radians(altitude_deg)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shade = (shade - shade.min()) / (np.ptp(shade) + 1e-8)
    return shade.astype(np.float32)


def gen_topographic_projection(
    input_np: np.ndarray,
    output_file_name: str,
    analysis_output_paths: AnalysisOutputPaths,
    axis: int = 1,
    output_path: Path | None = None,
):
    """
    Min-intensity projection of a watershed topographic surface with terrain colormap
    and light hillshade. Prefer this over MIP: max along Z smears ridges into blocks.
    Accepts (C, Z, Y, X) or (Z, Y, X).
    """
    logger.info("gen_topographic_projection:")
    logger.info(f"\toutput_file_name: {output_file_name}")
    logger.info(f"\tBefore Shape: {input_np.shape}")

    vol = np.asarray(input_np, dtype=np.float32)
    if vol.ndim == 4:
        # (C, Z, Y, X) — drop channel; map CZYX axis index onto ZYX
        vol = vol[0]
        z_axis = max(axis - 1, 0)
    elif vol.ndim == 3:
        # (Z, Y, X): axis=1 from CZYX callers means Z at 0; axis=0 also Z
        z_axis = 0 if axis in (0, 1) else axis
    else:
        raise ValueError(f"Expected 3D or 4D volume, got shape {vol.shape}")

    topo2d = vol.min(axis=z_axis)
    logger.info(f"\tMinIP Shape: {topo2d.shape}")

    lo, hi = np.percentile(topo2d, [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(topo2d.min()), float(topo2d.max())
        if hi <= lo:
            hi = lo + 1e-8
    topo_norm = np.clip((topo2d - lo) / (hi - lo), 0.0, 1.0)
    shade = _hillshade(topo2d)

    terrain_rgb = plt.get_cmap("terrain")(topo_norm)[..., :3]
    # Blend hillshade so relief reads without washing out height color
    shaded = terrain_rgb * (0.35 + 0.65 * shade[..., None])
    shaded = np.clip(shaded, 0.0, 1.0)

    fig, axes = plt.subplots(1, 1, figsize=(8, 8), dpi=800)
    axes.imshow(shaded, vmin=0, vmax=1)
    # Colorbar reflects the underlying height (percentile-clipped), not the shade blend
    sm = plt.cm.ScalarMappable(cmap="terrain", norm=plt.Normalize(vmin=lo, vmax=hi))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("height (MinIP)")
    axes.set_title(f"Topographic MinIP {output_file_name}")
    axes.axis("off")

    if output_path is None:
        out_path = analysis_output_paths.output_images / f"{output_file_name}.png"
    else:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    logger.info(f"\tSaving to: {out_path}")
    plt.close(fig)


def gen_instance_projection(
    input_np: np.ndarray,
    output_file_name: str,
    analysis_output_paths: AnalysisOutputPaths,
    axis: int = 1,
    output_path: Path | None = None,
    dpi: int = 800,
):
    """
    Colorized MIP of FISBe gt_instances (CZYX overlapping instance channels).
    Expects a CZYX image.
    """
    logger.info(f"gen_gt_projection:")
    logger.info(f"\tinput_np Shape: {input_np.shape}")
    # MIP along Z per channel, then collapse overlapping channels by max label id
    mip = np.max(input_np, axis=axis)  # -> (C, Y, X)
    logger.info(f"\tmip Shape: {mip.shape}")
    labels = mip.max(axis=0)  # (Y, X)
    logger.info(f"\tlabels Shape: {labels.shape}")

    # Checking valid input
    sample_element = labels.tolist()[0][0]
    assert type(sample_element) == int, f"Element {sample_element}. Found type {type(sample_element)}. This is likeley NOT an instance map of an image"

    # Color Assignment and Final Image Rendering
    rgb = np.zeros((*labels.shape, 3), dtype=np.float32)
    for lab in np.unique(labels):
        if lab == 0:
            continue
        color = np.random.randint(72, 255, 3).astype(np.float32)
        rgb[labels == lab] = color
    rgb /= 255.0
    logger.debug(f"\trgb Shape: {rgb.shape}")

    # Plotting
    fig, axes = plt.subplots(1, 1, figsize=(8, 8), dpi=dpi)
    axes.imshow(rgb)
    logger.debug(f"\toutput_file_name: {output_file_name}")
    output_file_name = output_file_name.replace('.zarr.tif', '')
    axes.set_title(f"GT instance MIP {output_file_name}")
    axes.axis("off")
    if output_path is None:
        rand_id = np.random.randint(0, len(np.unique(labels)))
        out_path = analysis_output_paths.output_images / f"{output_file_name}_id{rand_id}_instances.png"
    else:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"\tSaving to: {out_path}")
    plt.savefig(out_path)
    plt.close(fig)


def gen_scalar_mip(
    input_np: np.ndarray,
    output_file_name: str,
    analysis_output_paths: AnalysisOutputPaths,
    axis: int = 0,
    output_path: Path | None = None,
    dpi: int = 800,
) -> None:
    """Grayscale MIP of a ZYX volume along ``axis`` (default Z)."""
    vol = np.asarray(input_np)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D ZYX volume, got shape {vol.shape}")
    logger.info("gen_scalar_mip: %s shape %s axis %s", output_file_name, vol.shape, axis)
    mip = vol.max(axis=axis).astype(np.float32)
    mip = (mip - mip.min()) / (np.ptp(mip) + 1e-8)

    fig, axes = plt.subplots(1, 1, figsize=(8, 8), dpi=dpi)
    axes.imshow(mip, cmap="gray", vmin=0, vmax=1)
    axes.set_title(f"MIP {output_file_name}")
    axes.axis("off")
    if output_path is None:
        out_path = analysis_output_paths.output_images / f"{output_file_name}.png"
    else:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    logger.info("Saving to: %s", out_path)
    plt.close(fig)


def gen_rotations_and_projections(
    input_np: np.ndarray,
    output_file_name: str,
    analysis_output_paths: AnalysisOutputPaths,
    volume: str,
    rand_axis_int: int,
    k_rotations: int,
    axis: int = 1,
):
    """Generates rotated projections of MIPs and GT instances, using the same rotation params"""
    # match volume:
    #     case 'raw':
    #         gen_basic_mip(input_np, f"base_rotated_mip{output_file_name}", analysis_output_paths)
    #     case 'gt_instance':
    #         gen_gt_projection(input_np, f"base_rotated_{output_file_name}", analysis_output_paths, axis)
    rotated_image = axis_rotation(input_np, rand_axis_int, k_rotations)

    if volume == "raw":
        gen_basic_mip(rotated_image, f"rotated_{output_file_name}", analysis_output_paths)
    elif volume == "gt_instance":
        gen_instance_projection(rotated_image, f"rotated_{output_file_name}", analysis_output_paths, axis)

def gen_basic_image():
    pass