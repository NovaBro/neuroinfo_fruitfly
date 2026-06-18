"""BiaPy 3D output utilities: load TIFF volumes, build MIPs, and match FISBe raw data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import tifffile
import zarr


def _repo_root() -> Path:
    root = Path(__file__).resolve().parent
    if root.name == "scripts":
        root = root.parent
    if root.name == "ipynb":
        root = root.parent
    return root


REPO_ROOT = _repo_root()

# --- paths -----------------------------------------------------------------

BIAPY_RESULT_ROOT = Path(
    os.environ.get(
        "BIAPY_RESULT_ROOT",
        REPO_ROOT
        / "BiaPy/results/3d_instance_segmentation/results/3d_instance_segmentation_1",
    )
)
BIAPY_PER_IMAGE = BIAPY_RESULT_ROOT / "per_image"
BIAPY_PER_IMAGE_INSTANCES = BIAPY_RESULT_ROOT / "per_image_instances"
BIAPY_MIP_DIR = BIAPY_RESULT_ROOT / "mips"

TEST_RAW_DIR = Path(
    os.environ.get("BIAPY_TEST_RAW_DIR", REPO_ROOT / "fisbe/biapy/test/raw")
)
FISBE_ROOT = Path(os.environ.get("FISBE_ROOT", REPO_ROOT / "fisbe/completely"))
FISBE_SPLITS = ("train", "val", "test")

ZARR_TIFF_SUFFIXES = (".zarr.tif", ".zarr.tiff")
TIFF_SUFFIXES = ZARR_TIFF_SUFFIXES + (".tif", ".tiff")

ProjectionMethod = Literal["max", "min", "mean", "sum", "median"]
VolumeLayout = Literal["zcyx", "czyx"]


# --- naming / discovery ----------------------------------------------------

def sample_stem(path: Path) -> str:
    """Strip BiaPy-style TIFF suffixes and return the sample stem."""
    name = path.name
    for suffix in TIFF_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def resolve_tiff_path(directory: Path, stem: str) -> Path:
    """Return the first existing TIFF for ``stem`` under ``directory``."""
    for suffix in TIFF_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No TIFF for {stem!r} in {directory}")


def list_biapy_samples() -> list[str]:
    """List sample stems that have BiaPy per_image outputs."""
    return sorted(sample_stem(p) for p in BIAPY_PER_IMAGE.glob("*.tif*"))


def has_biapy_instances(stem: str) -> bool:
    """Return True if BiaPy per_image_instances output exists for ``stem``."""
    try:
        resolve_tiff_path(BIAPY_PER_IMAGE_INSTANCES, stem)
        return True
    except FileNotFoundError:
        return False


def get_biapy_instances_meta(stem: str) -> dict | None:
    """Return shape/dtype metadata for predicted instances, or None if missing."""
    if not has_biapy_instances(stem):
        return None
    path = resolve_tiff_path(BIAPY_PER_IMAGE_INSTANCES, stem)
    with tifffile.TiffFile(path) as tif:
        page = tif.series[0] if tif.series else tif.pages[0]
        shape = tuple(int(s) for s in page.shape)
        dtype = str(page.dtype)
    return {
        "shape": list(shape),
        "dtype": dtype,
    }


def find_fisbe_zarr(stem: str) -> tuple[Path, str]:
    """Locate ``{stem}.zarr`` in the FISBe dataset and return path + split."""
    for split in FISBE_SPLITS:
        zarr_path = FISBE_ROOT / split / f"{stem}.zarr"
        if zarr_path.is_dir():
            return zarr_path, split
    raise FileNotFoundError(
        f"No FISBe zarr for {stem!r} under {FISBE_ROOT}/{{{','.join(FISBE_SPLITS)}}}"
    )


# --- loading ---------------------------------------------------------------

def load_zarr_tiff(path: Path | str) -> np.ndarray:
    """Load a BiaPy OME-TIFF volume (``.zarr.tif`` / ``.zarr.tiff``)."""
    return tifffile.imread(path)


def load_biapy_per_image(stem: str) -> np.ndarray:
    """Load per_image probabilities for ``stem``, shape ``(Z, 2, Y, X)``."""
    return load_zarr_tiff(resolve_tiff_path(BIAPY_PER_IMAGE, stem))


def load_biapy_per_image_instances(stem: str) -> np.ndarray:
    """Load per_image_instances labels for ``stem``, shape ``(Z, Y, X)``."""
    return load_zarr_tiff(resolve_tiff_path(BIAPY_PER_IMAGE_INSTANCES, stem))


def load_biapy_test_raw(stem: str) -> np.ndarray:
    """Load the BiaPy test TIFF raw volume for ``stem``, shape ``(Z, C, Y, X)``."""
    return load_zarr_tiff(resolve_tiff_path(TEST_RAW_DIR, stem))


def load_fisbe_raw(
    stem: str,
    *,
    split: str | None = None,
    layout: VolumeLayout = "zcyx",
    zarr_key: str = "volumes/raw",
) -> np.ndarray:
    """Load the matching raw volume from the FISBe zarr dataset.

  By default returns ``(Z, C, Y, X)`` to match BiaPy TIFF layout. Pass
  ``layout='czyx'`` to keep the native FISBe ``(C, Z, Y, X)`` order.
  """
    if split is None:
        zarr_path, _ = find_fisbe_zarr(stem)
    else:
        zarr_path = FISBE_ROOT / split / f"{stem}.zarr"
        if not zarr_path.is_dir():
            raise FileNotFoundError(f"Missing FISBe zarr: {zarr_path}")

    raw = np.array(zarr.open(zarr_path, mode="r", path=zarr_key))
    if raw.ndim != 4:
        raise ValueError(f"{zarr_path.name}: expected 4D raw array, got {raw.shape}")

    if layout == "zcyx":
        return raw.transpose(1, 0, 2, 3)
    if layout == "czyx":
        return raw
    raise ValueError(f"Unknown layout {layout!r}; use 'zcyx' or 'czyx'.")


def load_biapy_test_sample(stem: str):
    """Load BiaPy test raw TIFF plus matching per_image and instance outputs."""
    return (
        load_biapy_test_raw(stem),
        load_biapy_per_image(stem),
        load_biapy_per_image_instances(stem),
    )


# --- MIP -------------------------------------------------------------------

def mip_3d(
    volume: np.ndarray,
    axis: int = 0,
    method: ProjectionMethod = "max",
) -> np.ndarray:
    """Maximum (or other) intensity projection along ``axis``."""
    if method == "max":
        return volume.max(axis=axis)
    if method == "min":
        return volume.min(axis=axis)
    if method == "mean":
        return volume.mean(axis=axis)
    if method == "sum":
        sum_dtype = np.uint32 if volume.dtype.kind in "ub" else np.float64
        return np.sum(volume, axis=axis, dtype=sum_dtype)
    if method == "median":
        return np.median(volume, axis=axis)
    raise ValueError(f"Unknown projection method {method!r}")


def _mip_to_uint8(mip: np.ndarray) -> np.ndarray:
    """Contrast-stretch a 2D projection to uint8 for PNG export."""
    data = mip.astype(np.float32)
    vmin, vmax = float(data.min()), float(data.max())
    if vmax > vmin:
        data = (data - vmin) / (vmax - vmin)
    else:
        data = np.zeros_like(data)
    return (data * 255).astype(np.uint8)


def save_mip(
    mip: np.ndarray,
    out_path: Path | str,
    *,
    save_png: bool = True,
) -> Path:
    """Write a 2D MIP as TIFF and optionally as an 8-bit PNG."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out_path, mip)

    if save_png:
        png_path = out_path.with_suffix(".png")
        if mip.ndim == 2:
            tifffile.imwrite(png_path, _mip_to_uint8(mip))
        elif mip.ndim == 3 and mip.shape[-1] in (3, 4):
            tifffile.imwrite(png_path, mip.astype(np.uint8))
        else:
            tifffile.imwrite(png_path, _mip_to_uint8(mip[0] if mip.ndim == 3 else mip))

    return out_path


def create_mip_file(
    volume: np.ndarray,
    out_path: Path | str,
    *,
    axis: int = 0,
    method: ProjectionMethod = "max",
    save_png: bool = True,
) -> Path:
    """Project a 3D volume and save the result as TIFF (+ optional PNG)."""
    mip = mip_3d(volume, axis=axis, method=method)
    return save_mip(mip, out_path, save_png=save_png)


def create_biapy_mips(
    stem: str,
    out_dir: Path | str | None = None,
    *,
    axis: int = 0,
    method: ProjectionMethod = "max",
) -> dict[str, Path]:
    """Create MIP files for test raw, per_image, and per_image_instances."""
    out_dir = Path(out_dir or BIAPY_MIP_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_biapy_test_raw(stem)
    probs = load_biapy_per_image(stem)
    instances = load_biapy_per_image_instances(stem)

    outputs = {
        "raw_rgb": save_mip(
            mip_rgb(raw, axis=axis),
            out_dir / f"{stem}_raw_rgb_mip.tif",
        ),
        "per_image_F": create_mip_file(
            probs[:, 0],
            out_dir / f"{stem}_per_image_F_mip.tif",
            axis=axis,
            method=method,
        ),
        "per_image_C": create_mip_file(
            probs[:, 1],
            out_dir / f"{stem}_per_image_C_mip.tif",
            axis=axis,
            method=method,
        ),
        "per_image_instances": create_mip_file(
            instances,
            out_dir / f"{stem}_per_image_instances_mip.tif",
            axis=axis,
            method=method,
        ),
    }
    return outputs


# --- viewing ---------------------------------------------------------------

def mip_rgb(raw: np.ndarray, axis: int = 0) -> np.ndarray:
    """Max-intensity projection of RGB channels, returned as ``(Y, X, 3)`` float in [0, 1]."""
    mip = raw.astype(np.float32).max(axis=axis)  # (C, Y, X)
    mip = (mip - mip.min()) / (np.ptp(mip) + 1e-8)
    return np.moveaxis(mip, 0, -1)


def show_biapy_mip(
    raw: np.ndarray,
    probs: np.ndarray,
    instances: np.ndarray,
    title_prefix: str = "",
    axis: int = 0,
    save_path: Path | str | None = None,
) -> None:
    """Four-panel max-intensity projection along ``axis`` (default Z)."""
    import matplotlib.pyplot as plt

    prob_mip = probs.max(axis=axis)
    inst_mip = instances.max(axis=axis)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(mip_rgb(raw, axis=axis))
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

    plt.suptitle(
        f"Maximum intensity projection (axis={axis}, depth={raw.shape[axis]} slices)",
        y=1.02,
    )
    plt.tight_layout()

    if save_path is None:
        save_path = BIAPY_RESULT_ROOT / f"mip_{title_prefix}.png"
    plt.savefig(save_path)
    plt.show()


# --- demo ------------------------------------------------------------------

if __name__ == "__main__":
    sample_names = list_biapy_samples()
    print(f"{len(sample_names)} test volumes with BiaPy outputs:")
    for name in sample_names:
        print(" ", name)

    if not sample_names:
        raise SystemExit("No BiaPy per_image outputs found.")

    sample = sample_names[0]
    raw_vol, prob_vol, inst_vol = load_biapy_test_sample(sample)
    fisbe_path, fisbe_split = find_fisbe_zarr(sample)
    fisbe_raw = load_fisbe_raw(sample, split=fisbe_split)

    print(f"\nSelected: {sample}")
    print("biapy test raw:", raw_vol.shape, raw_vol.dtype)
    print("per_image:", prob_vol.shape, prob_vol.dtype)
    print("per_image_instances:", inst_vol.shape, inst_vol.dtype, "labels:", len(np.unique(inst_vol)))
    print(f"fisbe raw ({fisbe_split}):", fisbe_raw.shape, fisbe_raw.dtype, "from", fisbe_path)

    mip_paths = create_biapy_mips(sample)
    print("\nWrote MIP files:")
    for label, path in mip_paths.items():
        print(f"  {label}: {path}")

    show_biapy_mip(raw_vol, prob_vol, inst_vol, title_prefix=f"{sample} | ")
