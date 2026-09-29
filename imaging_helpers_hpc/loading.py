import os
import re
import logging
from pathlib import Path

import zarr
import tifffile
import numpy as np

from imaging_helpers_hpc.paths import BiapyDataPaths, FisbeDataPaths

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_sample_stem(path: Path) -> str:
    name = path.name
    for suffix in (".zarr.tif", ".zarr.tiff", ".tif", ".tiff"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem

def _xyz_to_zcyx(arr: np.ndarray) -> np.ndarray:
    """Convert 3D XYZ volume to BiaPy-style (Z, 1, Y, X) for gen_basic_mip."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D XYZ volume, got shape {arr.shape}")
    zyx = np.transpose(arr, (2, 1, 0))
    return zyx[:, np.newaxis, :, :]


def _load_nrrd(full_path: Path) -> np.ndarray:
    """Load a 3D NRRD volume and normalize to BiaPy-style (Z, C, Y, X).

    Assumes NRRD axis order XYZ (common for VFB / space-direction headers),
    then returns (Z, 1, Y, X) so gen_basic_mip(..., axis=0) works.
    """
    try:
        import nrrd
    except ImportError as exc:
        raise ImportError(
            "Reading NRRD requires pynrrd. Install it or run inside the ppp overlay."
        ) from exc

    data, _header = nrrd.read(str(full_path))
    arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3D NRRD volume, got shape {arr.shape} from {full_path}"
        )
    return _xyz_to_zcyx(arr)


def _load_npy(full_path: Path) -> np.ndarray:
    """Load a NumPy .npy volume for any-MIP.

    3D arrays are treated as XYZ (CloudVolume / MANC crops) and converted to
    (Z, 1, Y, X). 4D arrays are assumed already (Z, C, Y, X) and returned as-is.
    """
    arr = np.asarray(np.load(full_path))
    if arr.ndim == 3:
        return _xyz_to_zcyx(arr)
    if arr.ndim == 4:
        return arr
    raise ValueError(
        f"Expected 3D or 4D NPY volume, got shape {arr.shape} from {full_path}"
    )


def load_any_volume(full_path) -> np.ndarray:
    """Load a TIFF, NRRD, or NPY volume for any-MIP.

    TIFF: returned as tifffile reads it (typically Z, C, Y, X).
    NRRD / 3D NPY: XYZ normalized to (Z, 1, Y, X).
    4D NPY: returned as-is (Z, C, Y, X).
    """
    path = Path(full_path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        return tifffile.imread(path)
    if suffix in (".nrrd", ".nhdr"):
        return _load_nrrd(path)
    if suffix == ".npy":
        return _load_npy(path)
    raise ValueError(
        f"Unsupported input type {suffix!r} for {path}; "
        f"expected .tif/.tiff/.nrrd/.nhdr/.npy"
    )


def load_any_tif(full_path):
    """Alias for load_any_volume (TIFF, NRRD, and NPY)."""
    return load_any_volume(full_path)

def load_biapy_test_sample(sample: str, biapy_paths:BiapyDataPaths):
    """Load raw test volume plus matching BiaPy per_image and instance outputs."""
    logger.info(f"\tload_biapy_test_sample:")
    raw_path = next(biapy_paths.BIAPY_TEST_RAW_DIR.glob(f"{sample}.zarr.tiff"))
    prob_path = biapy_paths.per_image / f"{sample}.zarr.tif"
    inst_path = biapy_paths.per_image_instances / f"{sample}.zarr.tif"

    logger.debug(f"\tModel Result Paths: {raw_path}")
    logger.debug(f"\t\traw_path: {raw_path}")
    logger.debug(f"\t\tprob_path: {prob_path}")
    logger.debug(f"\t\tinst_path: {inst_path}")

    raw = tifffile.imread(raw_path)          # (Z, C, Y, X)
    probs = tifffile.imread(prob_path)       # (Z, 2, Y, X)
    instances = tifffile.imread(inst_path)   # (Z, Y, X)

    logger.info(f"\tLoaded Sample {sample}")
    logger.info(f"\t\traw: {raw.shape}, {raw.dtype}")
    logger.info(f"\t\tprobs: {probs.shape}, {probs.dtype}")
    logger.info(f"\t\tinstances: {instances.shape}, {instances.dtype}")
    logger.info(f"\t\tlabels: {len(np.unique(instances))}")

    return raw, probs, instances

def load_fisbe_completely(sample: str, fisbe_data_paths:FisbeDataPaths, split:str = 'train'):
    logger.info(f"load_fisbe_completely:")
    loaded_sample = zarr.open_group(fisbe_data_paths.paths[split] / f"{sample}.zarr")

    raw = loaded_sample['volumes/raw']
    gt_instances = loaded_sample['volumes/gt_instances']

    raw = np.array(raw)
    gt_instances = np.array(gt_instances)

    logger.info(f"\traw shape: {raw.shape}")
    logger.info(f"\tgt_instances shape: {gt_instances.shape}")
    logger.info(f"\t\tall labels: {len(np.unique(gt_instances))}")
    logger.info(f"\t\tlabel-idx 0: {len(np.unique(gt_instances[0]))}")
    logger.info(f"\t\tlabel-idx 1: {len(np.unique(gt_instances[1]))}")
    logger.info(f"\t\tlabel-idx-value 0: {np.unique(gt_instances[0])}")
    logger.info(f"\t\tlabel-idx-value 1: {np.unique(gt_instances[1])}")

    return raw, gt_instances

