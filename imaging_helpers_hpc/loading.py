from __future__ import annotations

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


def list_hdf_volume_datasets(path: Path) -> list[str]:
    """Return dataset paths under /volumes/ in an HDF5 file."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Reading HDF5 requires h5py. Install it or run inside the ppp overlay."
        ) from exc

    keys: list[str] = []
    with h5py.File(path, "r") as f:
        if "volumes" not in f:
            return keys
        vol = f["volumes"]
        for name in vol.keys():
            if isinstance(vol[name], h5py.Dataset):
                keys.append(f"volumes/{name}")
    return keys


def _squeeze_leading_batch(arr: np.ndarray) -> np.ndarray:
    """Drop a leading size-1 batch axis (PPP snapshots are (1, ...))."""
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _normalize_hdf_volume(
    arr: np.ndarray,
    *,
    reduce_wide_channels: bool = True,
) -> np.ndarray:
    """Normalize PPP/gunpowder HDF arrays to (Z, C, Y, X) for gen_basic_mip.

    After squeezing a leading batch dim, 4D arrays are treated as (C, Z, Y, X)
    and transposed to (Z, C, Y, X). 3D arrays become (Z, 1, Y, X).
    When C > 3 and reduce_wide_channels is True (affinity stacks), channels
    are reduced with max so imshow stays valid.
    """
    arr = np.asarray(arr)
    arr = _squeeze_leading_batch(arr)

    if arr.ndim == 3:
        return arr[:, np.newaxis, :, :]

    if arr.ndim == 4:
        # PPP snapshots store (C, Z, Y, X)
        zcyx = np.transpose(arr, (1, 0, 2, 3))
        n_chan = zcyx.shape[1]
        if reduce_wide_channels and n_chan > 3:
            logger.info(
                "HDF volume has C=%d (>3); reducing channels with max for MIP",
                n_chan,
            )
            zcyx = zcyx.max(axis=1, keepdims=True)
        return zcyx

    raise ValueError(
        f"Expected 3D or 4D HDF volume after batch squeeze, got shape {arr.shape}"
    )


def _resolve_hdf_dataset(path: Path, dataset: str | None) -> str:
    keys = list_hdf_volume_datasets(path)
    if dataset is not None:
        return dataset
    if len(keys) == 1:
        return keys[0]
    listed = ", ".join(keys) if keys else "(none under /volumes/)"
    raise ValueError(
        f"HDF file {path} needs --dataset; available: {listed}"
    )


def _load_hdf(
    path: Path,
    dataset: str | None = None,
    *,
    reduce_wide_channels: bool = True,
) -> np.ndarray:
    """Load one HDF5 dataset and normalize to (Z, C, Y, X)."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Reading HDF5 requires h5py. Install it or run inside the ppp overlay."
        ) from exc

    key = _resolve_hdf_dataset(path, dataset)
    with h5py.File(path, "r") as f:
        if key not in f:
            available = list_hdf_volume_datasets(path)
            raise KeyError(
                f"Dataset {key!r} not in {path}; available: "
                f"{', '.join(available) if available else '(none)'}"
            )
        arr = np.asarray(f[key])
    logger.info("Loaded HDF %s [%s] shape=%s dtype=%s", path, key, arr.shape, arr.dtype)
    return _normalize_hdf_volume(arr, reduce_wide_channels=reduce_wide_channels)


def load_any_volume(
    full_path,
    dataset: str | None = None,
    *,
    reduce_wide_channels: bool = True,
) -> np.ndarray:
    """Load a TIFF, NRRD, NPY, or HDF5 volume for any-MIP.

    TIFF: returned as tifffile reads it (typically Z, C, Y, X).
    NRRD / 3D NPY: XYZ normalized to (Z, 1, Y, X).
    4D NPY: returned as-is (Z, C, Y, X).
    HDF (.hdf/.h5/.hdf5): requires dataset (e.g. volumes/raw) unless the file
    has exactly one dataset under /volumes/. Normalized to (Z, C, Y, X);
    affinity stacks with C>3 are max-reduced unless reduce_wide_channels=False.
    """
    path = Path(full_path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        return tifffile.imread(path)
    if suffix in (".nrrd", ".nhdr"):
        return _load_nrrd(path)
    if suffix == ".npy":
        return _load_npy(path)
    if suffix in (".hdf", ".h5", ".hdf5"):
        return _load_hdf(
            path, dataset, reduce_wide_channels=reduce_wide_channels
        )
    raise ValueError(
        f"Unsupported input type {suffix!r} for {path}; "
        f"expected .tif/.tiff/.nrrd/.nhdr/.npy/.hdf/.h5/.hdf5"
    )


def load_any_tif(full_path, dataset: str | None = None):
    """Alias for load_any_volume (TIFF, NRRD, NPY, and HDF)."""
    return load_any_volume(full_path, dataset=dataset)

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


SYNTH_MCFO_ROOT = Path("/scratch/vh2308/data/synth_MCFO")
SYNTH_MOUNT_ROOT = Path("/synth_MCFO")


class SynthMountMissing(RuntimeError):
    """Source SquashFS is not mounted at /synth_MCFO/<dataset>/samples."""


def synth_samples_dir(dataset: str) -> Path:
    return SYNTH_MOUNT_ROOT / dataset / "samples"


def require_synth_mount(dataset: str) -> None:
    """Raise when raw/gt symlinks into this dataset cannot resolve.

    The mount command is the sbatch that overlays the matching .sqf.
    """
    samples = synth_samples_dir(dataset)
    if samples.is_dir():
        return
    sqf = SYNTH_MCFO_ROOT / f"{dataset}.sqf"
    raise SynthMountMissing(
        f"missing {samples} (sqf did not mount at the expected path). "
        f"Mount {sqf} read-only. Example: "
        f"DATASETS={dataset} sbatch sbatch/ppp/prep/synth/inspect_sqf_dataset_sbatch.sh"
    )


def ppp_mount_dir(dataset: str) -> Path:
    return SYNTH_MOUNT_ROOT / f"{dataset}_ppp"


def require_ppp_mount(dataset: str) -> None:
    """Raise when prepared-array symlinks into the packed PPP image cannot resolve."""
    mount = ppp_mount_dir(dataset)
    if mount.is_dir():
        return
    raise SynthMountMissing(
        f"missing {mount} (packed PPP sqf did not mount). "
        f"DATASETS={dataset} sbatch sbatch/ppp/prep/pack_ppp_sqf_sbatch.sh"
    )


def zarr_array_metadata(zarr_path: Path, array: str) -> dict:
    """Header-only metadata for volumes/<array>. Does not read voxels.

    An unreadable array is returned with status set to the error so one
    dangling symlink does not abort a dataset scan.
    """
    vol = Path(zarr_path) / "volumes" / array
    row = {
        "shape": "",
        "dtype": "",
        "chunks": "",
        "symlink_target": "",
        "status": "ok",
    }
    try:
        if vol.is_symlink():
            row["symlink_target"] = os.readlink(vol)
    except OSError as exc:
        row["status"] = f"{type(exc).__name__}: {exc}"
        return row

    try:
        group = zarr.open_group(str(zarr_path), mode="r")
        arr = group[f"volumes/{array}"]
        chunks = getattr(arr, "chunks", None)
        row["shape"] = "x".join(str(int(x)) for x in arr.shape)
        row["dtype"] = str(arr.dtype)
        row["chunks"] = (
            "x".join(str(int(x)) for x in chunks) if chunks else ""
        )
    except Exception as exc:
        row["status"] = f"{type(exc).__name__}: {exc}"
    return row


def load_zarr_volume(zarr_path: Path, array: str) -> np.ndarray:
    """Load volumes/<array> fully into memory (CZYX for FISBe / synth)."""
    group = zarr.open_group(str(zarr_path), mode="r")
    return np.asarray(group[f"volumes/{array}"])

