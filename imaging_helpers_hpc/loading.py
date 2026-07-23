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

def load_any_tif(full_path):
    return tifffile.imread(full_path)

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

