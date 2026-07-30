import re
import os
import sys
import logging
from pathlib import Path

import zarr
import tifffile
import numpy as np

from imaging_helpers_hpc.paths import BiapyDataPaths

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

def _strip_ansi(text: str) -> str:
    """Remove ANSI color/formatting escape codes so log files stay readable."""
    return _ANSI_ESCAPE.sub("", text)

def get_stats_in_one_image(img_path:Path):
    logger.info(f'get_stats_in_one_image:')
    logger.debug(f'\tInput Image Path {img_path}')
    logger.debug(f'\tInput Image Suffix {img_path.suffix}')
    match img_path.suffix:
        case '.zarr':
            loaded_image = zarr.open_group(img_path)
            logger.error('ZARR processing not implemented!!')
            raise ValueError('ZARR processing not implemented!!')
        case '.tif':
            loaded_image = tifffile.imread(img_path)
        case _:
            error_msg = 'Extension of image path given is not valid, must be .tif (.zarr not implemented yet) image path given'
            logger.error(error_msg)
            raise ValueError(error_msg)
    unique_values = np.unique(loaded_image)

    logger.info(f"\tloaded_image.shape: {loaded_image.shape}")
    dtype_set = set([type(d) for d in unique_values.tolist()])
    logger.info(f"\tData type of values in {img_path.name}: [{dtype_set}]")
    if len(unique_values) < 1000:
        logger.info(f"\tNumber of unique values in {img_path.name} = {len(unique_values.tolist())}")
        logger.info(f"\tUnique values in {img_path.name}: {unique_values.tolist()}")
    else:
        logger.info(f"\tNumber of unique values in {img_path.name} = {len(unique_values.tolist())}")

def get_stats_in_dir(dir_path: Path):
    logger.info(f"get_stats_in_dir:")
    image_files = os.listdir(dir_path)
    logger.info(f"\tAnalyzing stats for {dir_path}")

    all_shapes = []
    for idx, image_name in enumerate(image_files):
        extension = image_name.split('.')[1]

        if extension == 'zarr':
            logger.info(f'\tLoading zarr files [{idx} / {len(image_files)}] {image_name}')
            loaded_image = zarr.open_group(dir_path / image_name)

            logger.info(f"\tImage: {image_name}")            
            logger.debug(f"\tGroup tree:\n{_strip_ansi(str(loaded_image.tree()))}")

            raw_image = loaded_image['volumes/raw']
            raw_image = np.array(raw_image)
            logger.info(f"\tRaw Shape: {raw_image.shape}")
            all_shapes.append(np.array(raw_image.shape))

            unique_values = np.unique(raw_image)

        else:
            logger.info(f'\tLoading tiff files {image_name}')
            loaded_image = tifffile.imread(dir_path / image_name)
            logger.info(f"\tImage Shape: {loaded_image.shape}")

            print(f"Image: {image_name}")
            print(f"\tShape: {loaded_image.shape}")
            all_shapes.append(np.array(loaded_image.shape))
            unique_values = np.unique(loaded_image)

        if len(unique_values) < 100:
            logger.info(f"\tUnique values in {image_name}: {unique_values.tolist()}")
        else:
            logger.info(f"\tUnique values in {image_name} > 100")

    all_shapes = np.array(all_shapes)
    avg_shape = np.mean(all_shapes, axis=0)
    med_shape = np.median(all_shapes, axis=0)
    max_shape = np.max(all_shapes, axis=0)
    min_shape = np.min(all_shapes, axis=0)
    logger.info(f"\tavg_shape: {avg_shape}")
    logger.info(f"\tmed_shape: {med_shape}")
    logger.info(f"\tmax_shape: {max_shape}")
    logger.info(f"\tmin_shape: {min_shape}")

    