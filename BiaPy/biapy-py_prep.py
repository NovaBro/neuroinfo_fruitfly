"""
Convert FISBe zarr volumes to BiaPy-compatible TIFF files.
For BiaPy, the input image dimensions are specified in the config file:
    INPUT_IMG_AXES_ORDER: CZYX   # raw (C, Z, Y, X)
    INPUT_MASK_AXES_ORDER: CZYX  # zarr GT also loaded with IMG axes → (1, Z, Y, X)

For the FISBe dataset, the image dimensions are specified in the doc:
    https://kainmueller-lab.github.io/fisbe/
    "The segmentation mask for each neuron is stored in a separate channel. The order of dimensions is CZYX."
"""
import argparse
import shutil
import logging
import itertools
import subprocess
from pathlib import Path

import numpy as np
import zarr
import tifffile
from tqdm import tqdm
from biapy.data.data_manipulation import save_tif

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imaging_helpers_hpc.processing import random_90_rotate_3d
# from imaging_helpers_hpc.gen_utils import log_wrapper

logger = logging.getLogger()

def merge_instance_masks(stacked: np.ndarray) -> np.ndarray:
    """Merge per-instance binary masks (I, Z, Y, X) into one label volume (Z, Y, X). I is number of instances, not number of samples"""
    logger.debug("merge_instance_masks")
    logger.debug(f"\tbefore array.ndim: {stacked.ndim}")
    logger.debug(f"\tbefore array.shape: {stacked.shape}")

    merged = np.zeros(stacked.shape[1:], dtype=np.uint16)
    for mask in stacked:
        instance_ids = mask > 0
        if not np.any(instance_ids):
            continue
        merged[instance_ids] = mask[instance_ids]
    logger.debug(f"\tnew merged.ndim: {merged.ndim}")
    logger.debug(f"\tnew merged.shape: {merged.shape}")
    return merged

def convert_split(input_dir: Path, output_dir: Path, args:argparse.Namespace) -> None:
    raw_dir = output_dir / "raw"
    label_dir = output_dir / "label"
    raw_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(input_dir.glob("*.zarr"))

    if args.max_num_samples: input_paths = input_paths[0:int(args.max_num_samples)]

    for index, in_path in enumerate(input_paths):
        if (raw_dir / in_path.name.replace('.zarr', '.tif')).exists():
            logger.info(f"This path already exists, skipping: '{raw_dir / in_path.name}'")
            continue

        in_path = Path(in_path)

        logger.info(f"Processing Sample [{index + 1} / {len(input_paths)}]: {in_path.name}")

        raw = np.array(zarr.open(
            in_path.__str__(), 
            mode="r", 
            path="volumes/raw"
        ))

        labels = np.array(zarr.open(
            in_path.__str__(), 
            mode="r", 
            path="volumes/gt_instances"
        ))

        # Apply Custom Augmentations Here

        # BiaPy Specific Formatting (c, z, y, x) --> (z, y, x, c)
        raw = np.transpose(raw, (1, 2, 3, 0))
        labels = merge_instance_masks(labels)
        raw = raw[np.newaxis, ...]
        labels = labels[np.newaxis, ..., np.newaxis]

        logger.debug(
            f"\tSaving - raw.shape: {raw.shape}"
            f" raw_dir: {raw_dir}"
            f" raw_names: {in_path.name}"
        )
        logger.debug(
            f"\tSaving - labels.shape: {labels.shape}"
            f" lage_dir: {label_dir}"
            f" label_names: {in_path.name}"
        )

        save_tif(raw, raw_dir, [in_path.name])
        save_tif(labels, label_dir, [in_path.name])

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", 
        "--output-dir",
        required=True,
        help="Output directory for BiaPy TIFF splits. Relative path",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default="fisbe/completely",
        help="Root directory containing train/test/val zarr splits",
    )

    parser.add_argument(
        "-s",
        "--splits",
        nargs="+",
        default=["test", "train", "val"],
        help="Dataset splits to convert",
    )
    parser.add_argument(
        '--max-num-samples',
        help="If set, then sets the maximum number of samples selected selected / processed. integer"
    )
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="Remove existing split directories before conversion",
    )
    parser.add_argument(
        "-l",
        "--log-output",
        default="biapy-prep-tiff_log.txt",
        help="specify output file for logging",
    )
    parser.add_argument(
        "-v",
        "--verbose-level",
        default='debug',
        help="verbose logging level"
    )

    args = parser.parse_args()

    source_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    logger.info(f"source: {source_root}")
    logger.info(f"output: {output_root}")

    match args.verbose_level:
        case 'debug':
            log_level=logging.DEBUG
        case 'info':
            log_level=logging.INFO
        case 'warning':
            log_level=logging.WARNING

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)-8s - %(funcName)-25s: %(message)s',
        handlers=[
            logging.FileHandler(f"{args.log_output}", 'w'),   # Writes to file
            logging.StreamHandler(sys.stdout)           # Prints to console (sys.stderr)
        ]
    )

    for split in args.splits:
        input_dir = source_root / split
        output_dir = output_root / split
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Missing zarr split directory: {input_dir}")

        if args.clean and output_dir.exists():
            logger.info(f"Removing directory: {output_dir}")
            subprocess.run(
                ['rm', '-rf', '--', str(output_dir)],
                check=True
            )
            # shutil.rmtree(output_dir)

        logger.info(
            f"Converting {split}, "
            f"Found ({len(list(input_dir.glob('*.zarr')))} volumes), "
            f"Processing {args.max_num_samples}"
        )
        convert_split(input_dir, output_dir, args)


if __name__ == "__main__":
    main()
