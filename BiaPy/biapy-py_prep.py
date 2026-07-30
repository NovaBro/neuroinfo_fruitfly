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
import subprocess
from pathlib import Path
from itertools import permutations

import numpy as np
import zarr
import tifffile
from tqdm import tqdm
from biapy.data.data_manipulation import save_tif

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imaging_helpers_hpc.processing import axis_rotation, channel_flip
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

def save_and_log_data(
    raw:np.ndarray, raw_dir:Path,
    labels:np.ndarray, label_dir:Path, 
    in_path:Path, aug_id:str
):
    # raw: (C, Z, Y, X) --> (1, Z, Y, X, C); labels: already merged (Z, Y, X) --> (1, Z, Y, X, 1)
    raw = np.transpose(raw, (1, 2, 3, 0))
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
    logger.info(f"Saving to lage_dir: {label_dir}")
    logger.info(f"Saving to raw_dir: {raw_dir}")

    save_tif(raw, raw_dir.as_posix(), [in_path.name + aug_id])
    save_tif(labels, label_dir.as_posix(), [in_path.name + aug_id])

def generate_augmentation_jobs(num_axis=2, num_rotations=2) -> list[dict]:
    """
    Generate jobs for augmentation.
    c - channel flip order
    a - rotation axis selection
    k - rotate number of 90 degrees
    """
    jobs = []

    random_order = permutations([0, 1, 2], 3) # Get all permutations
    for ro in random_order:
        # Channel Flipping
        augid = '_c' + ''.join([str(x) for x in ro])

        # Rotation
        rand_axis_idx = np.random.permutation(3)[0:num_axis] # Select permuted axis 
        rand_k_rotations = np.random.permutation(4)[0:num_rotations] # Select permuted rotations
        for a in rand_axis_idx:
            for k in rand_k_rotations:
                aug_id = augid + f"_r{a}" + f"_k{k}"
                jobs.append(
                    {
                        'aug_id': aug_id,
                        'c' : ro,
                        'a' : a,
                        'k' : k
                    }
                )

    return jobs

def apply_augmentation_set(image:np.ndarray, augmentation:dict):
    image = channel_flip(image, augmentation['c'])
    image = axis_rotation(image, augmentation['a'], augmentation['k'])
    return image

def convert_split(input_dir: Path, output_dir: Path, args:argparse.Namespace) -> None:
    raw_dir = output_dir / "raw"
    label_dir = output_dir / "label"
    raw_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(input_dir.glob("*.zarr"))

    if args.max_num_samples: input_paths = input_paths[0:int(args.max_num_samples)]

    aug_jobs = generate_augmentation_jobs()
    logger.info(f"Number of augmentation jobs: {len(aug_jobs)}")
    logger.debug(aug_jobs)

    for index, in_path in enumerate(input_paths):
        if (raw_dir / in_path.name.replace('.zarr', '.tif')).exists() and not args.clean:
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

        labels_merged = merge_instance_masks(labels)

        if args.augment:
            for a in aug_jobs:
                raw_aug = apply_augmentation_set(raw, a)
                labels_aug = axis_rotation(
                    labels_merged[np.newaxis, ...], a['a'], a['k']
                )[0]
                save_and_log_data(
                    raw_aug, raw_dir,
                    labels_aug, label_dir,
                    in_path, a['aug_id']
                )
        else:
            save_and_log_data(
                raw, raw_dir,
                labels_merged, label_dir,
                in_path, ''
            )


def get_args():
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
        "-a",
        "--augment",
        action="store_true",
        help="Applies augmentation to data",
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

    return parser.parse_args()

def main() -> None:

    args = get_args()

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
        case _:
            raise ValueError("Not valid logging error")

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

