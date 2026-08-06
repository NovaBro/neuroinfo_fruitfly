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
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import permutations
from pathlib import Path
import random
from typing import Dict

import numpy as np
import zarr
import tifffile
from tqdm import tqdm
from biapy.data.data_manipulation import save_tif


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imaging_helpers_hpc.processing import axis_rotation, channel_flip
# from imaging_helpers_hpc.gen_utils import log_wrapper

logger = logging.getLogger()

NUM_CHANNELS = 3


def _dtype_max(dtype) -> float:
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else np.finfo(dtype)
    return float(info.max)


def _scale_clip_cast(raw_f32: np.ndarray, dtype) -> np.ndarray:
    """Clip float32 intensities to ``dtype`` range and cast back."""
    return np.clip(raw_f32, 0, _dtype_max(dtype)).astype(dtype, copy=False)


def per_channel_multiplicative_scale(
    raw: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    """Multiply each channel of ``raw`` (C,Z,Y,X) by ``scales`` (C,)."""
    scales = np.asarray(scales, dtype=np.float32)
    if scales.shape != (raw.shape[0],):
        raise ValueError(
            f"channel scales shape {scales.shape} != ({raw.shape[0]},)"
        )
    out = raw.astype(np.float32, copy=False) * scales.reshape(-1, 1, 1, 1)
    return _scale_clip_cast(out, raw.dtype)


def per_instance_intensity_scale(
    raw: np.ndarray,
    instances: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Scale raw voxels per instance mask; overlaps accumulate as a product.

    Parameters
    ----------
    raw : (C, Z, Y, X)
    instances : (I, Z, Y, X)
    scales : (I,)
    """
    scales = np.asarray(scales, dtype=np.float32)
    if scales.shape != (instances.shape[0],):
        raise ValueError(
            f"instance scales shape {scales.shape} != ({instances.shape[0]},)"
        )
    out = raw.astype(np.float32, copy=True)
    for i, mask in enumerate(instances):
        fg = mask > 0
        if not np.any(fg):
            continue
        out[:, fg] *= scales[i]
    return _scale_clip_cast(out, raw.dtype)


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
    # raw = np.transpose(raw, (1, 2, 3, 0))
    # raw = raw[np.newaxis, ...]
    # labels = labels[np.newaxis, ..., np.newaxis]

    # ImageJ Tiffle Format: TZCYX
        # raw: (C, Z, Y, X) --> (Z, C, Y, X); 
        # labels: already merged (Z, Y, X)
    raw = np.transpose(raw, (1, 0, 2, 3))

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

    raw_file_name = in_path.stem + aug_id
    lablel_file_name = in_path.stem + aug_id

    logger.info(f"Saving raw to file: {raw_file_name} , Dir: {raw_dir}")
    logger.info(f"Saving labelto file: {lablel_file_name}, Dir: {label_dir}")

    # save_tif(raw, raw_dir.as_posix(), [raw_file_name])
    # save_tif(labels, label_dir.as_posix(), [lablel_file_name])

    tifffile.imwrite(
        (raw_dir / f"{raw_file_name}.tif").as_posix(), raw, 
        imagej=True, metadata={"axes": "ZCYX"}
    )
    tifffile.imwrite(
        (label_dir / f"{lablel_file_name}.tif").as_posix(), labels, 
        imagej=True, metadata={"axes": "ZYX"}
    )

def _sample_uniform(lo: float, hi: float, size: int) -> np.ndarray:
    return np.random.uniform(lo, hi, size=size).astype(np.float32)


def generate_augmentation_jobs(
    num_channel_flips=1,
    num_axis=1,
    num_rotations=1,
    *,
    num_instances: int = 0,
    channel_scale_range: tuple[float, float] | None = (0.25, 1.5),
    instance_scale_range: tuple[float, float] | None = (0.25, 1.5),
    enable_channel_scale: bool = True,
    enable_instance_scale: bool = True,
) -> list[dict]:
    """
    Generate jobs for augmentation.
    c - channel flip order
    a - rotation axis selection
    k - rotate number of 90 degrees
    One intensity draw (channel / instance scales) per geometry job.
    """
    jobs = []

    all_orders = list(permutations([0, 1, 2], 3))
    idx = np.random.choice(len(all_orders), size=num_channel_flips, replace=False)
    random_order = [all_orders[i] for i in idx]

    for ro in random_order:
        # Channel Flipping
        augid = '_c' + ''.join([str(x) for x in ro])

        # Rotation
        rand_axis_idx = np.random.permutation(3)[0:num_axis] # Select permuted axis 
        rand_k_rotations = np.random.permutation(4)[0:num_rotations] # Select permuted rotations
        for a in rand_axis_idx:
            for k in rand_k_rotations:
                aug_id = augid + f"_r{a}" + f"_k{k}"
                job: dict = {
                    'aug_id': aug_id,
                    'c': ro,
                    'a': a,
                    'k': k,
                }

                if enable_channel_scale and channel_scale_range is not None:
                    lo, hi = channel_scale_range
                    channel_scales = _sample_uniform(lo, hi, NUM_CHANNELS)
                    job['channel_scales'] = channel_scales
                    job['aug_id'] += '_cs' + '_'.join(f'{s:.2f}' for s in channel_scales)

                if (
                    enable_instance_scale
                    and instance_scale_range is not None
                    and num_instances > 0
                ):
                    lo, hi = instance_scale_range
                    job['instance_scales'] = _sample_uniform(lo, hi, num_instances)
                    job['aug_id'] += '_is'

                jobs.append(job)

    return jobs


def apply_augmentation_set(
    raw: np.ndarray,
    augmentation: dict,
    instances: np.ndarray | None = None,
) -> np.ndarray:
    """Apply intensity then geometry augs to raw (C,Z,Y,X). Labels are not modified."""
    image = raw
    if 'instance_scales' in augmentation:
        if instances is None:
            raise ValueError("instance_scales present but instances is None")
        image = per_instance_intensity_scale(
            image, instances, augmentation['instance_scales']
        )
    if 'channel_scales' in augmentation:
        image = per_channel_multiplicative_scale(
            image, augmentation['channel_scales']
        )
    image = channel_flip(image, augmentation['c'])
    image = axis_rotation(image, augmentation['a'], augmentation['k'])
    return image

def _drain_completed(futures: set, *, wait_for_all: bool = False) -> set:
    """Wait for completed futures, raise on errors, return remaining futures."""
    if not futures:
        return futures
    if wait_for_all:
        done, futures = wait(futures)
    else:
        done, futures = wait(futures, return_when=FIRST_COMPLETED)
    for f in done:
        f.result()
    return futures


def convert_split(input_dir: Path, output_dir: Path, args:argparse.Namespace) -> None:
    raw_dir = output_dir / "raw"
    label_dir = output_dir / "label"
    raw_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(input_dir.glob("*.zarr"))

    if args.max_num_samples: input_paths = input_paths[0:int(args.max_num_samples)]

    logger.info(f"Save workers: {args.workers}")

    def process_samples(ex: ThreadPoolExecutor | None = None) -> None:
        futures: set = set()

        for index, in_path in enumerate(input_paths):
            if (raw_dir / in_path.name.replace('.zarr', '.tif')).exists() and not args.clean:
                logger.info(f"This path already exists, skipping: '{raw_dir / in_path.name}'")
                continue

            in_path = Path(in_path)

            logger.info(f"/// Processing Sample [{index + 1} / {len(input_paths)}]: {in_path.name} ///")

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
                assert ex is not None
                aug_jobs = generate_augmentation_jobs(
                    num_instances=labels.shape[0],
                    channel_scale_range=tuple(args.channel_scale_range),
                    instance_scale_range=tuple(args.instance_scale_range),
                    enable_channel_scale=not args.no_channel_scale,
                    enable_instance_scale=not args.no_instance_scale,
                )
                logger.info(f"Number of augmentation jobs: {len(aug_jobs)}")
                logger.debug(aug_jobs)

                for a in aug_jobs:
                    # NOTE: If want to check for already generated augmented data
                    # aug_id = a["aug_id"]
                    # raw_out = raw_dir / f"{in_path.stem}{aug_id}.tif"
                    # label_out = label_dir / f"{in_path.stem}{aug_id}.tif"
                    # if not args.clean and raw_out.exists() and label_out.exists():
                    #     logger.info(f"Skipping existing augmented output: {raw_out}")
                    #     continue

                    raw_aug = apply_augmentation_set(raw, a, instances=labels)
                    labels_aug = axis_rotation(
                        labels_merged[np.newaxis, ...], a['a'], a['k']
                    )[0]
                    futures.add(ex.submit(
                        save_and_log_data,
                        raw_aug, raw_dir,
                        labels_aug, label_dir,
                        in_path, a['aug_id'],
                    ))
                    if len(futures) >= args.workers:
                        futures = _drain_completed(futures)
            else:
                save_and_log_data(
                    raw, raw_dir,
                    labels_merged, label_dir,
                    in_path, ''
                )

        _drain_completed(futures, wait_for_all=True)

    if args.augment:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            process_samples(ex)
    else:
        process_samples()

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
        "--channel-scale-range",
        nargs=2,
        type=float,
        default=[0.25, 1.5],
        metavar=("LO", "HI"),
        help="Per-channel multiplicative scale range (default: 0.25 1.5). Used with --augment.",
    )
    parser.add_argument(
        "--instance-scale-range",
        nargs=2,
        type=float,
        default=[0.25, 1.5],
        metavar=("LO", "HI"),
        help="Per-instance multiplicative scale range (default: 0.25 1.5). Used with --augment.",
    )
    parser.add_argument(
        "--no-channel-scale",
        action="store_true",
        help="Disable per-channel intensity scaling when --augment is set.",
    )
    parser.add_argument(
        "--no-instance-scale",
        action="store_true",
        help="Disable per-instance intensity scaling when --augment is set.",
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
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of threads for parallel TIFF saves (default: 4). Use 1 for serial.",
    )

    return parser.parse_args()

def _validate_scale_range(name: str, lo: float, hi: float) -> None:
    if not (0 < lo <= hi):
        raise ValueError(f"{name} must satisfy 0 < lo <= hi, got ({lo}, {hi})")


def main() -> None:

    args = get_args()

    if args.augment:
        _validate_scale_range(
            "--channel-scale-range",
            args.channel_scale_range[0],
            args.channel_scale_range[1],
        )
        _validate_scale_range(
            "--instance-scale-range",
            args.instance_scale_range[0],
            args.instance_scale_range[1],
        )

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

