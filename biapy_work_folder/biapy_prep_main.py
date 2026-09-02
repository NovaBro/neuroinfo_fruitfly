"""
Convert FISBe zarr volumes to BiaPy-compatible Zarr files.
FISBe source axes are CZYX. Written BiaPy volumes are:
    INPUT_IMG_AXES_ORDER: ZYXC   # raw after CZYX -> ZYXC transpose
    INPUT_MASK_AXES_ORDER: ZYXC  # merged instance-ID volume with singleton C

For the FISBe dataset, the image dimensions are specified in the doc:
    https://kainmueller-lab.github.io/fisbe/
    "The segmentation mask for each neuron is stored in a separate channel. The order of dimensions is CZYX."
"""
import argparse
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imaging_helpers_hpc.processing import axis_rotation

from biapy_prep_augments import apply_augmentation_set, generate_augmentation_jobs
from biapy_prep_utils import (
    _drain_completed,
    any_augment_enabled,
    get_args,
    merge_instance_masks,
    normalize_aug_args,
    save_and_log_data,
    setup_logging,
)

logger = logging.getLogger(__name__)


def convert_split(input_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    raw_dir = output_dir / "raw"
    label_dir = output_dir / "label"
    raw_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(input_dir.glob("*.zarr"))

    if args.max_num_samples:
        input_paths = input_paths[0:int(args.max_num_samples)]

    do_augment = any_augment_enabled(args)
    logger.info(f"Save workers: {args.workers}")
    logger.info(f"Augmentations enabled: {do_augment}")

    def process_samples(ex: ThreadPoolExecutor | None = None) -> None:
        futures: set = set()

        for index, in_path in enumerate(input_paths):
            if (raw_dir / f"{in_path.stem}.zarr").exists() and not args.clean:
                logger.info(f"This path already exists, skipping: '{raw_dir / in_path.name}'")
                continue

            in_path = Path(in_path)

            logger.info(
                f"/// Processing Sample [{index + 1} / {len(input_paths)}]: {in_path.name} ///"
            )

            raw = np.array(
                zarr.open(in_path.__str__(), mode="r", path="volumes/raw")
            )

            labels = np.array(
                zarr.open(in_path.__str__(), mode="r", path="volumes/gt_instances")
            )

            labels_merged = merge_instance_masks(labels)

            if do_augment:
                assert ex is not None
                aug_jobs = generate_augmentation_jobs(
                    num_instances=labels.shape[0],
                    enable_channel_flip=args.channel_flip,
                    num_channel_flips=args.num_channel_flips or 1,
                    enable_rotate=args.rotate,
                    num_axes=args.num_axes or 1,
                    num_rotations=args.num_rotations or 1,
                    enable_channel_scale=args.channel_scale,
                    channel_scale_range=(
                        tuple(args.channel_scale_range)
                        if args.channel_scale_range is not None
                        else None
                    ),
                    num_channel_scales=args.num_channel_scales or 1,
                    enable_instance_scale=args.instance_scale,
                    instance_scale_range=(
                        tuple(args.instance_scale_range)
                        if args.instance_scale_range is not None
                        else None
                    ),
                    num_instance_scales=args.num_instance_scales or 1,
                )
                logger.info(f"Number of augmentation jobs: {len(aug_jobs)}")
                logger.debug(aug_jobs)

                for a in aug_jobs:
                    raw_aug = apply_augmentation_set(raw, a, instances=labels)
                    labels_aug = axis_rotation(
                        labels_merged[np.newaxis, ...], a["a"], a["k"]
                    )[0]
                    futures.add(
                        ex.submit(
                            save_and_log_data,
                            raw_aug,
                            raw_dir,
                            labels_aug,
                            label_dir,
                            in_path,
                            a["aug_id"],
                        )
                    )
                    if len(futures) >= args.workers:
                        futures = _drain_completed(futures)
            else:
                save_and_log_data(
                    raw, raw_dir, labels_merged, label_dir, in_path, ""
                )

        _drain_completed(futures, wait_for_all=True)

    if do_augment:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            process_samples(ex)
    else:
        process_samples()


def main() -> None:
    args = get_args()

    setup_logging(args.verbose_level, args.log_output)
    normalize_aug_args(args)

    source_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    logger.info(f"source: {source_root}")
    logger.info(f"output: {output_root}")

    for split in args.splits:
        input_dir = source_root / split
        output_dir = output_root / split
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Missing zarr split directory: {input_dir}")

        if args.clean and output_dir.exists():
            logger.info(f"Removing directory: {output_dir}")
            shutil.rmtree(output_dir)

        logger.info(
            f"Converting {split}, "
            f"Found ({len(list(input_dir.glob('*.zarr')))} volumes), "
            f"Processing {args.max_num_samples}"
        )
        convert_split(input_dir, output_dir, args)


if __name__ == "__main__":
    main()
