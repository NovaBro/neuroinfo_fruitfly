"""CLI args and I/O helpers for BiaPy FISBe TIFF prep."""
import argparse
import logging
import sys
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path

import numpy as np
import tifffile

logger = logging.getLogger(__name__)

_DEFAULT_NUM_CHANNEL_FLIPS = 1
_DEFAULT_NUM_AXES = 1
_DEFAULT_NUM_ROTATIONS = 1
_DEFAULT_NUM_CHANNEL_SCALES = 1
_DEFAULT_NUM_INSTANCE_SCALES = 1
_DEFAULT_SCALE_RANGE = (0.25, 1.5)


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert FISBe zarr volumes to BiaPy-compatible TIFF files."
        )
    )
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

    # Opt-in augmentations
    parser.add_argument(
        "--channel-flip",
        action="store_true",
        help="Enable channel-order permutation augmentations.",
    )
    parser.add_argument(
        "--num-channel-flips",
        type=int,
        default=None,
        metavar="N",
        help=f"Number of channel permutations to sample (default: {_DEFAULT_NUM_CHANNEL_FLIPS}). Requires --channel-flip.",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Enable 90-degree axis rotation augmentations.",
    )
    parser.add_argument(
        "--num-axes",
        type=int,
        default=None,
        metavar="N",
        help=f"Number of spatial axes to sample for rotation (default: {_DEFAULT_NUM_AXES}). Requires --rotate.",
    )
    parser.add_argument(
        "--num-rotations",
        type=int,
        default=None,
        metavar="N",
        help=f"Number of k=0..3 rotation amounts to sample (default: {_DEFAULT_NUM_ROTATIONS}). Requires --rotate.",
    )
    parser.add_argument(
        "--channel-scale",
        action="store_true",
        help="Enable per-channel multiplicative intensity scaling.",
    )
    parser.add_argument(
        "--channel-scale-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help=f"Channel scale range (default: {_DEFAULT_SCALE_RANGE[0]} {_DEFAULT_SCALE_RANGE[1]}). Requires --channel-scale.",
    )
    parser.add_argument(
        "--num-channel-scales",
        type=int,
        default=None,
        metavar="N",
        help=f"Number of channel-scale draws to sample (default: {_DEFAULT_NUM_CHANNEL_SCALES}). Requires --channel-scale.",
    )
    parser.add_argument(
        "--instance-scale",
        action="store_true",
        help="Enable per-instance multiplicative intensity scaling.",
    )
    parser.add_argument(
        "--instance-scale-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help=f"Instance scale range (default: {_DEFAULT_SCALE_RANGE[0]} {_DEFAULT_SCALE_RANGE[1]}). Requires --instance-scale.",
    )
    parser.add_argument(
        "--num-instance-scales",
        type=int,
        default=None,
        metavar="N",
        help=f"Number of instance-scale draws to sample (default: {_DEFAULT_NUM_INSTANCE_SCALES}). Requires --instance-scale.",
    )

    parser.add_argument(
        "-s",
        "--splits",
        nargs="+",
        default=["test", "train", "val"],
        help="Dataset splits to convert",
    )
    parser.add_argument(
        "--max-num-samples",
        help="If set, then sets the maximum number of samples selected selected / processed. integer",
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
        default="debug",
        help="verbose logging level",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of threads for parallel TIFF saves (default: 4). Use 1 for serial.",
    )

    return parser.parse_args()


def any_augment_enabled(args: argparse.Namespace) -> bool:
    return bool(
        args.channel_flip
        or args.rotate
        or args.channel_scale
        or args.instance_scale
    )


def normalize_aug_args(args: argparse.Namespace) -> None:
    """Validate param/enable pairing and fill defaults for enabled augs (in-place)."""
    _require_enable(args.channel_flip, args.num_channel_flips, "--num-channel-flips", "--channel-flip")
    _require_enable(args.rotate, args.num_axes, "--num-axes", "--rotate")
    _require_enable(args.rotate, args.num_rotations, "--num-rotations", "--rotate")
    _require_enable(
        args.channel_scale, args.channel_scale_range, "--channel-scale-range", "--channel-scale"
    )
    _require_enable(
        args.channel_scale, args.num_channel_scales, "--num-channel-scales", "--channel-scale"
    )
    _require_enable(
        args.instance_scale, args.instance_scale_range, "--instance-scale-range", "--instance-scale"
    )
    _require_enable(
        args.instance_scale, args.num_instance_scales, "--num-instance-scales", "--instance-scale"
    )

    if args.channel_flip and args.num_channel_flips is None:
        args.num_channel_flips = _DEFAULT_NUM_CHANNEL_FLIPS
    if args.rotate:
        if args.num_axes is None:
            args.num_axes = _DEFAULT_NUM_AXES
        if args.num_rotations is None:
            args.num_rotations = _DEFAULT_NUM_ROTATIONS
    if args.channel_scale:
        if args.channel_scale_range is None:
            args.channel_scale_range = list(_DEFAULT_SCALE_RANGE)
        if args.num_channel_scales is None:
            args.num_channel_scales = _DEFAULT_NUM_CHANNEL_SCALES
    if args.instance_scale:
        if args.instance_scale_range is None:
            args.instance_scale_range = list(_DEFAULT_SCALE_RANGE)
        if args.num_instance_scales is None:
            args.num_instance_scales = _DEFAULT_NUM_INSTANCE_SCALES

    if args.channel_scale:
        _validate_scale_range(
            "--channel-scale-range",
            args.channel_scale_range[0],
            args.channel_scale_range[1],
        )
        _validate_positive_count("--num-channel-scales", args.num_channel_scales)
    if args.instance_scale:
        _validate_scale_range(
            "--instance-scale-range",
            args.instance_scale_range[0],
            args.instance_scale_range[1],
        )
        _validate_positive_count("--num-instance-scales", args.num_instance_scales)


def _require_enable(enabled: bool, value, param_flag: str, enable_flag: str) -> None:
    if value is not None and not enabled:
        raise ValueError(f"{param_flag} requires {enable_flag}")


def _validate_scale_range(name: str, lo: float, hi: float) -> None:
    if not (0 < lo <= hi):
        raise ValueError(f"{name} must satisfy 0 < lo <= hi, got ({lo}, {hi})")


def _validate_positive_count(name: str, n: int) -> None:
    if n < 1:
        raise ValueError(f"{name} must be >= 1, got {n}")


def setup_logging(verbose_level: str, log_output: str) -> None:
    match verbose_level:
        case "debug":
            log_level = logging.DEBUG
        case "info":
            log_level = logging.INFO
        case "warning":
            log_level = logging.WARNING
        case _:
            raise ValueError("Not valid logging error")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)-8s - %(funcName)-25s: %(message)s",
        handlers=[
            logging.FileHandler(f"{log_output}.txt", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def merge_instance_masks(stacked: np.ndarray) -> np.ndarray:
    """Merge per-instance binary masks (I, Z, Y, X) into one label volume (Z, Y, X).

    I is number of instances, not number of samples.
    """
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
    raw: np.ndarray,
    raw_dir: Path,
    labels: np.ndarray,
    label_dir: Path,
    in_path: Path,
    aug_id: str,
):
    # ImageJ TIFF format:
    #   raw: (C, Z, Y, X) --> (Z, C, Y, X)
    #   labels: already merged (Z, Y, X)
    raw = np.transpose(raw, (1, 0, 2, 3))

    logger.debug(
        f"\tSaving - raw.shape: {raw.shape}"
        f" raw_dir: {raw_dir}"
        f" raw_names: {in_path.name}"
    )
    logger.debug(
        f"\tSaving - labels.shape: {labels.shape}"
        f" label_dir: {label_dir}"
        f" label_names: {in_path.name}"
    )

    raw_file_name = in_path.stem + aug_id
    label_file_name = in_path.stem + aug_id

    logger.info(f"Saving raw to file: {raw_file_name} , Dir: {raw_dir}")
    logger.info(f"Saving label to file: {label_file_name}, Dir: {label_dir}")

    tifffile.imwrite(
        (raw_dir / f"{raw_file_name}.tif").as_posix(),
        raw,
        imagej=True,
        metadata={"axes": "ZCYX"},
    )
    tifffile.imwrite(
        (label_dir / f"{label_file_name}.tif").as_posix(),
        labels,
        imagej=True,
        metadata={"axes": "ZYX"},
    )


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
