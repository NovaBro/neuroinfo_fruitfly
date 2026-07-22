"""
Convert FISBe zarr volumes to BiaPy-compatible TIFF files.
For BiaPy, the input image dimensions are specified in the config file, 
here it is for BiaPy/3d_instance_segmentation.yaml:
    INPUT_IMG_AXES_ORDER: CZYX
    INPUT_MASK_AXES_ORDER: ZYX

For the FISBe dataset, the image dimensions are specified in the doc:
    https://kainmueller-lab.github.io/fisbe/
    "The segmentation mask for each neuron is stored in a separate channel. The order of dimensions is CZYX."
"""
import argparse
import shutil
import logging
import itertools
from pathlib import Path

import numpy as np
import zarr
import tifffile
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imaging_helpers_hpc.processing import random_90_rotate_3d
# from imaging_helpers_hpc.gen_utils import log_wrapper

logger = logging.getLogger()

def merge_instance_masks(stacked: np.ndarray) -> np.ndarray:
    """Merge per-instance binary masks (N, Z, Y, X) into one label volume (Z, Y, X)."""
    logger.debug("merge_instance_masks")
    logger.debug(f"\tarray.ndim: {stacked.ndim}")
    logger.debug(f"\tarray.shape: {stacked.shape}")

    merged = np.zeros(stacked.shape[1:], dtype=np.uint16)
    for mask in stacked:
        instance_ids = mask > 0
        if not np.any(instance_ids):
            continue
        merged[instance_ids] = mask[instance_ids]
    logger.debug(f"\tnew array.ndim: {stacked.ndim}")
    logger.debug(f"\tnew array.shape: {stacked.shape}")
    return merged

# def to_biapy_volume_format(array: np.ndarray) -> np.ndarray:
#     """Wrap a ZYX or ZYXC array as a 6D TZCYXS volume for BiaPy."""
#     if array.ndim == 3:
#         array = array[..., np.newaxis]
#     if array.ndim != 4:
#         raise ValueError(f"Expected ZYX or ZYXC array, got shape {array.shape}")

#     volume = array.transpose(0, 3, 1, 2)  # ZYXC -> ZCYX
#     return volume[np.newaxis, :, :, :, :, np.newaxis]  # TZCYXS

def to_biapy_volume_format(array: np.ndarray) -> np.ndarray:
    """Wrap a ZYX or ZYXC array as a 6D TZCYXS volume for BiaPy."""
    logger.debug(f" - array.ndim: {array.ndim}"
                 f" - array.shape: {array.shape}")
    if array.ndim == 3:
        array = array[..., np.newaxis]
    if array.ndim != 4:
        raise ValueError(f"Expected ZYX or ZYXC array, got shape {array.shape}")

    volume = array.transpose(0, 3, 1, 2)  # ZYXC -> ZCYX
    return volume[np.newaxis, :, :, :, :, np.newaxis]  # TZCYXS

def apply_format_and_save(
        raw:np.ndarray, 
        seg:np.ndarray, 
        zarr_path:Path, 
        raw_dir:Path, 
        label_dir:Path, 
        aug_id:str=''
):
    logger.debug(f"apply_format_and_save:")
    if aug_id: logger.debug(f"\taugment id: {aug_id}")
    if raw.ndim != 4 or seg.ndim != 4:
        raise ValueError(
            f"{zarr_path.name}: expected 4D raw/seg arrays, got raw={raw.shape}, seg={seg.shape}"
        )

    # logger.info(f"Converting Raw to ImageJ Format")
    # raw = to_biapy_volume_format(raw.transpose(1, 2, 3, 0))  # CZYX -> ZYXC -> TZCYXS
    # logger.info(f"Converting Segmentation to ImageJ Format")
    merged_labels = merge_instance_masks(seg)
    # label_volume = to_biapy_volume_format(merged_labels)

    stem = zarr_path.name
    # biapy_imwrite(str(raw_dir / f"{stem}{aug_id}.tiff"), raw_volume)
    # biapy_imwrite(str(label_dir / f"{stem}_seg{aug_id}.tiff"), label_volume)
    # biapy_imwrite(str(raw_dir / f"{stem}{aug_id}.tiff"), raw)
    # biapy_imwrite(str(label_dir / f"{stem}_seg{aug_id}.tiff"), merged_labels)
    raw_save = str(raw_dir / f"{stem}{aug_id}.zarr")
    label_save = str(label_dir / f"{stem}_seg{aug_id}.zarr")
    logger.debug(f"\traw_save path: {raw_save}")
    logger.debug(f"\tlabel_save path: {label_save}")
    # tifffile.imwrite(
    #     raw_save, 
    #     raw,
    #     imagej=True,
    #     metadata={'axes': 'CZYX'}
    # )
    # tifffile.imwrite(
    #     label_save, 
    #     merged_labels,
    #     imagej=True,
    #     metadata={'axes': 'CZYX'}
    # )
    zarr.save_array(
        str(raw_save),
        raw
    )
    zarr.save_array(
        str(label_save),
        merged_labels
    )

def convert_split(zarr_split_dir: Path, tiff_split_dir: Path, args) -> None:
    raw_dir = tiff_split_dir / "raw"
    label_dir = tiff_split_dir / "label"
    raw_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    zarr_paths = sorted(zarr_split_dir.glob("*.zarr"))
    if args.max_num_samples:
        max_num_samples = int(args.max_num_samples)
        zarr_paths = zarr_paths[np.random.randint(0, len(zarr_paths), max_num_samples)]
        logger.info(f"Selected Subset of Samples: {[z.name for z in zarr_paths]}")
    logger.info(f"Number of files: {len(zarr_paths)}")

    if args.augment_data: logger.info(f"Augmenting Data")
    for index, zarr_path in enumerate(tqdm(zarr_paths)):
        zarr_path = Path(zarr_path)
        # tqdm.write(f"zarr_path: {zarr_path}")
        logger.info(f"zarr_path [{index} / {len(zarr_paths)}]: {zarr_path}")
        raw = np.array(zarr.open(
            zarr_path.__str__(), 
            mode="r", 
            path="volumes/raw"
        ))
        seg = np.array(zarr.open(
            zarr_path.__str__(), 
            mode="r", 
            path="volumes/gt_instances"
        ))
        logger.debug(f"raw.ndim: {raw.ndim}"
                     f" - raw.shape: {raw.shape}"
                     f" - seg.ndim: {seg.ndim}"
                     f" - seg.shape: {seg.shape}")

        if args.augment_data:
            # Rotations
            # rand_axis_int = np.random.randint(0, 3)
            # k_rotations = np.random.randint(0, 4)
            k_chosen = np.random.choice(range(0, 4), size=3, replace=False)

            for r in range(0, 3):
                for k in k_chosen:
                    logger.debug(f"\tRotating Raw:")
                    raw = random_90_rotate_3d(raw, r, k)
                    logger.debug(f"\tRotating Seg:")
                    seg = random_90_rotate_3d(seg, r, k)
            
                    # Color Permutation
                    # random_order = np.random.permutation(3)
                    all_perms = np.array(list(itertools.permutations([0, 1, 2])))
                    c_chosen = np.random.choice(len(all_perms), size=3, replace=False)
                    c_chosen = all_perms[c_chosen]

                    for c in c_chosen:
                        raw = raw[c, ...]
                        s = "".join(str(n) for n in c)
                        apply_format_and_save(raw, seg, zarr_path, raw_dir, label_dir, f"_r{r}-k{k}-c{s}")
        else:
            apply_format_and_save(raw, seg, zarr_path, raw_dir, label_dir)
        logger.debug(f"Finished {zarr_path}")

def main() -> None:
    # python3 BiaPy/biapy_prep_tiff.py --output "fisbe/aug_biapy" --splits test train val -a
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input-dir",
        default="fisbe/completely",
        help="Root directory containing train/test/val zarr splits",
    )
    parser.add_argument(
        "-o", 
        "--output-dir",
        help="Output directory for BiaPy TIFF splits",
    )
    parser.add_argument(
        "-s",
        "--splits",
        nargs="+",
        default=["test", "train", "val"],
        help="Dataset splits to convert",
    )
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="Remove existing split directories before conversion",
    )
    parser.add_argument(
        "-a",
        "--augment-data",
        action="store_true",
        help="Allow data augmentation: rotations and channels color changes",
    )
    parser.add_argument(
        "-l",
        "--log-output",
        default="biapy-prep-tiff_log.txt",
        help="specify output file for logging",
    )
    parser.add_argument(
        "--max-num-samples",
        help="Maximum number of samples to process"
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
        format='%(asctime)s - %(levelname)s - %(funcName)-30s: %(message)s',
        handlers=[
            logging.FileHandler(f"{args.log_output}", 'w'),   # Writes to file
            logging.StreamHandler(sys.stdout)           # Prints to console (sys.stderr)
        ]
    )

    for split in args.splits:
        zarr_split_dir = source_root / split
        tiff_split_dir = output_root / split
        if not zarr_split_dir.is_dir():
            raise FileNotFoundError(f"Missing zarr split directory: {zarr_split_dir}")

        if args.clean and tiff_split_dir.exists():
            logger.info(f"removing tree: {tiff_split_dir}")
            shutil.rmtree(tiff_split_dir)

        # tqdm.write(f"Converting {split} ({len(list(zarr_split_dir.glob('*.zarr')))} volumes)")
        logger.info(f"Converting {split} ({len(list(zarr_split_dir.glob('*.zarr')))} volumes)")
        convert_split(zarr_split_dir, tiff_split_dir, args)


if __name__ == "__main__":
    main()
