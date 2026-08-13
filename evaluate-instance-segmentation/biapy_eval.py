import os
import argparse
from pathlib import Path
from time import time

import toml
import zarr
import tifffile
import numpy as np
# from tqdm import tqdm
from evalinstseg import evaluate_file
# from biapy.data.data_manipulation import read_img_as_ndarray

ROOT_DIR = Path("/scratch/wmz2007/neuroinfo_fruitfly")
RESULT_ROOT_DIR = ROOT_DIR / "metrics" / "biapy"

TIFF_SUFFIXES = (".zarr.tif", ".zarr.tiff", ".tif", ".tiff")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str)
    parser.add_argument("--job-name", type=str)
    parser.add_argument("--run-id", default=0, type=str)
    parser.add_argument("--split", default='test', help="Either train, test, or val", type=str)
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        default=False,
        help="Recompute metrics even if cached .toml results already exist",
    )

    args = parser.parse_args()
    return args


def tiff_stem(path: Path) -> str:
    """Strip BiaPy-style TIFF suffixes and return the sample stem."""
    name = path.name
    for suffix in TIFF_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def tiff_to_zarr(tiff_path, zarr_path, zarr_key="volumes/pred_instance"):
    """Load a TIFF and write it as a Zarr group array under ``zarr_key``."""
    # Load TIFF file as numpy array
    arr = np.squeeze(tifffile.imread(tiff_path))
    if not np.issubdtype(arr.dtype, np.integer):
        arr = np.rint(arr).astype(np.int32)
    print(f"Shape: {arr.shape}")

    # Create or open the Zarr file and store the array
    z = zarr.open_group(zarr_path, mode="w")
    group, dataset = os.path.split(zarr_key)
    g = z.require_group(group) if group else z

    # Remove dataset if it already exists (overwrite)
    # g.create_array(dataset, data=arr, overwrite=True)
    g.create_dataset(dataset, data=arr, overwrite=True)

    print(f"Converted {tiff_path} -> {zarr_path}:{zarr_key}")


def pred_zarr_needs_int_cast(zarr_path, zarr_key="volumes/pred_instance"):
    """True if pred_instance is missing or not an integer dtype."""
    try:
        root = zarr.open(zarr_path, mode="r")
        arr = root[zarr_key]
    except Exception:
        return True
    return not np.issubdtype(arr.dtype, np.integer)


def main(args:argparse.Namespace):
    config_name = args.config_name
    job_name = args.job_name
    run_id = args.run_id
    gt_root = ROOT_DIR / f"fisbe/completely/{args.split}"

    print(f"config_name: {config_name}")
    print(f"run_id: {run_id}")

    experiment_dir = RESULT_ROOT_DIR / f"{config_name}/results/{job_name}_{run_id}"
    assert experiment_dir.exists(), f"Experiment doesn't exist! Maybe run it? or wrong path! ({experiment_dir})"
    tiff_dir = experiment_dir / "per_image_instances"
    zarr_dir = experiment_dir / "per_image_instances_zarr"

    out_dir = RESULT_ROOT_DIR / f"{config_name}/results/{job_name}_{run_id}/eval-inst_metrics/{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert missing and/or non-integer pred Zarrs from BiaPy TIFFs
    zarr_files = list(zarr_dir.glob("*.zarr")) if zarr_dir.exists() else []
    tif_files = sorted(
        p for p in tiff_dir.glob("*.tif*") if p.suffix.lower() in {".tif", ".tiff"}
    ) if tiff_dir.exists() else []

    # Decide which TIFFs to (re)write as integer-label Zarrs:
    # 1) no matching Zarr yet, or 2) Zarr exists but pred_instance is non-integer
    # (e.g. float32 from an earlier conversion). Equal TIFF/Zarr counts alone must
    # not skip reconversion, or float preds would keep failing in filter_components.
    tif_by_stem = {tiff_stem(tif): tif for tif in tif_files}
    existing_stems = {zpath.stem for zpath in zarr_files}
    tifs_to_convert = []
    seen_stems = set()

    # Case 1: TIFF with no Zarr yet
    for tif in tif_files:
        stem = tiff_stem(tif)
        if stem not in existing_stems and stem not in seen_stems:
            tifs_to_convert.append(tif)
            seen_stems.add(stem)

    # Case 2: existing Zarr with non-integer labels; rebuild from the matching TIFF
    for zarr_path in zarr_files:
        stem = zarr_path.stem
        if stem in seen_stems:
            continue
        if pred_zarr_needs_int_cast(zarr_path) and stem in tif_by_stem:
            tifs_to_convert.append(tif_by_stem[stem])
            seen_stems.add(stem)

    if tifs_to_convert:
        print(
            f"Converting ({len(tifs_to_convert)}) tif to integer-label zarr files "
            f"(missing and/or non-integer preds)"
        )
        zarr_dir.mkdir(parents=True, exist_ok=True)
        print(f"Number of files to process: {len(tifs_to_convert)}")
        for idx, tif in enumerate(tifs_to_convert):
            print(f"Converting [{idx + 1} / {len(tifs_to_convert)}]: {tif}")
            tiff_to_zarr(tif, zarr_dir / f"{tiff_stem(tif)}.zarr")
        zarr_files = list(zarr_dir.glob("*.zarr"))

    print(f"Evaluating [ {len(zarr_files) + 1} ] Files:")
    for idx, res_file in enumerate(zarr_dir.glob("*.zarr")):
        print(f"Evaluating [{idx + 1} / {len(zarr_files)}]: {res_file}")
        res_file_name = res_file.name
        res_file = res_file.as_posix()
        gt_file_path = (gt_root / res_file_name).as_posix()
        # print("res_file_path:", zarr_file_path)
        print("res_file_name:", res_file_name)
        print("res_file_path:", res_file)
        print("gt_file_path:", gt_file_path)
        metrics = evaluate_file(
            res_file=res_file,
            res_key="volumes/pred_instance",
            gt_file=gt_file_path,
            gt_key="volumes/gt_instances",
            out_dir=out_dir,
            ndim=3,
            localization_criterion="cldice",
            assignment_strategy="greedy",
            remove_small_components=500,
            evaluate_false_labels=True,
            add_general_metrics=[
                "avg_gt_skel_coverage",
                "avg_f1_cov_score",
                "false_merge",
                "false_split",
                "avg_gt_cov_dim",
                "avg_gt_cov_overlap",
            ],
            fm_thresh=0.1,
            fs_thresh=0.05,
            eval_dim=True,
            visualize_type="neuron",
            partly=False,
            from_scratch=args.from_scratch,
        )
        metric_file = out_dir / f"{res_file_name}.toml"
        print(f"Dumping metrics to toml file ({metric_file})")
        toml.dump(metrics, open(metric_file, "w"))
        print("Metrics:", metrics)
        print("Confusion_matrix:", metrics['confusion_matrix'])
        print("General:", metrics['general'], '\n')

if __name__ == "__main__":
    print("Starting evaluation...")
    total_start_time = time() 
    args = get_args()
    main(args)
    total_end_time = time()
    print(f"Total time taken: {total_end_time - total_start_time} seconds")
