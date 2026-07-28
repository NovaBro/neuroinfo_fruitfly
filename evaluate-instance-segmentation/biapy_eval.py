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
    print(f"Shape: {arr.shape}")

    # Create or open the Zarr file and store the array
    z = zarr.open_group(zarr_path, mode="w")
    group, dataset = os.path.split(zarr_key)
    g = z.require_group(group) if group else z

    # Remove dataset if it already exists (overwrite)
    # g.create_array(dataset, data=arr, overwrite=True)
    g.create_dataset(dataset, data=arr, overwrite=True)

    print(f"Converted {tiff_path} -> {zarr_path}:{zarr_key}")


def main(args:argparse.Namespace):
    config_name = args.config_name
    job_name = config_name
    run_id = args.run_id
    gt_root = ROOT_DIR / f"fisbe/completely/{args.split}"

    print(f"config_name: {config_name}")
    print(f"run_id: {run_id}")

    experiment_dir = RESULT_ROOT_DIR / f"{config_name}/results/{job_name}_{run_id}"
    assert experiment_dir.exists(), "Experiment doesn't exist! Maybe run it? or wrong path!"
    tiff_dir = experiment_dir / "per_image_instances"
    zarr_dir = experiment_dir / "per_image_instances_zarr"

    out_dir = RESULT_ROOT_DIR / f"{config_name}/results/{job_name}_{run_id}/eval-inst_metrics/{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # If no zarr predictions yet, convert BiaPy TIFFs from per_image_instances
    zarr_files = list(zarr_dir.glob("*.zarr")) if zarr_dir.exists() else []
    tif_files = sorted(
        p for p in tiff_dir.glob("*.tif*") if p.suffix.lower() in {".tif", ".tiff"}
    ) if tiff_dir.exists() else []
    if len(zarr_files) < len(tif_files):
        print(
            f"There are more tif files than zarr files ({len(zarr_files)})! "
            f"Converting ({len(tif_files)}) tif to zarr files now"
        )
        zarr_dir.mkdir(parents=True, exist_ok=True)
        print(f"Number of files to process: {len(tif_files)}")
        for idx, tif in enumerate(tif_files):
            print(f"Converting [{idx} / {len(tif_files)}]: {tif}")
            tiff_to_zarr(tif, zarr_dir / f"{tiff_stem(tif)}.zarr")

    for zarr_file in zarr_dir.glob("*.zarr"):
        zarr_file_name = zarr_file.name
        # zarr_file_path = zarr_file.as_posix()
        gt_file_path = (gt_root / zarr_file_name).as_posix()
        # print("res_file_path:", zarr_file_path)
        print("gt_file_path:", gt_file_path)
        print("res_file:", zarr_file)
        print("res_file_name:", zarr_file_name)
        metrics = evaluate_file(
            # res_file=zarr_file_path,
            res_file=zarr_file,
            res_key="volumes/pred_instance",
            gt_file=gt_file_path,
            gt_key="volumes/gt_instances",
            out_dir=out_dir,
            ndim=3,
            app="flylight",
            partly=False,

            remove_small_components=800
        )
        metric_file = out_dir / f"{zarr_file_name}.toml"
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
