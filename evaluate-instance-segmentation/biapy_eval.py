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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str)
    parser.add_argument("--job-name", type=str)
    parser.add_argument("--run-id", default=0, type=str)
    parser.add_argument("--split", default='test', help="Either train, test, or val", type=str)

    args = parser.parse_args()
    return args

def main(args:argparse.Namespace):
    config_name = args.config_name
    job_name = config_name
    run_id = args.run_id
    gt_dir = os.listdir(ROOT_DIR / f'fisbe/completely/{args.split}')

    print(f"config_name: {config_name}")
    print(f"run_id: {run_id}")

    experiment_dir = RESULT_ROOT_DIR / f"{config_name}/results/{job_name}_{run_id}"
    assert experiment_dir.exists(), "Experiment doesn't exist! Maybe run it? or wrong path!"
    results_dir = experiment_dir / "per_image_instances_zarr"

    out_dir = RESULT_ROOT_DIR / f"{config_name}/results/{job_name}_{run_id}/tests/eval-inst_metrics/{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check that there are zarr files in res_dir. If not, generate by conversion
    zarr_files = list(results_dir.glob("*.zarr"))
    tif_files = list(results_dir.glob("*.tif"))
    if len(zarr_files) == 0 and len(tif_files) > 0:
        print(
            f"There are no zarr files ({len(zarr_files)})! "
            f"Converting ({len(tif_files)}) tif files now"
        )
        for tif in tif_files:
            print(f"Converting: {tif}")
            np_tif = tifffile.imread(results_dir / tif)
            zarr.save_array(results_dir / tif.name + '.zarr' ,np_tif)
            

        # raise FileNotFoundError(f"No zarr files found in {results_dir}. Please run BiaPy/my_metric_prep_util.py to create them.")

    for res_file in results_dir.glob("*.zarr"):
        res_file_name = res_file.name
        res_file_path = (results_dir / res_file_name).as_posix()
        gt_file_path = (gt_dir / res_file_name).as_posix()
        print("res_file_path:", res_file_path)
        print("gt_file_path:", gt_file_path)
        print("res_file:", res_file)
        print("res_file_name:", res_file_name)
        metrics = evaluate_file(
            res_file=res_file_path,
            res_key="volumes/pred_instance",
            gt_file=gt_file_path,
            gt_key="volumes/gt_instances",
            out_dir=out_dir,
            ndim=3,
            app="flylight",
            partly=False,

            remove_small_components=800
        )
        print("dump metrics to toml file...")
        toml.dump(metrics, open(out_dir / f"{res_file_name}.toml", "w"))
        print("\n", "Metrics:", metrics)
        print("Confusion_matrix:", metrics['confusion_matrix'])
        print("General:", metrics['general'], '\n')

if __name__ == "__main__":
    print("Starting evaluation...")
    total_start_time = time() 
    args = get_args()
    main(args)
    total_end_time = time()
    print(f"Total time taken: {total_end_time - total_start_time} seconds")

    