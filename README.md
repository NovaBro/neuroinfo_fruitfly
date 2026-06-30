# README

# Dataset
## Fisbe Download Method
```bash
nohup bash -c '
  mkdir -p fisbe &&
  echo "[$(date)] Starting download..." &&
  curl -L -s -w "[$(date)] Download complete: %{size_download} bytes, %{time_total}s\n" \
    https://zenodo.org/api/records/10875063/files-archive -o fisbe/archive.zip &&
  echo "[$(date)] Extracting..." &&
  unzip -o fisbe/archive.zip -d fisbe/ &&
  rm fisbe/archive.zip &&
  echo "[$(date)] Done."
' > download.txt 2>&1 &

echo "PID: $!"
```

# Environment Setup
## Conda Singularity
Follow this tutorial
https://services.rt.nyu.edu/docs/hpc/containers/singularity_with_conda/#using-your-singularity-container-in-a-slurm-batch-job

For each model method and the evaluation method, you need to create a new environment.
BiaPy: https://biapy.readthedocs.io/en/latest/get_started/installation.html
Fisbe Evaluate Instance Segmentation: https://github.com/Kainmueller-Lab/evaluate-instance-segmentation
PatchPerPix: https://github.com/Kainmueller-Lab/PatchPerPix

# Model
## PatchPerPix
```bash
# data preperation
# /scratch/wmz2007/neuroinfo_fruitfly/PatchPerPix/experiments/flylight/prepare_fisbe_for_ppp.py
# /scratch/wmz2007/neuroinfo_fruitfly/fisbe
python3 PatchPerPix/experiments/flylight/prepare_fisbe_for_ppp.py --fisbe-root fisbe --opening-radius 1
```

``` bash
# extra forgotten packages
pip install monai pynrrd torchinfo torchmetrics
```

```bash
# Base
cd PatchPerPix/experiments/
nohup env CUDA_VISIBLE_DEVICES=0 \
python -u run_ppp.py \
--setup setup01 \
--config flylight/setups/setup01/default_train_code_l40s.toml \
-d train validate_checkpoints predict decode label evaluate \
--app flylight \
--root ppp_experiments \
--test-checkpoint last \
> running.txt 2>&1 &

# For evaluation only on an existing experiment (after you have a real checkpoint):
cd PatchPerPix/experiments/
nohup env CUDA_VISIBLE_DEVICES=0 \
python -u run_ppp.py \
--setup setup01 \
--config flylight/setups/setup01/default_train_code.toml \
-id ppp_experiments/flylight_setup01_260610_113923_103163 \
--run_from_exp \
-d validate_checkpoints predict decode label evaluate \
--app flylight \
--checkpoint 4000 \
> evaluate.txt 2>&1 &
```


## BiaPy Tutorial
https://biapy.readthedocs.io/en/latest/workflows/semantic_segmentation.html

```bash
# datapreprocessing to tiff
# /scratch/wmz2007/neuroinfo_fruitfly/fisbe/biapy/prepare_tiff_data.py
python3 neuroinfo_fruitfly/fisbe/biapy/prepare_tiff_data.py --splits test train val
```

in `./fisbe/biapy/biapy.sh`
```bash
# Configuration file
job_cfg_file=./fisbe/biapy/3d_instance_segmentation.yaml
# Where the experiment output directory should be created
result_dir=./fisbe/biapy/results
# Just a name for the job
job_name=3d_instance_segmentation
# Number that should be increased when one need to run the same job multiple times (reproducibility)
job_counter=1
# Number of the GPU to run the job in (according to 'nvidia-smi' command)
gpu_number=0

# Load the environment
conda activate BiaPy_env

biapy \
    --config $job_cfg_file \
    --result_dir $result_dir  \
    --name $job_name    \
    --run_id $job_counter  \
    --gpu "$gpu_number"

# Then
conda activate BiaPy_env
python fisbe/biapy/prepare_tiff_data.py --splits test --clean
```

## Web Viewer

An isolated web app for browsing FISBe 3D volumes lives in [`web/`](web/). See [`web/README.md`](web/README.md) for setup: a FastAPI server serves Zarr slices/MIPs, and a Vite + React frontend provides a sample browser and orthographic slice viewer.

## Evaluation
NOTE: For `BiaPy` model, must run the `BiaPy/my_metric_prep_util.py` script to get zarr file format instead of tif.

```bash
# Base Example
evalinstseg \
  --res_file tests/pred/sample_01.hdf \
  --res_key volumes/gmm_label_cleaned \
  --gt_file tests/gt/sample_01.zarr \
  --gt_key volumes/gt_instances \
  --split_file assets/sample_list_per_split.txt \
  --out_dir tests/results \
  --app flylight

# Folders
evalinstseg \
  --res_file BiaPy/results/3d_instance_segmentation/results/3d_instance_segmentation_1/per_image_instances_zarr \
  --res_key volumes/pred_instance \
  --gt_file fisbe/completely/train \
  --gt_key volumes/gt_instances \
  --out_dir tests/results/biapy \
  --app flylight

evalinstseg \
  --res_file BiaPy/results/3d_instance_segmentation/results/3d_instance_segmentation_1/per_image_instances_zarr/JRC_SS04989-20160318_24_A2.zarr \
  --res_key volumes/pred_instance \
  --gt_file fisbe/completely/test/JRC_SS04989-20160318_24_A2.zarr \
  --gt_key volumes/gt_instances \
  --out_dir tests/results/biapy \
  --app flylight
```