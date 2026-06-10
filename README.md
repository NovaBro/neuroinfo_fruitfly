# README

## Overall TODO:
- [X] Finish Reading FISBe Paper
- [ ] Research unsupervised segmentation methods

- [ ] FISBe EDA
- [ ] Run / Train / Test / Segmentation Models
- [ ] Skeletonize Segmentation Outputs
- [ ] Run NBLAST on Skeletonized Outputs

## Google Drive TODO:
- [ ] Segment the MCFO data into individual neurons William Zheng
    - [ ] Filter then segment neurons based on colors, isolate color channels, RGB
    - [ ] Track hyperparameters used in the FISBe paper, 
    - [ ] Track citations for the FISBe dataset
    - [ ] Generate a bunch of interactive visualizations, compare neuron segmentations across different segmentations
    - [ ] Models to Try for Segmentation
        - [ ] BIApy (General Segmentation): https://biapy.readthedocs.io/en/latest/index.html
        - [ ] PatchPerPix: https://arxiv.org/pdf/2001.07626
        - [ ] https://github.com/Kainmueller-Lab/PatchPerPix
        - [ ] Flood Filling Networks
        - [ ] https://github.com/google/ffn

- [ ] Training / Test: FISBe https://kainmueller-lab.github.io/fisbe/
- [ ] Test on MCFO Data
    - [ ] MCFO processing into zarr file

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

## PatchPerPix

```bash
# Base
cd PatchPerPix/experiments/
nohup env CUDA_VISIBLE_DEVICES=0 \
python -u run_ppp.py \
--setup setup01 \
--config flylight/setups/setup01/default_train_code.toml \
-d train validate_checkpoints predict decode label evaluate \
--app flylight \
--root ppp_experiments \
--test-checkpoint last \
> running.txt 2>&1 &

# For evaluation only on an existing experiment (after you have a real checkpoint):
cd PatchPerPix/experiments/
nohup env CUDA_VISIBLE_DEVICES=0 \
python -u run_ppp.py \
-id ppp_experiments/flylight_setup01_260609_133559_546752 \
--run_from_exp \
-d predict decode label evaluate \
--app flylight \
--checkpoint 4000 \
> evaluate.txt 2>&1 &
```


## BiaPy Tutorial
https://biapy.readthedocs.io/en/latest/workflows/semantic_segmentation.html

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

