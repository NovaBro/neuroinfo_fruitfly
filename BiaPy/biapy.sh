#!/bin/bash

# Configuration file
job_cfg_file=./BiaPy/3d_instance_segmentation.yaml
# Where the experiment output directory should be created
result_dir=./BiaPy/results
# Just a name for the job
job_name=3d_instance_segmentation
# Number that should be increased when one need to run the same job multiple times (reproducibility)
job_counter=1
# Number of the GPU to run the job in (according to 'nvidia-smi' command)
gpu_number=0

# Load the environment
# conda activate BiaPy_env

python ./BiaPy/run_biapy.py \
    --config $job_cfg_file \
    --result_dir $result_dir  \
    --name $job_name    \
    --run_id $job_counter  \
    --gpu "$gpu_number"