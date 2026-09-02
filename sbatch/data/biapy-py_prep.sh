#!/bin/bash
#SBATCH --job-name=biapy-debug-zarr
#SBATCH --cpus-per-task=12
#SBATCH --time=2:00:00
#SBATCH --mem=256g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/data/%x-%j.out
#SBATCH --error=sbatch/data/%x-%j.err
set -e
module purge
# >>>> Set Job Config >>>>
output_dir="fisbe/${SLURM_JOB_NAME}"
# <<<< Set Job Config <<<<

echo "Output Data to $output_dir"


# ==== Select Split Generation ====
# Generate all augmented biapy-py_prep.py Data
singularity exec --nv \
    --overlay env/BiaPy_env.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; \
    conda activate BiaPy_env; \
    echo \"running biapy_work_folder/biapy_prep_main.py\"; \
    python3 -u biapy_work_folder/biapy_prep_main.py \
        -o ${output_dir} \
        -s train val test \
        --max-num-samples 2 \
        -v info \
        -c \
        -w ${SLURM_CPUS_PER_TASK} \
        -l ${SLURM_JOB_NAME}"

# --max-num-samples 2 \

# Channel-Rot
# --channel-flip \
# --num-channel-flips 3 \
# --rotate \
# --num-axes 3 \
# --num-rotations 3 \

# Channel Scale
# --channel-scale \
# --num-channel-scales 4 \
# --channel-scale-range \ MUST PROVIDE VALUES

# Instance Scale
# --instance-scale \
# --num-instance-scales 4 \
# --instance-scale-range \ MUST PROVIDE VALUES


