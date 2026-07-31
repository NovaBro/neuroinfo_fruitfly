#!/bin/bash
#SBATCH --job-name=data-BiaPy-channel-rot
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --mem=128g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/data/biapy-py_fisbe_prep-%j.out
#SBATCH --error=sbatch/data/biapy-py_fisbe_prep-%j.err
set -e
module purge
# >>>> Set Job Config >>>>
output_dir="fisbe/biapy-data_channel-rot"
# <<<< Set Job Config <<<<

echo "Output Data to $output_dir"

# # Generate all augmented biapy-py_prep.py Data
# singularity exec --nv \
#     --overlay env/BiaPy_env.ext3:ro \
#     /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
#     /bin/bash -c "source /ext3/env.sh; \
#     conda activate BiaPy_env; \
#     echo \"running BiaPy/biapy-py_prep.py\"; \
#     python3 -u BiaPy/biapy-py_prep.py \
#         -o ${output_dir} \
#         -s train val test\
#         -v info \
#         -a \
#         -c \
#         -w ${SLURM_CPUS_PER_TASK}"

# Generate augmented biapy-py_prep.py train data
# singularity exec --nv \
#     --overlay env/BiaPy_env.ext3:ro \
#     /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
#     /bin/bash -c "source /ext3/env.sh; \
#     conda activate BiaPy_env; \
#     echo \"running BiaPy/biapy-py_prep.py\"; \
#     python3 -u BiaPy/biapy-py_prep.py \
#         -o ${output_dir} \
#         -s train \
#         -v info \
#         -a \
#         -c \
#         -w ${SLURM_CPUS_PER_TASK}"

# Generate non-augmented biapy-py_prep.py test val data
singularity exec --nv \
    --overlay env/BiaPy_env.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; \
    conda activate BiaPy_env; \
    echo \"running BiaPy/biapy-py_prep.py\"; \
    python3 -u BiaPy/biapy-py_prep.py \
        -o ${output_dir} \
        -s test val \
        -v info \
        -c \
        -w ${SLURM_CPUS_PER_TASK}"


