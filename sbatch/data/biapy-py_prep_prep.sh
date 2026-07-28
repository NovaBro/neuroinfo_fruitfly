#!/bin/bash
#SBATCH --job-name=data-BiaPy
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --mem=32g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/data/biapy-py_fisbe_prep-%j.out
#SBATCH --error=sbatch/data/biapy-py_fisbe_prep-%j.err
set -e
module purge
output_dir="fisbe/biapy-py_data"

echo "Output Data to $output_dir"

# Generate biapy-py_prep.py Data
singularity exec --nv \
    --overlay env/BiaPy_env.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; \
    conda activate BiaPy_env; \
    echo \"running BiaPy/biapy-py_prep.py\"; \
    python3 -u BiaPy/biapy-py_prep.py \
        -o ${output_dir} \
        -s train test val \
        -v info \
        -c"
