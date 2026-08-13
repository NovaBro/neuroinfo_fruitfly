#!/bin/bash
#SBATCH --job-name=EI-BP-v1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=32g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/evalinstseg/%x-%j.out
#SBATCH --error=sbatch/evalinstseg/%x-%j.err

module purge

singularity exec \
    --overlay env/evaluate.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; \
    conda activate evalinstseg; \
    cd evaluate-instance-segmentation; \
    python -u biapy_eval.py \
    --config-name=biapy-v1-no-aug \
    --job-name=biapy-v1-no-aug \
    --run-id=0 \
    --from-scratch"


