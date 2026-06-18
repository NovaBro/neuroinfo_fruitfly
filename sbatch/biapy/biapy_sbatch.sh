#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=32g
#SBATCH --gres=gpu:1
#SBATCH --job-name=BiaPy
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/biapy/biapy.out
#SBATCH --error=sbatch/biapy/biapy.err

module purge

singularity exec --nv \
--overlay env/BiaPy_env.ext3:ro \
/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
/bin/bash -c 'source /ext3/env.sh; conda activate BiaPy_env; ./BiaPy/biapy.sh'
