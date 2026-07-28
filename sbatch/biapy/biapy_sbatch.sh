#!/bin/bash
#SBATCH --job-name=BiaPy
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/biapy/biapy-%j.out
#SBATCH --error=sbatch/biapy/biapy-%j.err
set -e
module purge
# config_folder="train_Dn_0707"
config_folder="augment_small"
# config_folder="augment_medium"
# config_folder="augment_all"
echo "SBATCH Run: ${config_folder}"

# Train Model
singularity exec --nv \
    --overlay env/BiaPy_env.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate BiaPy_env; \
    echo \"running ./BiaPy/biapy.sh\"; \
    ./BiaPy/biapy.sh -c \"$config_folder\""

# Test Model on Test Data
singularity exec --nv \
    --overlay env/BiaPy_env.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate BiaPy_env; \
    echo \"running ./BiaPy/biapy.sh\"; \
    ./BiaPy/biapy.sh -c \"$config_folder\" -t test"

# Test Model on Train Data
singularity exec --nv \
    --overlay env/BiaPy_env.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate BiaPy_env; \
    echo \"running ./BiaPy/biapy.sh\"; \
    ./BiaPy/biapy.sh -c \"$config_folder\" -t train"
