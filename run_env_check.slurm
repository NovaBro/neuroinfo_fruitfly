#!/bin/bash
#SBATCH --job-name=env_check
#SBATCH --output=logs/env_check_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --account=torch_pr_61_general

cd /scratch/vh2308/GitHub/Repos/neuroinfo_fruitfly
mkdir -p logs

singularity exec --fakeroot \
  --overlay /scratch/vh2308/conda-envs/test_env/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -c "source /ext3/env.sh && conda activate /ext3/miniforge3/envs/neurofly_2 && python -c '
import seaborn, plotly, pandas, numpy
print(\"seaborn\", seaborn.__version__)
print(\"plotly \", plotly.__version__)
print(\"pandas \", pandas.__version__)
'"
