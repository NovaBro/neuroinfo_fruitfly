#!/bin/bash
#SBATCH --job-name=data-BiaPy
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --mem=24g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/data/biapy_fisbe_prep-%j.out
#SBATCH --error=sbatch/data/biapy_fisbe_prep-%j.err
set -e
module purge
# splits="train test val"
# splits_tag="${splits// /_}"
data_output_dir="aug_biapy_s1_a2"
max_num_samples="1"
augment_num_select="2"

echo "Output to $data_output_dir"

splits="train"
splits_tag="${splits}"
# Generate Augmentation
# rm -rf "fisbe/${data_output_dir}/${splits}"
singularity exec --nv \
    --overlay env/BiaPy_env.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate BiaPy_env; echo \"running sbatch/data/biapy_fisbe_prep.sh\"; \
    python3 -u BiaPy/biapy_prep_tiff.py -o fisbe/${data_output_dir} -s ${splits} -l ${data_output_dir}_prep_${splits_tag}.txt \
    -c -a \
    --max-num-samples ${max_num_samples} \
    --augment-num-select ${augment_num_select}"

splits="test"
splits_tag="${splits}"
# Generate Augmentation
# rm -rf "fisbe/${data_output_dir}/${splits}"
singularity exec --nv \
    --overlay env/BiaPy_env.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate BiaPy_env; echo \"running sbatch/data/biapy_fisbe_prep.sh\"; \
    python3 -u BiaPy/biapy_prep_tiff.py -o fisbe/${data_output_dir} -s ${splits} -l ${data_output_dir}_prep_${splits_tag}.txt \
    -c "

splits="val"
splits_tag="${splits}"
# Generate Augmentation
# rm -rf "fisbe/${data_output_dir}/${splits}"
singularity exec --nv \
    --overlay env/BiaPy_env.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate BiaPy_env; echo \"running sbatch/data/biapy_fisbe_prep.sh\"; \
    python3 -u BiaPy/biapy_prep_tiff.py -o fisbe/${data_output_dir} -s ${splits} -l ${data_output_dir}_prep_${splits_tag}.txt \
    -c "
