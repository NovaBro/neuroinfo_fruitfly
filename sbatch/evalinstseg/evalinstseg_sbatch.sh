#!/bin/bash
#SBATCH --job-name=Dn_eval
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=24g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=sbatch/evalinstseg/train_Dn_3d_instance_segmentation_evalinst.out
#SBATCH --error=sbatch/evalinstseg/train_Dn_3d_instance_segmentation_evalinst.err

module purge

# nohup nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 3 > gpu_usage_log_ppp.csv &

# Single file
# singularity exec --nv \
# --overlay env/evaluate.ext3:ro \
# /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
# /bin/bash -c 'source /ext3/env.sh; conda activate evalinstseg; cd evalinstseg\
# evalinstseg \
#   --res_file /scratch/wmz2007/neuroinfo_fruitfly/BiaPy/results/3d_instance_segmentation/results/3d_instance_segmentation_1/per_image_instances_zarr/JRC_SS04989-20160318_24_A2.zarr \
#   --res_key volumes/pred_instance \
#   --gt_file /scratch/wmz2007/neuroinfo_fruitfly/fisbe/completely/test/JRC_SS04989-20160318_24_A2.zarr \
#   --gt_key volumes/gt_instances \
#   --out_dir /scratch/wmz2007/neuroinfo_fruitfly/BiaPy/results/3d_instance_segmentation/results/3d_instance_segmentation_1/tests/results/biapy \
#   --app flylight'

# Folder
singularity exec \
--overlay env/evaluate.ext3:ro \
/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
/bin/bash -c 'source /ext3/env.sh; conda activate evalinstseg; cd evaluate-instance-segmentation; python -u biapy_eval.py --job_name=train_Dn_3d_augmentation'


