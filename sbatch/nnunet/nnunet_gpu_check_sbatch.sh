#!/bin/bash
#SBATCH --job-name=nnunet-gpu-check
#SBATCH --account=torch_pr_61_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=00:20:00
#SBATCH --output=sbatch/nnunet/logs/%x-%j.out
#SBATCH --error=sbatch/nnunet/logs/%x-%j.err

# Short GPU smoke test of env/nnunet.ext3: CUDA visible, a 3D conv runs under AMP, and
# torch.compile works in the CUDA 12.1 image. Submit from repo root:
#   sbatch sbatch/nnunet/nnunet_gpu_check_sbatch.sh

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from the repo root}"
cd "${REPO_ROOT}"
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton" TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/inductor"
nvidia-smi

singularity exec --nv --overlay env/nnunet.ext3:ro "${SIF}" /bin/bash -s <<'INNER'
set -eo pipefail
source /ext3/env.sh
conda activate /ext3/miniforge3/envs/nnunet
python - <<'PY'
import time, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
# Roughly the ResEnc-L topology nnU-Net plans for FISBe (6 stages, 3 in-channels, 3 region outputs).
net = ResidualEncoderUNet(
    input_channels=3, n_stages=6, features_per_stage=(32, 64, 128, 256, 320, 320),
    conv_op=torch.nn.Conv3d, kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2),
    n_blocks_per_stage=(1, 3, 4, 6, 6, 6), num_classes=3, n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
    conv_bias=True, norm_op=torch.nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
    nonlin=torch.nn.LeakyReLU, nonlin_kwargs={"inplace": True}, deep_supervision=True).cuda()
x = torch.randn(2, 3, 128, 128, 128, device="cuda")
for name, m in [("eager", net), ("compiled", torch.compile(net))]:
    for i in range(3):
        torch.cuda.synchronize(); t = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            out = m(x)
        sum(o.float().mean() for o in out).backward()
        torch.cuda.synchronize()
        print(f"{name} iter {i}: {time.time() - t:.2f} s, max mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
print("gpu check OK")
PY
INNER
