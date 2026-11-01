# Shared env for mcfo brain synth SLURM jobs. Sourced by *.sbatch scripts and
# mcfo_submit.sh. Override any of these at submit time, e.g.:
#   RUN_DIR=lm2lineage/runs/brain_5k N_IMAGES=5000 N_SHARDS=32 ./sbatch/mcfo/mcfo_submit.sh
#   RUN_DIR=lm2lineage/runs/brain_v1 N_IMAGES=1500 N_SHARDS=32 ./sbatch/mcfo/mcfo_submit.sh
#
# Defaults below are smoke-scale; raise N_IMAGES / N_SHARDS and set
# MAX_NEURONS=0 for a full production run.

# ---------------------------------------------------------------------------
# Paths / runtime environment
# ---------------------------------------------------------------------------

# Repo root (absolute). Used to resolve relative RUN_DIR and default paths.
REPO_ROOT="${REPO_ROOT:-/scratch/wmz2007/neuroinfo_fruitfly}"

# Conda hook script sourced before activating the synth env.
CONDA_SH="${CONDA_SH:-/scratch/wmz2007/miniforge3/etc/profile.d/conda.sh}"

# Conda env that has mcfo_sim deps (not the Singularity ppp overlay).
CONDA_ENV="${CONDA_ENV:-lm2lineage}"

# Working directory for this run. Relative values are resolved against REPO_ROOT.
# Holds cache/, pool/, manifest/, rendered/ by default.
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/lm2lineage/runs/brain_v1}"
case "${RUN_DIR}" in
  /*) ;;
  *) RUN_DIR="${REPO_ROOT}/${RUN_DIR}" ;;
esac

# ---------------------------------------------------------------------------
# Stage directories (under RUN_DIR unless overridden)
# ---------------------------------------------------------------------------

# Download cache: annotations.feather + swc_jrc2018u/*.swc
CACHE="${CACHE:-${RUN_DIR}/cache}"

# Output of `brain.cli pool` (pool.parquet, footprints.npy, tissue_grid.npz)
POOL_OUT="${POOL_OUT:-${RUN_DIR}/pool}"

# Output of `brain.cli manifest` (manifest_images / manifest_neurons + config)
MANIFEST_OUT="${MANIFEST_OUT:-${RUN_DIR}/manifest}"

# Output of `brain.cli render` (<image_id>.zarr, .json, MIP, skeletons)
RENDER_OUT="${RENDER_OUT:-${RUN_DIR}/rendered}"

# ---------------------------------------------------------------------------
# Synth design / compute knobs (passed into brain.cli)
# ---------------------------------------------------------------------------

# Cap on eligible pool neurons. 0 = keep all (~50k). Smoke default 3000.
MAX_NEURONS="${MAX_NEURONS:-1500}"

# Number of images to design in the manifest (no pixels yet).
N_IMAGES="${N_IMAGES:-12}"

# Render array length: task k runs --shard k/N_SHARDS (0-indexed).
N_SHARDS="${N_SHARDS:-3}"

# Parallel download/parse workers for the pool stage.
WORKERS="${WORKERS:-16}"

# Optional path to calibration.json from `validate calibrate` (FISBe train).
# Empty = sample imaging params without FISBe-fitted ranges.
CALIBRATION="${CALIBRATION:-}"

# ---------------------------------------------------------------------------
# Install into fisbe/ (mcfo_submit.sh install / install_fisbe.sbatch)
# ---------------------------------------------------------------------------

# Parent of <DATASET>/{train,val,test}/ where symlinks are created.
FISBE_ROOT="${FISBE_ROOT:-${REPO_ROOT}/fisbe}"

# Dataset folder name under FISBE_ROOT (usually "synthetic").
DATASET="${DATASET:-synthetic}"

# sample_list_per_split.txt rewritten by install (--update-sample-list).
SAMPLE_LIST="${SAMPLE_LIST:-${REPO_ROOT}/evaluate-instance-segmentation/assets/sample_list_per_split.txt}"

# 1 = pass --force to install (replace existing symlinks that point elsewhere).
# Never deletes real non-symlink directories.
FORCE_INSTALL="${FORCE_INSTALL:-0}"

# ---------------------------------------------------------------------------
# SLURM account / partitions
# ---------------------------------------------------------------------------

# Billing account for all mcfo jobs.
ACCOUNT="${ACCOUNT:-torch_pr_61_general}"

# Default CPU partition for pool, manifest, integrity, install.
PARTITION="${PARTITION:-cs}"

# Partition for the render array (defaults to PARTITION; override if needed).
RENDER_PARTITION="${RENDER_PARTITION:-${PARTITION}}"

# ---------------------------------------------------------------------------
# SLURM resources — pool
# ---------------------------------------------------------------------------

POOL_CPUS="${POOL_CPUS:-32}"           # cpus-per-task (download threads)
POOL_MEM="${POOL_MEM:-64G}"            # memory
POOL_TIME="${POOL_TIME:-04:00:00}"     # walltime

# ---------------------------------------------------------------------------
# SLURM resources — manifest
# ---------------------------------------------------------------------------

MANIFEST_CPUS="${MANIFEST_CPUS:-4}"
MANIFEST_MEM="${MANIFEST_MEM:-16G}"
MANIFEST_TIME="${MANIFEST_TIME:-01:00:00}"

# ---------------------------------------------------------------------------
# SLURM resources — render array
# ---------------------------------------------------------------------------

RENDER_CPUS="${RENDER_CPUS:-4}"        # per array task
RENDER_MEM="${RENDER_MEM:-32G}"        # ~10–12G peak per 0.44 µm image; leave headroom
RENDER_TIME="${RENDER_TIME:-24:00:00}" # walltime per shard
RENDER_PCT="${RENDER_PCT:-8}"          # array concurrency throttle (%N in --array=0-(N-1)%RENDER_PCT)

# ---------------------------------------------------------------------------
# SLURM resources — integrity
# ---------------------------------------------------------------------------

INTEGRITY_CPUS="${INTEGRITY_CPUS:-4}"
INTEGRITY_MEM="${INTEGRITY_MEM:-32G}"
INTEGRITY_TIME="${INTEGRITY_TIME:-02:00:00}"

# ---------------------------------------------------------------------------
# SLURM resources — install (symlink only; light)
# ---------------------------------------------------------------------------

INSTALL_CPUS="${INSTALL_CPUS:-2}"
INSTALL_MEM="${INSTALL_MEM:-4G}"
INSTALL_TIME="${INSTALL_TIME:-00:30:00}"

# ---------------------------------------------------------------------------
# Job helper: activate conda, set PYTHONPATH, cd to RUN_DIR
# ---------------------------------------------------------------------------

mcfo_activate() {
  mkdir -p "${RUN_DIR}" "${CACHE}"
  mkdir -p "${REPO_ROOT}/sbatch/mcfo/logs"
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  export PYTHONPATH="${REPO_ROOT}/lm2lineage/mcfo_sim${PYTHONPATH:+:${PYTHONPATH}}"
  cd "${RUN_DIR}"
  echo "mcfo_env RUN_DIR=${RUN_DIR} CACHE=${CACHE} POOL_OUT=${POOL_OUT} MANIFEST_OUT=${MANIFEST_OUT} RENDER_OUT=${RENDER_OUT}"
  echo "mcfo_env MAX_NEURONS=${MAX_NEURONS} N_IMAGES=${N_IMAGES} N_SHARDS=${N_SHARDS} WORKERS=${WORKERS}"
}
