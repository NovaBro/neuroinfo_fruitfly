# neuroinfo_fruitfly

Research workspace for **instance segmentation of fruit-fly (Drosophila) neurons** from the
[FISBe](https://kainmueller-lab.github.io/fisbe/) light-microscopy dataset, run on **NYU Greene HPC**.

## Goal

Segment FISBe / MCFO volumes into individual neurons → skeletonize → run NBLAST.

There is no single application here; it is a collection of independently runnable subprojects that
share the FISBe data and the HPC environment.

## Repo layout

| Path | Role |
|------|------|
| [`PatchPerPix/`](PatchPerPix/) | Proposal-free instance segmentation (see [`CODEBASE.md`](PatchPerPix/CODEBASE.md)) |
| [`biapy_work_folder/`](biapy_work_folder/) | BiaPy 3D instance-seg configs, prep, Ray Tune, watershed tools |
| [`web/`](web/) | FastAPI + React viewer for FISBe Zarrs and prediction overlays |
| [`sbatch/`](sbatch/) | SLURM job scripts (`ppp/`, `biapy/`, `evalinstseg/`, `web/`, `vesselfm/`) |
| [`md_guides/`](md_guides/) | Experiment notes (PPP layers, Ray Tune, VesselFM, losses) |
| [`ipynb/`](ipynb/) | Exploratory notebooks (EDA, augmentation review, metrics) |
| [`evaluate-instance-segmentation/`](evaluate-instance-segmentation/) | Upstream eval tooling (`evalinstseg`) |
| `metrics/` | Run outputs (gitignored): `metrics/ppp/`, `metrics/biapy/` |
| `env/` | Singularity ext3 overlays (gitignored) |
| `fisbe/` | Dataset (gitignored) |

## HPC / containers (required)

**Do not run GPU or heavy I/O work on a login node.** Jobs use a Singularity CUDA image plus a
per-project **ext3 overlay** that holds the conda env.

- CUDA image: `/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif`
- Overlays under `env/` (examples): `ppp.ext3` → env `ppp`, `BiaPy_env.ext3` → `BiaPy_env`, `webdev.ext3`
- Inside the container: `source /ext3/env.sh; conda activate <env>`
- Submit **from the repo root** (`/scratch/wmz2007/neuroinfo_fruitfly`) so `sbatch` `--output`/`--error` paths resolve
- Account: `torch_pr_61_general`; typically `--gres=gpu:1`
- GPU jobs should keep utilization high (public partitions cancel idle GPUs). Job scripts usually background an `nvidia-smi … -l … > gpu_usage_log_*.csv` logger — keep that when copying patterns
- Install packages into an overlay with `:rw --fakeroot` (one writer at a time). Do **not** combine `:rw` with `--writable-tmpfs` on apptainer 1.5.1

Canonical launch pattern:

```bash
singularity exec --nv --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c 'source /ext3/env.sh; conda activate ppp; <command>'
```

Overlay setup tutorial:
https://services.rt.nyu.edu/docs/hpc/containers/singularity_with_conda/#using-your-singularity-container-in-a-slurm-batch-job

Upstream installs (inside the matching overlay):

- BiaPy: https://biapy.readthedocs.io/en/latest/get_started/installation.html
- PatchPerPix: https://github.com/Kainmueller-Lab/PatchPerPix
- evaluate-instance-segmentation: https://github.com/Kainmueller-Lab/evaluate-instance-segmentation

## Dataset

Volumes are **Zarr, CZYX**, under `fisbe/completely/{train,val,test}/` (and `fisbe/partly/`).
The split manifest is `fisbe/sample_list_per_split.txt`.

### FISBe download

```bash
nohup bash -c '
  mkdir -p fisbe &&
  echo "[$(date)] Starting download..." &&
  curl -L -s -w "[$(date)] Download complete: %{size_download} bytes, %{time_total}s\n" \
    https://zenodo.org/api/records/10875063/files-archive -o fisbe/archive.zip &&
  echo "[$(date)] Extracting..." &&
  unzip -o fisbe/archive.zip -d fisbe/ &&
  rm fisbe/archive.zip &&
  echo "[$(date)] Done."
' > download.txt 2>&1 &

echo "PID: $!"
```

## PatchPerPix

Architecture reference: [`PatchPerPix/CODEBASE.md`](PatchPerPix/CODEBASE.md).
Experiment notes: [`md_guides/ppp_*.md`](md_guides/).

### One-time setup

```bash
# Inside the ppp overlay
cd PatchPerPix && pip install -e .
# Pin pycuda for Python 3.9 (2025.1+ breaks label/vote_instances)
pip install 'pycuda<=2024.1.2'

# Add FISBe arrays expected by PPP (raw_normalized, gt_instances_rm_5, …)
python3 PatchPerPix/experiments/flylight/prepare_fisbe_for_ppp.py \
  --fisbe-root fisbe --opening-radius 1
```

### Run (SLURM)

From the repo root:

```bash
# Default / exploratory job (see script for staging, GPU logger, configs)
sbatch sbatch/ppp/ppp_sbatch.sh

# Example chain: long8h train then infer (afterok)
bash sbatch/ppp/ppp_basic_long8h_chain.sh
```

Entry point is `PatchPerPix/experiments/run_ppp.py` with `-d train validate_checkpoints predict decode label evaluate`.
Configs live under `PatchPerPix/experiments/flylight/setups/setup01/` (base `default_train_code_l40s.toml` plus overlays such as `basic_long8h.toml`, `vi_th_0_*.toml`).

Active experiment roots are under `metrics/ppp/` (gitignored). Older runs may use `PatchPerPix/experiments/ppp_experiments/`.

Prefer **separate jobs** for train vs predict/label (do not `@fork` train with PreCache after CUDA init). See comments in `run_ppp.py` and the `sbatch/ppp/ppp_basic_*` scripts.

## BiaPy

Docs: https://biapy.readthedocs.io/en/latest/workflows/semantic_segmentation.html

### Data prep

```bash
python biapy_work_folder/biapy_prep_main.py -o fisbe/biapy-channel-scale-zarr ...
```

### Train / test (SLURM)

Configs: `biapy_work_folder/configs/<stem>.yaml` (e.g. `biapy-aug-zarr-seunet-FDb-skel`, winner YAMLs from Ray Tune).

```bash
# From repo root; stem = YAML basename without .yaml
./sbatch/biapy/biapy-py_sbatch_chain.sh biapy-aug-zarr-seunet-FDb-skel train test
./sbatch/biapy/biapy-py_sbatch_chain.sh -r 0 <stem> preprocessing train test
```

Runner: [`biapy_work_folder/run_biapy-py.py`](biapy_work_folder/run_biapy-py.py).
Outputs: `metrics/biapy/` (preferred); legacy under `biapy_work_folder/results/`.

**Warning:** changing `PROBLEM.INSTANCE_SEG.DATA_CHANNELS` or `DATA_CHANNELS_EXTRA_OPTS` regenerates the data cache — avoid concurrent jobs writing the same cache dir.

### Ray Tune and watershed

- Hyperparameter search: [`biapy_work_folder/raytune/`](biapy_work_folder/raytune/) + `sbatch/biapy/raytune_*_sbatch.sh`
- Instance watershed probes: [`biapy_work_folder/watershed_tune/`](biapy_work_folder/watershed_tune/) + `sbatch/biapy/watershed_tune_*.sh`
- Notes: [`md_guides/raytune_*.md`](md_guides/), [`md_guides/biapy_watershed_param_tune.md`](md_guides/biapy_watershed_param_tune.md)

## Evaluation

For BiaPy TIFF/instance outputs, convert to Zarr before scoring:

```bash
python biapy_work_folder/evalinstseg_prep_zarr.py ...
```

Submit via:

```bash
sbatch sbatch/evalinstseg/evalinstseg_sbatch.sh
```

Example (`evalinstseg` from the evaluate-instance-segmentation env):

```bash
evalinstseg \
  --res_file path/to/per_image_instances_zarr \
  --res_key volumes/pred_instance \
  --gt_file fisbe/completely/test \
  --gt_key volumes/gt_instances \
  --split_file fisbe/sample_list_per_split.txt \
  --out_dir tests/results/biapy \
  --app flylight
```

## Web viewer

Browse FISBe volumes and overlay BiaPy / PatchPerPix predictions.

```bash
sbatch sbatch/web/web.sh
```

Full setup, tunneling, and API details: [`web/README.md`](web/README.md).

## Guides and notebooks

- Experiment write-ups: [`md_guides/`](md_guides/)
- Augmentation / metric notebooks: [`ipynb/`](ipynb/), helpers under [`ipynb/view_augments/`](ipynb/view_augments/)
