# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A research workspace for **instance segmentation of fruit-fly (Drosophila) neurons** from the
[FISBe](https://kainmueller-lab.github.io/fisbe/) light-microscopy dataset, run on **NYU Greene HPC**.
The end goal (see `README.md` TODO) is: segment FISBe / MCFO volumes into individual neurons →
skeletonize → run NBLAST. There is no single application here; it is a collection of
independently-runnable subprojects that share the FISBe data and the HPC environment.

The three things actually being worked on:
1. **PatchPerPix** (`PatchPerPix/`) — proposal-free instance segmentation. Has its own detailed
   architecture guide: **read `PatchPerPix/CODEBASE.md` before touching anything under `PatchPerPix/`.**
2. **BiaPy** (`biapy_work_folder/`) — off-the-shelf 3D instance-segmentation workflow driven by a YAML config.
3. **Web viewer** (`web/`) — FastAPI + React app for browsing FISBe Zarr volumes and overlaying
   model predictions. Has its own `web/README.md`.

Supporting code: data download/prep scripts, FISBe EDA notebooks (`ipynb/`), skeletonization
(`create_skeleton.py`, `skeletons/`), and the `evaluate-instance-segmentation`/`fisbe` eval repos.

## HPC execution model (read this first)

**Nothing GPU-bound runs on a login node or in a bare shell.** Work runs inside **Singularity
containers** with a CUDA `.sif` image plus a per-project **ext3 overlay** that carries a conda env.
This is the single most important convention in the repo.

- CUDA image: `/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif`
- Overlays live in `env/` (gitignored): `ppp.ext3` (conda env `ppp`), `BiaPy_env.ext3`
  (conda env `BiaPy_env`), `webdev.ext3`, plus generic `overlay-15GB-500K.ext3`.
- Activation pattern inside every container: `source /ext3/env.sh; conda activate <env>`.

Canonical launch (from `sbatch/ppp/ppp_sbatch.sh`):

```bash
singularity exec --nv --overlay env/ppp.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c 'source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments/; \
    python3 -u run_ppp.py --setup setup01 \
      --config flylight/setups/setup01/default_train_code_l40s.toml \
      -d train validate_checkpoints predict decode label evaluate \
      --app flylight --root ppp_experiments --test-checkpoint last'
```

SLURM jobs are under `sbatch/` (`ppp/`, `biapy/`), each pairing a `*_sbatch.sh` with its
`.out`/`.err`. Account `torch_pr_61_general`, typically `--gres=gpu:1`. To install packages into an
overlay, mount it writable with `--overlay env/<x>.ext3:rw --fakeroot` (no `--writable-tmpfs` — on
apptainer 1.5.1 that combo is rejected and plain `:rw` hits "upper dir not writable"). The ppp sbatch also
backgrounds an `nvidia-smi ... -l 1 > gpu_usage_log.csv` GPU-utilization logger — keep that when
copying the pattern (NYU partitions cancel low-utilization jobs).

To submit: `sbatch sbatch/ppp/ppp_sbatch.sh`. The script paths are relative to the **repo root**, so
submit from `/scratch/wmz2007/neuroinfo_fruitfly`, not from inside `sbatch/`.

## PatchPerPix

`PatchPerPix/CODEBASE.md` is the authoritative architecture reference (pipeline, file call-graph,
TOML config sections, per-script function tables). Do not duplicate it here. Key facts to know up front:

- Install once: `cd PatchPerPix && pip install -e .` (registers the `PatchPerPix` package so
  `experiments/run_ppp.py` can import it).
- **Pin `pycuda<=2024.1.2` in the `ppp` overlay** (env is Python 3.9; pycuda 2025.1+ dropped 3.9
  support). With a newer pycuda the `label`/`vote_instances` step dies importing `pycuda.autoinit`
  (`TypeError: unsupported operand type(s) for |` in `compyte/dtypes.py`) → `child process died`.
- One entry point, `experiments/run_ppp.py`, driven by `-d/--do` task list:
  `train validate_checkpoints predict decode label evaluate`. It dynamically imports
  `experiments/flylight/setups/setup01/{train,predict_no_gp,decode}.py` by naming convention.
- Config is TOML; the active one is
  `experiments/flylight/setups/setup01/default_train_code_l40s.toml`. Experiment outputs go under
  `experiments/ppp_experiments/<exp_id>/` (gitignored), each with its own `config.toml`.
- Resume / re-run a specific experiment with `-id ppp_experiments/<exp_id>` (optionally
  `--run_from_exp`, `--checkpoint <N>` or `--test-checkpoint last`).
- Data prep is a separate one-time step:
  `python PatchPerPix/experiments/flylight/prepare_fisbe_for_ppp.py --fisbe-root fisbe --opening-radius 1`
- The learning side (PyTorch + gunpowder, in `experiments/`) and the assembly side (CUDA graph
  algorithm, in `PatchPerPix/vote_instances/`) communicate **only through Zarr files on disk**.

## BiaPy

YAML-config-driven workflow. The runner is [`biapy_work_folder/run_biapy-py.py`](biapy_work_folder/run_biapy-py.py),
launched via [`sbatch/biapy/biapy-py_sbatch_chain.sh`](sbatch/biapy/biapy-py_sbatch_chain.sh) with configs under
`biapy_work_folder/configs/`. Active run outputs go to `metrics/biapy/`; legacy outputs may live under
`biapy_work_folder/results/`. Data prep:

```bash
python biapy_work_folder/biapy_prep_main.py -o fisbe/biapy-channel-scale-zarr ...
```

Outputs under `metrics/biapy/...` and `biapy_work_folder/results/...` are consumed by the web viewer.

## Web viewer

See `web/README.md`. Two processes:
- `web/server/` — FastAPI; reads Zarr and serves slices/MIPs/downsampled volumes (it never ships
  full volumes to the browser). Run: `uvicorn main:app --reload --port 8000`. Configured via
  `FISBE_ROOT`, `SAMPLE_LIST_PATH`, `BIAPY_RESULT_ROOT` env vars (`web/server/config.py`). Zarr
  access logic is in `web/server/services/zarr_reader.py`; BiaPy overlay loading in
  `services/biapy_loader.py` (shares helpers with `ipynb/scripts/biapy.py`).
- `web/client/` — Vite + React + TS. `npm install` then `npm run dev` (port 5173). The 3D viewer is
  `web/client/src/components/VolumeViewer3D.tsx`.

## Data

FISBe data lives under `fisbe/` (**gitignored** — download via the `nohup curl … zenodo …` recipe in
`README.md`). Volumes are **Zarr, CZYX layout**, split into `train/`/`val/`/`test/` under
`fisbe/completely/` (and `fisbe/partly/`). PatchPerPix prep adds arrays
(`raw_normalized`, `gt_instances_rm_5`, `gt_numinst`, `gt_fg_rm_5`) into each `.zarr`.
The split manifest is `fisbe/sample_list_per_split.txt` (mirrored under
`evaluate-instance-segmentation/assets/`). A small committed sample lives at
`JRC_SS05008-20160318_24_B2_crop.zarr/`.

## Conventions & gotchas

- **`PatchPerPix/experiments/ppp_experiments/`, `env/`, `fisbe/`, `downloads/`, and `web` build/venv
  dirs are gitignored.** Don't add generated experiment output, overlays, or datasets to git.
- Long-running jobs are commonly launched with `nohup … > <name>.txt 2>&1 &` and their logs
  (`running.txt`, `nohup.out`, `evaluate.txt`, `job.txt`) tracked manually — these are scratch logs,
  not source.
- Notebooks in `ipynb/` are exploratory (FISBe EDA, split EDA, GPU-util analysis from
  `gpu_usage_log.csv`); reusable notebook code is factored into `ipynb/scripts/`.
- `CLAUDE_SETUP_GUIDE.md` documents the author's preferred Claude Code workflow (CLAUDE.md hierarchy,
  memory system, container-first rules) — a process doc, not part of the pipeline.
