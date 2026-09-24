# Ray Tune for BiaPy (scheduler-only, nested sbatch)

Bayesian hyperparameter search over BiaPy YAML configs on NYU Greene, without
running training inside Ray workers. Ray Tune only samples configs and
orchestrates trials; each trial is a nested SLURM GPU job via the existing
`biapy-py_sbatch.sh` path.

## Architecture

```text
sbatch raytune_controller.sh          ← CPU-only, long walltime
  └── python ray_tune_biapy.py        ← Ray Tune + Optuna (Bayesian)
        ├── write configs/trials/<id>.yaml
        ├── sbatch biapy-py_sbatch.sh … train …   ← GPU Singularity + BiaPy
        ├── wait (sacct/squeue)
        ├── parse metric via biapy_train_track.consolidate_training_log
        └── report metric → next Bayesian suggestion
```

| Piece | Role |
|-------|------|
| Ray Tune + Optuna | Search loop / Bayesian suggestions |
| Trial trainable | Write YAML → sbatch → wait → parse metric |
| `biapy-py_sbatch.sh` | Unchanged GPU train (staging, overlay, logger) |

Bayesian search only needs completed `(hparams → metric)` pairs. Mid-trial
reporting / ASHA is not required (and is awkward with full nested jobs). Keep
`max_concurrent_trials` modest (1–2) so the surrogate can guide later trials.

### Suggested file layout

```text
biapy_work_folder/
  raytune/
    __init__.py
    write_trial_config.py   # Stage 1
    slurm_trial.py          # Stage 2
    trainable.py            # Stage 3
    search_space.py         # Stage 6
    run_tune.py             # Stages 3–6 entrypoint
  configs/
    trials/                 # generated YAMLs (gitignore recommended)
sbatch/biapy/
  raytune_controller_sbatch.sh   # Stage 4
metrics/biapy/raytune/<exp>/     # results, best config, trial table
```

Submit the controller from the **repo root** (same convention as other sbatch scripts).

---

## Preconditions (Stage 0)

### Checklist

1. Pick a base YAML (e.g. `biapy-aug-zarr-seunet-FDb-skel.yaml`).
2. Preprocessing already done for that channel/skel cache under
   `fisbe/biapy-no-aug-zarr/{train,val}/label_F*_Sk*_I`.
3. Manual train via `biapy-py_sbatch_chain.sh` / `biapy-py_sbatch.sh` still works.
4. Choose objective: e.g. `min(val_loss)` or `max(val_iou_f)`.
5. Decide search GPU (`h200` vs `l40s`) — batch size / LR are not portable across types
   (see comments in the skel YAMLs).

### Commands to verify

```bash
# From repo root — preprocess once if needed
./sbatch/biapy/biapy-py_sbatch_chain.sh -r 0 biapy-aug-zarr-seunet-FDb-skel preprocessing

# Smoke train (optional short EPOCHS edit in a copy first)
./sbatch/biapy/biapy-py_sbatch_chain.sh -r smoke biapy-aug-zarr-seunet-FDb-skel train
```

### Do not sweep without re-preprocessing

These change the instance-channel cache folder name / contents:

- `PROBLEM.INSTANCE_SEG.DATA_CHANNELS`
- `PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS`
- `LOSS.SKELETON_RECALL.ENABLE` / `TUBE_DILATIONS`
- Channel geometry opts (erosion, decline_power, …)

Parallel jobs must not race writing the same `label_F…` path.

### Safe knobs (first sweep)

| Knob | YAML path |
|------|-----------|
| Learning rate | `TRAIN.LR` |
| Batch size | `TRAIN.BATCH_SIZE` (GPU-conditioned) |
| Scheduler | `TRAIN.LR_SCHEDULER.NAME`, `MIN_LR`, `WARMUP_COSINE_DECAY_EPOCHS` |
| Channel loss weights | `PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS` |
| Skeleton recall weight | `LOSS.SKELETON_RECALL.WEIGHT` |
| Epochs / patience | `TRAIN.EPOCHS`, `TRAIN.PATIENCE` (cap for search) |

---

## Stage 1 — Trial config writer

### Goal

Turn `(base_yaml, hparams, trial_id)` into a hand-submittable YAML under
`biapy_work_folder/configs/trials/<trial_id>.yaml`.

### Why write a file (not only an in-memory dict)

`biapy-py_sbatch.sh` resolves `CONFIG_PATH=biapy_work_folder/configs/${config_file}`
and greps it for `DATA_CHANNELS` when staging channel GT. Nested path
`trials/<id>.yaml` is supported as `$1` to the sbatch script.

### Implementation details

**Module:** `biapy_work_folder/raytune/write_trial_config.py`

Suggested API:

```python
def write_trial_config(
    base_yaml: Path | str,
    hparams: dict,
    trial_id: str,
    *,
    search_epochs: int | None = 40,
    search_patience: int | None = 10,
    out_dir: Path | None = None,
) -> Path:
    """Write configs/trials/<trial_id>.yaml; return path relative to configs/."""
```

Steps inside:

1. `yaml.safe_load` the base config; `copy.deepcopy`.
2. Apply **train-mode flags** (mirror `apply_mode_overrides` in `run_biapy-py.py`):
   - `TRAIN.ENABLE = True`
   - `TEST.ENABLE = False`
   - `MODEL.LOAD_CHECKPOINT = False`
3. Map flat hparams → nested YAML keys, e.g.:
   - `hparams["lr"]` → `TRAIN.LR`
   - `hparams["batch_size"]` → `TRAIN.BATCH_SIZE`
   - `hparams["skel_weight"]` → `LOSS.SKELETON_RECALL.WEIGHT`
   - `hparams["scheduler"]` → `TRAIN.LR_SCHEDULER.NAME`
   - optional `channel_weights` → `PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS`
4. Cap search budget: override `TRAIN.EPOCHS` / `TRAIN.PATIENCE` when
   `search_epochs` / `search_patience` are set (leave full-length for the winner later).
5. `mkdir` `biapy_work_folder/configs/trials/`; `yaml.safe_dump` with a header
   comment listing `trial_id`, base stem, and hparams.
6. Return relative config path for sbatch: `trials/<trial_id>.yaml`.

### Job-name / run-id conventions

Pass these explicitly to sbatch (do **not** rely on auto-derivation from nested
paths):

| Arg | Suggested value | Why |
|-----|-----------------|-----|
| `job_name` (`$3`) | Base stem, e.g. `biapy-aug-zarr-seunet-FDb-skel` | Keeps checkpoints under one BiaPy `name/` tree |
| `run_id` (`$4`) | Sanitized Ray trial id, e.g. `rt_abc123` | Unique outputs; SLURM name uses `-r${run_id}-train` |

Sanitize trial ids for filesystem/SLURM: replace `/` and truncate length.

### Manual test

```bash
python -c 'from biapy_work_folder.raytune.write_trial_config import write_trial_config; ...'
# Then:
sbatch --job-name=... --cpus-per-task=16 --time=24:00:00 --mem=384g \
  --gres=gpu:1 --constraint='h200' \
  sbatch/biapy/biapy-py_sbatch.sh \
  trials/<trial_id>.yaml train biapy-aug-zarr-seunet-FDb-skel <trial_id>
```

**Done when:** a hparam dict produces a YAML that stages and trains without editing by hand.

---

## Stage 2 — SLURM launcher + waiter

### Goal

One Python function: submit a train job, block until it finishes, return a float metric
(and metadata).

### Implementation details

**Module:** `biapy_work_folder/raytune/slurm_trial.py`

Suggested API:

```python
@dataclass
class TrialJobResult:
    job_id: str
    out_path: Path
    metric: float
    state: str  # COMPLETED | FAILED | TIMEOUT | CANCELLED | ...

def submit_train_job(
    config_relpath: str,   # e.g. trials/rt_abc.yaml
    job_name: str,
    run_id: str,
    *,
    account: str = "torch_pr_61_general",
    cpus: int = 16,
    mem: str = "384g",
    time: str = "24:00:00",
    constraint: str = "h200",
    gres: str = "gpu:1",
) -> str:
    """Return SLURM job id from sbatch --parsable."""

def wait_for_job(job_id: str, *, poll_s: float = 60.0, timeout_s: float | None = None) -> str:
    """Poll sacct/squeue until terminal state; return that state."""

def find_train_out(job_name: str, run_id: str, job_id: str) -> Path:
    """Resolve sbatch/biapy/<slurm_job_name>-<job_id>.out."""

def metric_from_out(out_path: Path, metric: str = "val_loss", mode: str = "min") -> float:
    """Parse with consolidate_training_log; aggregate across epochs."""

def run_train_trial(...) -> TrialJobResult:
    """submit → wait → parse; raise on failed/empty."""
```

### Match current train resources

Mirror `biapy-py_sbatch_chain.sh` train block (update if chain changes):

```bash
sbatch --parsable \
  --account=torch_pr_61_general \
  --job-name="${job_name}-r${run_id}-train" \
  --cpus-per-task=16 \
  --time=24:00:00 \
  --mem=384g \
  --gres=gpu:1 \
  --constraint='h200' \
  --output=sbatch/biapy/%x-%j.out \
  --error=sbatch/biapy/%x-%j.err \
  sbatch/biapy/biapy-py_sbatch.sh \
  "${config_relpath}" train "${job_name}" "${run_id}"
```

Notes:

- `#SBATCH --output` in `biapy-py_sbatch.sh` is already `sbatch/biapy/%x-%j.out`;
  CLI `--job-name` sets `%x`. Expected path:
  `sbatch/biapy/<job_name>-r<run_id>-train-<jobid>.out`.
- Always pass `$3` job_name explicitly so nested `trials/…` does not become the BiaPy name.
- Working directory must be **repo root** (`sbatch` inherits cwd; controller should `cd` there).

### Wait / poll

Prefer `sacct -j <id> -n -o State,ExitCode -P` in a loop:

- Terminal success: `COMPLETED` with exit `0:0`
- Terminal failure: `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, `NODE_FAIL`, …
- While pending/running: `PENDING`, `RUNNING`, `CONFIGURING`, …

Backoff: sleep 30–60s. Optional overall timeout → cancel with `scancel` and mark failed.

### Metric extraction

Reuse existing parser (do not reimplement log regexes):

```python
from biapy_work_folder.biapy_train_track import consolidate_training_log

df = consolidate_training_log(out_path)
if df.empty or "val_loss" not in df.columns:
    raise RuntimeError(f"no usable metrics in {out_path}")
# min val_loss or max val_iou_f across epochs / partitions
metric = float(df["val_loss"].min())
```

Available columns after a typical skel train include `val_loss`, `val_iou_f`,
`val_skel_recall`, `train_*`, `lr`, `early_stopped`.

### Error handling

| Case | Behavior |
|------|----------|
| `sbatch` non-zero | Raise immediately; Ray marks trial ERROR |
| Job `FAILED` / OOM | Raise with `.err` tail snippet |
| Empty / unparsable `.out` | Raise (partial train without val stats) |
| Metric column missing | Raise with available columns listed |

### Manual smoke (short epochs)

Write a trial YAML with `EPOCHS: 2`, call `run_train_trial(...)`, print metric.
Confirm staging still resolves exactly one `label_F*_Sk*_I` for the trial YAML
(same `DATA_CHANNELS` as base).

**Done when:** one Python “submit + wait + metric” call works for a short smoke train.

---

## Stage 3 — Ray trainable

### Goal

Wrap Stage 1–2 so Ray Tune can call it as a trial function / Trainable and receive
`{"val_loss": ...}` (or chosen metric).

### Implementation details

**Module:** `biapy_work_folder/raytune/trainable.py`  
**Entrypoint (later):** `biapy_work_folder/raytune/run_tune.py`

Trainable body (function API):

```python
def biapy_sbatch_trainable(config: dict) -> dict:
    trial_id = tune.get_context().get_trial_id()  # or ray.train.get_context()
    run_id = sanitize(trial_id)
    rel = write_trial_config(
        BASE_YAML,
        config,
        run_id,
        search_epochs=config.get("_search_epochs", 40),
    )
    result = run_train_trial(rel, JOB_NAME, run_id, **SLURM_KWARGS)
    return {METRIC_NAME: result.metric, "slurm_job_id": result.job_id}
```

Wire with `Tuner`:

```python
from ray import tune
from ray.tune.search.optuna import OptunaSearch  # Stage 6

tuner = tune.Tuner(
    biapy_sbatch_trainable,
    param_space=search_space,          # Stage 6
    tune_config=tune.TuneConfig(
        num_samples=2,                 # smoke: 1–2
        max_concurrent_trials=1,       # nested jobs; start serial
        metric="val_loss",
        mode="min",
        search_alg=OptunaSearch(),     # after Stage 5–6
    ),
    run_config=tune.RunConfig(
        name="biapy_fdb_skel_smoke",
        storage_path="metrics/biapy/raytune",
    ),
)
results = tuner.fit()
```

### Concurrency model

Each concurrent Ray trial calls `sbatch` and blocks in `wait_for_job`. Ray’s
`max_concurrent_trials` ≈ number of GPU jobs in flight. For Bayesian search prefer
**1–2**. Do not set this equal to a large `num_samples`.

### Resources on the Ray side

Controller Ray actors/tasks should request **CPU only** (e.g. `num_cpus=0.5` or 1).
GPUs belong exclusively to the nested train jobs. Never set `num_gpus` on the
trainable in this pattern.

### Smoke success criteria

- `num_samples=1` or `2`, `max_concurrent_trials=1`
- Trial YAML appears under `configs/trials/`
- Nested GPU job appears in `squeue`
- Metric returned; Ray experiment dir under `metrics/biapy/raytune/`

**Done when:** `Tuner` smoke completes end-to-end without manual sbatch.

---

## Stage 4 — Nested controller job

### Goal

A long-lived **CPU-only** SLURM job that runs `run_tune.py`, which then submits
nested GPU train jobs.

### Implementation

**Script:** [`sbatch/biapy/raytune_controller_sbatch.sh`](../sbatch/biapy/raytune_controller_sbatch.sh)

- Account `torch_pr_61_general`, 4 CPUs, 16G, walltime `7-00:00:00`
- **No `--gres=gpu` / no `--nv`** — controller must not sit idle on a GPU partition
- Overlay: `env/BiaPy_env.ext3:ro`, conda env `BiaPy_env` (Ray installed in **Stage 5**)
- **Controller SIF:** `ubuntu-24.04.4.sif` (glibc ≥ 2.38 so host `/opt/slurm/bin/sbatch`
  works inside the container after Greene RHEL 10). Nested GPU trains still use
  `biapy-py_sbatch.sh`’s CUDA Ubuntu 22.04 image — only the controller needs
  host-SLURM glibc compatibility.
- Bind host SLURM so nested Stage 2 commands work inside Singularity:
  `--bind /opt/slurm`, export host `SLURM_CONF` (Greene conf-cache),
  `--bind /run/munge`, host `/etc/passwd`+`/etc/group` (for `SlurmUser=slurm`),
  and host `libmunge.so.2` into `/usr/lib64` with `LD_LIBRARY_PATH` — without these,
  nested `sbatch` fails (conf / auth_munge plugin)
- Troubleshoot: `GLIBC_2.38 not found` on nested `sbatch` → controller is on an
  old Ubuntu 22.04 SIF; switch back to `ubuntu-24.04.4.sif`
- `RAY_TMPDIR=/tmp/${USER}_ray_${SLURM_JOB_ID}` (NFS breaks Ray sockets)
- `run_tune.py` calls `ray.init(num_cpus=SLURM_CPUS_PER_TASK, …)` under SLURM —
  otherwise Ray auto-detects the whole node (e.g. 128 CPUs) and worker prestart hangs
  inside the cgroup/Singularity overlay before any nested GPU job is submitted
- All CLI args after the script are forwarded to `run_tune.py`; empty args → usage + exit 1

Walltime budget:

`controller_time ≳ (num_samples / max_concurrent) * train_time_limit + slack`

### Submit (from repo root)

```bash
# Dry-run (needs Ray in BiaPy_env; no nested GPU)
sbatch sbatch/biapy/raytune_controller_sbatch.sh \
  --exp biapy_fdb_skel_smoke --num-samples 1 --dry-run \
  --lr 1e-3 --batch-size 30 --skel-weight 0.1

# Short nested GPU smoke
sbatch sbatch/biapy/raytune_controller_sbatch.sh \
  --exp biapy_fdb_skel_smoke --num-samples 1 \
  --search-epochs 2 --search-patience 2 \
  --lr 1e-3 --batch-size 30

# Override controller resources
sbatch --time=1:00:00 --mem=8g sbatch/biapy/raytune_controller_sbatch.sh \
  --exp biapy_fdb_skel_smoke --num-samples 1 --dry-run --lr 1e-3
```

### Nesting policy

- Nested `sbatch` from a batch job is common on Greene; confirm current policy if submit fails.
- Fallback: run the same Singularity recipe under `salloc` (CPU) and invoke
  `python -u biapy_work_folder/raytune/run_tune.py …` interactively; Stage 2–3 unchanged.
- Preflight fails clearly if Ray is missing (complete Stage 5 first).

**Done when:** one controller job submits ≥1 train job and collects a metric unattended
(after Stage 5 Ray install).

---

## Stage 5 — Environment

### Goal

Controller can `import ray` and `import optuna`; GPU train jobs unchanged.

### Chosen approach: A (`BiaPy_env`)

Install Ray+Optuna into **`env/BiaPy_env.ext3`** / conda env **`BiaPy_env`** — the same
overlay Stage 4’s controller already uses. No separate Ray-only overlay.

### Install script

**Script:** [`sbatch/biapy/raytune_env_install_sbatch.sh`](../sbatch/biapy/raytune_env_install_sbatch.sh)

CPU-only job (`cpu_short`, no GPU): `singularity exec --fakeroot --overlay env/BiaPy_env.ext3:rw`
then `pip install -U "ray[tune]" optuna` and verify imports. Uses the same
`ubuntu-24.04.4.sif` as the controller (not the GPU train CUDA 22.04 image).

Conventions:

- Use `:rw --fakeroot` **without** `--writable-tmpfs` (apptainer 1.5.1).
- **One writer at a time** — never mount `BiaPy_env.ext3` from another job while install runs
  (no train/test/controller). Check `squeue -u $USER` first.
- Train / controller jobs keep `:ro` after install.
- Controller sets `RAY_TMPDIR=/tmp/...` (NFS breaks Ray sockets) — see `CLAUDE_SETUP_GUIDE.md`.

### Operator checklist

```bash
# 1. From repo root — exclusive overlay write
sbatch sbatch/biapy/raytune_env_install_sbatch.sh

# 2. Confirm verify line in sbatch/biapy/biapy-raytune-env-install-<jobid>.out
#    e.g. ray <ver> optuna <ver> OptunaSearch ...

# 3. Smoke controller dry-run (needs Ray; no nested GPU)
sbatch sbatch/biapy/raytune_controller_sbatch.sh \
  --exp biapy_fdb_skel_smoke --num-samples 1 --dry-run --lr 1e-3
```

Re-run the install script later to upgrade packages.

### Controller imports needed

- `ray`, `ray.tune`, `ray.tune.search.optuna`
- `yaml`, `pandas` (for `biapy_train_track`)
- Local modules under `biapy_work_folder/raytune/`
- **Not** required on controller path beyond what BiaPy_env already has for Stage 2 log parse

### Verify (inside overlay after install)

```bash
python -c 'import ray, optuna; from ray.tune.search.optuna import OptunaSearch; print(ray.__version__, optuna.__version__)'
```

GPU path unchanged: existing train sbatch still runs BiaPy under `BiaPy_env` with `:ro`.

**Done when:** controller imports Ray/Optuna; a nested train job still starts BiaPy as before.

---

## Stage 6 — Search space + Tune config

### Goal

Real Bayesian sweeps via Optuna, with persisted ranked results. Resume / winner
retrain automation is **Stage 7** (no `--resume` here yet).

### Module

[`biapy_work_folder/raytune/search_space.py`](../biapy_work_folder/raytune/search_space.py)

- `constraint_for_gpu(gpu)` → SLURM constraint (`h200` / `l40s`)
- `build_search_space(gpu=..., include_skel_weight=...)` → Tune samplers:
  `lr`, `batch_size`, `scheduler`, and **`skel_weight` only when** the base YAML
  has `LOSS.SKELETON_RECALL.ENABLE: True` (skel). Dice bases omit `skel_weight`.
  `run_tune.py --search optuna` sets this from the base config automatically.
- `apply_hparams` also **skips writing** `skel_weight` when skeleton-recall is
  not enabled (no orphan `LOSS.SKELETON_RECALL` on dice trial YAMLs).

### `run_tune.py` modes

| `--search` | Behavior |
|------------|----------|
| `fixed` (default) | One-point space from CLI hparams / smoke defaults |
| `optuna` | `build_search_space` + `OptunaSearch` |

| Flag | Notes |
|------|--------|
| `--gpu {h200,l40,l40s}` | Search-space branch; sets `--constraint` unless you pass `--constraint` |
| `--num-samples` | Default **1** (`fixed`) or **8** (`optuna`) if omitted |
| `--max-concurrent` | Default **1** (prefer 1–2 for Bayesian) |
| `--exp` | Ray experiment name under `metrics/biapy/raytune/<exp>/` |

### Artifacts (after `fit`)

Under `metrics/biapy/raytune/<exp>/`:

- `results_table.csv` — trials, configs, metric, slurm ids / paths when present
- `best_config.yaml` — winning hparams + metadata (`run_id`, `config_relpath`, …)

### Examples

```bash
# Fixed dry-run (Stage 3 smoke)
python biapy_work_folder/raytune/run_tune.py \
  --exp biapy_fdb_skel_smoke --num-samples 1 --dry-run \
  --lr 1e-3 --batch-size 30 --skel-weight 0.1

# Optuna dry-run (samples configs; Stage 2 dry_run — no GPU)
python biapy_work_folder/raytune/run_tune.py --search optuna --gpu h200 \
  --exp biapy_fdb_skel_optuna_dry --num-samples 4 --dry-run

# Real nested GPU sweep via controller
sbatch sbatch/biapy/raytune_controller_sbatch.sh \
  --search optuna --gpu h200 --exp biapy_fdb_skel_optuna \
  --num-samples 8 --max-concurrent 1 \
  --search-epochs 40 --search-patience 10
```

**Done when:** a 4–8 trial Optuna sweep finishes and produces a ranked table + best config
(after Stage 5 Ray/Optuna install).

---

## Stage 7 — Hardening

### Goal

Failed / restarted trials don’t waste GPU work; sweeps can resume; best hparams
export to a full-epoch winner YAML for the normal train/test chain.
`biapy_work_folder/configs/trials/` is already gitignored.

### Skip-finished SLURM reuse

`run_train_trial(..., skip_finished=True)` (default) looks for existing
`sbatch/biapy/{job}-r{run_id}-train-*.out`. If one parses to a usable metric, it
returns `state=REUSED` and **does not** resubmit. OOM / failure messages include
an `OOM:` prefix and `.err` tail when relevant.

### Trial YAML overwrite

`write_trial_config(..., overwrite=False)` keeps an existing
`configs/trials/<id>.yaml` (resume-friendly). Winner export uses `overwrite=True`.

### Resume

```bash
# New experiment — fails if metrics/biapy/raytune/<exp>/ already non-empty
python biapy_work_folder/raytune/run_tune.py --search optuna --exp my_exp ...

# Continue unfinished / errored trials
python biapy_work_folder/raytune/run_tune.py --resume --exp my_exp ...
# (or via controller with the same flags)
```

### Queue hygiene

- `run_tune.py --max-user-trains 4` (default): refuse to start if that many
  user `*-train` jobs are already in `squeue` (skipped for `--dry-run`; `-1` disables).
- Controller: `MAX_USER_TRAINS` env (default 4); same skip on `--dry-run`.

### Winner export

After `fit`, when a best config exists:

- `biapy_work_folder/configs/trials/<exp>_winner.yaml` — **full** base epochs
  (no search budget cap)
- `metrics/biapy/raytune/<exp>/winner_pointer.yaml` — path + chain command
- stdout JSON includes `chain_command`

Retrain manually:

```bash
./sbatch/biapy/biapy-py_sbatch_chain.sh -r <exp>_winner <stem> train test
```

**Done when:** reuse skips GPU for finished `run_id`s; `--resume` works; winner
YAML + chain command are emitted; queue guard blocks overloaded starts.

---

## Suggested build order

| Order | Stages | Deliverable |
|------:|--------|-------------|
| 1 | 0–1 | Trial YAML writer + manual sbatch |
| 2 | 2 | Submit / wait / parse smoke |
| 3 | 3 | Single-trial Ray trainable |
| 4 | 5 | Ray + Optuna on controller env |
| 5 | 4 | Nested controller sbatch |
| 6 | 6–7 | Real Bayesian sweep + resume/docs |

---

## Related paths

- Runner: `biapy_work_folder/run_biapy-py.py`
- Train sbatch: `sbatch/biapy/biapy-py_sbatch.sh`
- Chain (resource defaults): `sbatch/biapy/biapy-py_sbatch_chain.sh`
- Metric parse: `biapy_work_folder/biapy_train_track.py`
- Example skel config: `biapy_work_folder/configs/biapy-aug-zarr-seunet-FDb-skel.yaml`
- Container / Ray tmp notes: `CLAUDE_SETUP_GUIDE.md`, workspace `CLAUDE.md`
