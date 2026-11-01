# PatchPerPix SLURM scripts

Submit from the repo root. Train and inference are separate launchers; everything else is grouped by role.

```bash
./sbatch/ppp/ppp_submit_train.sh [--dry-run] <preset> [flags…]
./sbatch/ppp/ppp_submit_test.sh [--dry-run] <mode> <preset> [flags…]
```

Test modes: `array`, `run`, `evaluate`, `ckpt-sweep`, `vi-sweep`. Both submitters set resources and call `ppp_job.sh`. Shared presets live in `ppp_submit_common.sh`. Python helpers the job calls (`ppp_datasets.py`, `needs_decode.py`, `check_prediction_complete.py`, `merge_ppp_summary_csvs.py`, `strip_pred_affs.py`) stay in this directory. `ppp_submit.sh` only prints this split and exits.

| Directory | What it is |
|-----------|------------|
| [`prep/`](prep/) | Shared dataset jobs (PPP arrays, FG points, stage tars), plus [`himanshu/`](prep/himanshu/), [`synth/`](prep/synth/), and [`fisbe/`](prep/fisbe/) |
| [`env/`](env/) | One-writer installs into `env/ppp.ext3` (PyAV, tensorboard) |
| [`tools/`](tools/) | Dataloader bench, blockwise-decode checks, inference cleanup, old experiment diagnostics |
| [`logs/`](logs/) | Job stdout/stderr (`%x-%j` and array logs). Gitignored. |

Prep example:

```bash
DATASETS=completely,partly,synth_vh sbatch sbatch/ppp/prep/prepare_fisbe_for_ppp_sbatch.sh
```

Ops write-up: [`md_guides/ppp/ppp_overview.md`](../../md_guides/ppp/ppp_overview.md).
Prep scripts: [`md_guides/ppp/ppp_prep.md`](../../md_guides/ppp/ppp_prep.md).
