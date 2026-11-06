# setup01 configs

TOML configs for `run_ppp.py --setup setup01`. Configs are merged left to right
(`run_ppp.merge_dicts`, later wins). A run stacks them as:

```
base  →  preset  →  [data] [train] [exp]  →  [vi] [eval]
```

## Referring to a config

The submitters and `ppp_job.sh` take **bare file names** (`--base original_default.toml`,
`--config vi_th_0_3.toml`). `sbatch/ppp/ppp_config_path.py` resolves each name to the one
matching file under `configs/` (`archive/` is skipped). If two folders ever hold the same name,
pass the subfolder path instead (`--config vi/vi_th_0_3.toml`). Keep file names unique across folders.

When calling `run_ppp.py` by hand, give the full path relative to `experiments/`:
`--config flylight/setups/setup01/configs/base/original_default.toml`.

Check how a stack resolves: `python3 sbatch/ppp/ppp_config_path.py NAME...`
(and `needs_decode.py` / `ppp_datasets.py --split test` for the merged result).

## Folders

### `base/`: complete defaults (always first; `--base`)
| file | role |
|---|---|
| `original_default.toml` | upstream PatchPerPix default (noenc). Base for experiment_v1 and all `our-noenc-*` presets |
| `original_default_train_code.toml` | upstream PPP+dec default (`train_code = true`). Base for `*decoder*` presets |
| `default_l40s.toml` | FISBe/L40S noenc base, `overwrite = true`. Fallback `DEFAULT_CONFIG` when no `--base` is given |
| `default_train_code_l40s.toml` | FISBe/L40S PPP+dec base |

### `presets/`: model + data + training (one per `PRESET_TABLE` row in `sbatch/ppp/ppp_submit_common.sh`)
| file | model | train data |
|---|---|---|
| `our-noenc-c1k_v1.toml` | noenc fmaps20 bs18 | synthetic `1k_v1` (current, experiment_v1) |
| `our-noenc-c1k_v1_u8.toml` | noenc fmaps24 bs18 | synthetic `1k_v1_u8` |
| `our-decoder-c1k_v1.toml` / `_u8` | PPP+dec fmaps20 bs30 | `1k_v1` / `1k_v1_u8` |
| `our-noenc-cs.toml` | noenc fmaps24 | `synth_vh` |
| `base1-noenc-c.toml` / `base1-decoder-c.toml` | noenc fmaps24 / PPP+dec fmaps20 | completely (baselines) |
| `basic-ppp-noenc.toml`, `basic-ppp-noenc-cp.toml`, `basic-ppp-decoder.toml` | older 400k baselines | completely (`-cp`: + partly) |

### `data/`: dataset-selection overlays
`datasets_completely_partly.toml` (train/val/test on completely + partly),
`datasets_test_himanshu.toml` (MCFO test-only inference),
`datasets_test_1k_v1[_u8]_keep_completely.toml` (one-shot evaluate on completely + 1k_v1, only for `evaluate`).

### `train/`: run-length / speed overlays (stack last)
`smoke.toml` (150 it) + `smoke_noval.toml`; `profile_noenc_{l40s,completely_l40s,h200}.toml` (300-it
profiling); `channels_last.toml`; `raw_fp16.toml` (fp16 raw handoff).

### `exp/`: experiment-specific overlays
`exp_v1_h200.toml` (experiment_v1 H200 settings: 50k it, ckpt 2k, 14 workers, fp16 raw),
`exp_v1_n500.toml` (EXP 2: first 500 train volumes). `sbatch/ppp/exp_v1_finetune.sh` writes its generated
`exp3_ft_<src>_<ckpt>.toml` fine-tune overlays here (note: `--dry-run` writes the overlay too).

### `vi/`: vote_instances / label overlays
`vi_th_0_{3..7}.toml` (patch/fc threshold; `0_3` is the `ppp_submit_test.sh` default, `vi-sweep` runs
all five); `vi_split_gpu.toml` → `vi_split_cpu.toml` (two-phase blockwise label, added automatically by
`ppp_submit_test.sh`); `label_resume.toml` (`overwrite = false`, added by `LABEL_RESUME=1`).

### `eval/`: evaluation overlays
`eval_rm200.toml` (default `--eval`), `eval_rm0.toml` (keep every component),
`eval_pred.toml` (prediction-only metrics, skips VI).

### `archive/`: not resolvable by bare name, kept for reference
`default.toml`, `default_train_code.toml` (upstream files with stale local edits, laptop paths; use
`base/original_default*.toml`), `h200-noenc.toml`, `h200-decoder.toml` (superseded by
`exp/exp_v1_h200.toml`), `our-noenc-staged.toml` (superseded by the EXP 3 fine-tune flow), `vi_cpu.toml`
(superseded by `vi_split_*`), `vi_merge_*.toml` (Layer B VI ablations, Sep 2026),
`exp3_ft_noenc-c1k_v1_50000.toml` (EXP 3 run cancelled 2026-11-06). Use one with an explicit
path, e.g. `--config archive/vi_cpu.toml`.
