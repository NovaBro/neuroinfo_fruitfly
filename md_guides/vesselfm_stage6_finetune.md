# Stage 6 detail — fine-tune smoke on FISBe patches

Parent plan: [`vesselfm_fisbe.md`](vesselfm_fisbe.md).  
**Status:** SMOKE SUBMITTED (job `17847501`, `NUM_SHOTS=8`, `MAX_STEPS=50`).
Await completion on next loop tick; full train still TODO.

## Goal

Run `vesselfm/seg/finetune.py` with `data=eval_fisbe` starting from the wrapped
pretrained ckpt. This tick: **smoke** only (short `max_steps`, small
`num_shots`) to prove the training path works on Greene.

## Locked decisions (smoke)

| Choice | Value |
|--------|--------|
| Data | `data=eval_fisbe` → smoke patches under `fisbe/vesselfm-fisbe-finetune` |
| `num_shots` | `8` (first 8 sorted train patches) |
| `max_steps` | `50` |
| `val_check_interval` | `25` |
| `offline` | `True` |
| `path_to_chkpt` | `metrics/vesselfm/checkpoints/vesselFM_all_lightning.ckpt` |
| `chkpt_folder` | `metrics/vesselfm/finetune_ckpts` |
| `dataloader.num_workers` | `4` |
| GPU | 1× h100\|h200, short walltime (2h) |

## Full train (later — not this tick)

After Stage 4 **full** patch build (`PATCHES_TRAIN=32`):

- `num_shots=<train count>`
- `max_steps` several thousand
- Longer walltime (12–24h)

## Deliverables

- `sbatch/vesselfm/vesselfm_finetune_sbatch.sh`
- Smoke job log + at least one ckpt under `metrics/vesselfm/finetune_ckpts/`

## Exit criteria (smoke)

- Job completes (or cleanly validates+fits briefly) without import/path errors.
- Checkpoint directory created.
- Parent Stage 6 marked `[SMOKE DONE]` until full train runs.
