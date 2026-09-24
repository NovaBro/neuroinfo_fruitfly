# Stage 2 detail — binary GT masks for vesselFM

Parent plan: [`vesselfm_fisbe.md`](vesselfm_fisbe.md).  
**Status:** DONE (job `17846666`, 2026-09-15). Counts 18/5/7; parity OK; smoke shape check OK.

## Goal

Produce paired binary NIfTI masks for FISBe completely splits so that:

1. Inference can score Dice/clDice (`mask_path` must match image basenames).
2. Stage 4 patch extraction has FG labels for train/val (and optionally test).

## Decisions locked in this step

| Choice | Value | Why |
|--------|--------|-----|
| GT array | `volumes/gt_fg_rm_5` | Binary FG after opening/rm; not instance IDs |
| Binarize | `(volume > 0).astype(uint8)` via `--binarize` | vesselFM expects binary masks |
| Layout | `fisbe/vesselfm-fisbe/{split}_masks/` | Separate dir from images (inference requirement) |
| Splits this job | `train`, `val` (and re-check `test`) | test_masks already exists |
| Spacing | `1 1 1` | Match Stage 1 images |
| Execution | CPU sbatch + `env/vesselfm.ext3:ro` | Same pattern as Stage 1 convert |

## Deliverables

| Path | Expected count |
|------|----------------|
| `fisbe/vesselfm-fisbe/train_masks/*.nii.gz` | 18 |
| `fisbe/vesselfm-fisbe/val_masks/*.nii.gz` | 5 |
| `fisbe/vesselfm-fisbe/test_masks/*.nii.gz` | 7 (already present; job re-checks parity) |
| `sbatch/vesselfm/vesselfm_convert_masks_sbatch.sh` | New |

## Implementation steps

1. Add `sbatch/vesselfm/vesselfm_convert_masks_sbatch.sh` modeled on
   `vesselfm_convert_images_sbatch.sh`.
2. Inside container: for each of `train`/`val`/`test`, run
   `convert_fisbe_zarr_to_nii.py` with `--array-key volumes/gt_fg_rm_5 --binarize`.
3. Skip existing files (no `--overwrite`) so re-runs are safe.
4. After convert: basename parity vs image dirs; fail job on mismatch.
5. Optional: one SimpleITK shape compare (image vs mask) for a single train stem
   as a smoke check in the job log.

## Commands

```bash
# from repo root
sbatch sbatch/vesselfm/vesselfm_convert_masks_sbatch.sh
```

## Exit criteria

- Counts 18 / 5 / 7 for train/val/test masks.
- Empty `comm -3` between image and mask basenames per split.
- Parent roadmap Stage 2 marked `[DONE]`.

## Out of scope

- Patch dataset (Stage 4).
- GPU inference / training.
- Overwriting existing `test_masks` unless regenerating with `--overwrite`.
