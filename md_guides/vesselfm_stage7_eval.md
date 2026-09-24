# Stage 7 detail — evaluate finetuned vesselFM vs zero-shot

Parent plan: [`vesselfm_fisbe.md`](vesselfm_fisbe.md).  
**Status:** planned (blocked on Stage 6 smoke success).

## Goal

Export the best Lightning finetune checkpoint to a raw `state_dict`, run
inference on the same FISBe completely/test NIfTIs as Stage 3, and compare
mean Dice/clDice.

## Locked decisions

| Choice | Value |
|--------|--------|
| Test images | `fisbe/vesselfm-fisbe/test` |
| Test masks | `fisbe/vesselfm-fisbe/test_masks` |
| Output | `metrics/vesselfm/fisbe-completely-test-ft` |
| Baseline | Stage 3 mean Dice **0.0618** / clDice **0.0454** |
| Threshold / percentiles | Match Stage 3 (`merging.threshold=0.5`, 1–99) |
| Export path | `metrics/vesselfm/checkpoints/fisbe_finetuned.pt` |

## Steps (after Stage 6 produces a ckpt)

1. Locate best ckpt under `metrics/vesselfm/finetune_ckpts/…`.
2. Export via small script or one-liner (strip `model.` prefix).
3. Copy/adapt `vesselfm_inference_sbatch.sh` with `ckpt_path=` + new `OUTPUT_FOLDER`.
4. Submit **one** GPU smoke-style infer job (same walltime class as Stage 3).
5. Diff mean metrics vs Stage 3 in `vesselfm_stage3_zeroshot.md` / loop progress.

## Deliverables to add when unblocked

- `vesselFM/export_finetune_ckpt.py` (optional helper)
- `sbatch/vesselfm/vesselfm_inference_ft_sbatch.sh` or env overrides on existing infer sbatch

## Exit criteria

Side-by-side test metrics + pred NIfTIs for pretrained vs finetuned.
