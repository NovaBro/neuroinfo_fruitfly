# Stage 3 detail — zero-shot vesselFM baseline on FISBe test

Parent plan: [`vesselfm_fisbe.md`](vesselfm_fisbe.md).  
**Status:** DONE (prior job `17353283`, 2026-09-10/11). No re-submit needed.

## Goal

Score pretrained Hugging Face `bwittmann/vesselFM` (`vesselFM_all.pt`) on
`fisbe/vesselfm-fisbe/test` with GT masks, before fine-tuning.

## Locked choices

| Choice | Value |
|--------|--------|
| Images | `fisbe/vesselfm-fisbe/test` (7 NIfTIs) |
| Masks | `fisbe/vesselfm-fisbe/test_masks` |
| Output | `metrics/vesselfm/fisbe-completely-test` |
| Sbatch | `sbatch/vesselfm/vesselfm_inference_sbatch.sh` |
| Weights | HF fallback (no local `ckpt_path`) |
| Threshold | inference default `0.5` (keep for Stage 7 compare) |

## Archived metrics (job `17353283`)

Logged as `Mean metrics: dice 6.18, cldice 4.54` (percent-scale in the logger).

| Volume | Dice | clDice |
|--------|------|--------|
| JRC_SS04989-20160318_24_A2 | 0.0207 | 0.0192 |
| R14A02-20180905_65_A6 | 0.1533 | 0.0875 |
| R54A09-20181019_64_H1 | 0.0377 | 0.0252 |
| VT011145-20171222_63_I1 | 0.0919 | 0.0727 |
| VT027175-20171031_62_H3 | 0.0225 | 0.0181 |
| VT027175-20171031_62_H4 | 0.0672 | 0.0635 |
| VT050157-20171110_61_C1 | 0.0396 | 0.0313 |
| **Mean** | **0.0618** | **0.0454** |

Artifacts: 7 `*_pred.nii.gz` under `metrics/vesselfm/fisbe-completely-test/`.

## Interpretation for later stages

Zero-shot Dice is very low (~0.06). Fine-tuning on FISBe completely/train is
justified. Do **not** retune Stage 3 knobs (`tta.invert`, percentiles) until after
a first fine-tune unless debugging inference crashes — keep Stage 3/7 settings
aligned for a fair comparison.

## Re-run (only if needed)

```bash
sbatch sbatch/vesselfm/vesselfm_inference_sbatch.sh
```

## Exit criteria (met)

- Mean Dice/clDice recorded for all 7 test volumes.
- Pred NIfTIs present.
- Parent roadmap Stage 3 marked `[DONE]`.
