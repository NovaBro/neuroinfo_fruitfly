# Stage 5 detail — Hydra FISBe data config + Lightning-wrapped ckpt

Parent plan: [`vesselfm_fisbe.md`](vesselfm_fisbe.md).  
**Status:** DONE (wrapped ckpt + `eval_fisbe.yaml`; DynUNet strict load OK).

## Goal

1. Add `data=eval_fisbe` pointing at `fisbe/vesselfm-fisbe-finetune`.
2. Wrap HF `vesselFM_all.pt` into a Lightning-like checkpoint for `finetune.py`.

## Locked decisions

| Choice | Value |
|--------|--------|
| Data YAML | `vesselFM/vesselfm/seg/configs/data/eval_fisbe.yaml` |
| Dataset key | `FISBe` (becomes run/metric name) |
| Path | absolute repo path to `fisbe/vesselfm-fisbe-finetune` |
| Wrap strategy | Option A (prefix `model.` + `state_dict` dict) |
| Wrapped path | `metrics/vesselfm/checkpoints/vesselFM_all_lightning.ckpt` |
| Source HF file | `.cache/huggingface/hub/.../vesselFM_all.pt` (already cached) |

## Deliverables

- `eval_fisbe.yaml`
- `metrics/vesselfm/checkpoints/vesselFM_all_lightning.ckpt`
- Optional helper: `vesselFM/wrap_vesselfm_ckpt.py`

## Exit criteria

- Config file present.
- Wrapped ckpt loads; DynUNet `load_state_dict` succeeds (CPU smoke).
