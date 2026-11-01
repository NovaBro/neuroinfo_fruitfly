# Stage 4 detail — finetune patch dataset from FISBe completely

Parent plan: [`vesselfm_fisbe.md`](vesselfm_fisbe.md).  
**Status:** SMOKE DONE (job `17846712`: train 72 / val 10 / test 14; mean train FG≈0.020).
Full patch rebuild (`PATCHES_TRAIN=32`) still pending before serious training.

## Goal

Build `fisbe/vesselfm-fisbe-finetune/{train,val,test}/<id>/{img,mask}.npy` so
`UnionDataset(..., finetune=True)` can train vesselFM on patches from
**FISBe completely** splits only.

## Locked decisions

| Choice | Value | Why |
|--------|--------|-----|
| Source | Zarr `volumes/raw` (channel MIP) + `volumes/gt_fg_rm_5` | Avoid depending on NIfTI round-trip; same arrays as Stages 1–2 |
| Patch size | `128³` | Matches `input_size` / DynUNet |
| Train sampling | FG-biased random crops; reject if FG frac &lt; `min_fg_frac` | Avoid empty patches |
| Val/test | Same patch extraction (not full volumes) for first smoke | Faster iteration; Stage 7 still uses flat NIfTI inference |
| File format | `img.npy` float32, `mask.npy` bool | `default_finetune.yaml` `file_format: npy` |
| Sample IDs | Zero-padded global counters per split (`000000`, …) | Stable sort for `num_shots` |
| Smoke defaults | `--patches-per-volume 4` (train), `2` (val/test) | Small CPU job; full run later |
| Full defaults | `--patches-per-volume 32` train / `8` val/test | Enough diversity without huge disk |

## Deliverables

| Path | Role |
|------|------|
| `vesselFM/prepare_fisbe_finetune_patches.py` | Builder script |
| `sbatch/vesselfm/vesselfm_prepare_patches_sbatch.sh` | CPU sbatch (smoke vs full via env) |
| `fisbe/vesselfm-fisbe-finetune/` | Output root |
| Manifest CSV | `fisbe/vesselfm-fisbe-finetune/manifest.csv` |

## Algorithm

For each split ∈ {train, val, test}:

1. List `fisbe/completely/{split}/*.zarr` (sorted).
2. Load raw → channel MIP → float32 ZYX; load `gt_fg_rm_5` → bool.
3. Collect FG voxel indices (`mask > 0`). If none, skip volume with a warning.
4. For `i in range(patches_per_volume)`:
   - Sample a FG voxel as crop center (or random if FG too sparse near borders).
   - Extract `128³` window with reflect pad if near border.
   - If `mask.mean() < min_fg_frac`, retry (cap retries).
   - Write `train/000012/img.npy` + `mask.npy`; append manifest row.

## Smoke vs full

```bash
# smoke (this tick)
PATCHES_TRAIN=4 PATCHES_VAL=2 PATCHES_TEST=2 MIN_FG=0.001 \
  sbatch sbatch/vesselfm/vesselfm_prepare_patches_sbatch.sh

# full (later, after smoke OK)
PATCHES_TRAIN=32 PATCHES_VAL=8 PATCHES_TEST=8 \
  sbatch sbatch/vesselfm/vesselfm_prepare_patches_sbatch.sh
```

Expected smoke counts ≈ 18×4 + 5×2 + 7×2 = 72 + 10 + 14 = **96** sample dirs.

## Exit criteria

- Script + sbatch exist.
- Smoke job completes; train FG mean ≫ 0.
- `UnionDataset`-compatible layout verified by loading one sample.
- Parent Stage 4 marked `[DONE]` after **full** patch build (smoke alone →
  `[PARTIAL]`).
