# nnU-Net v2 on FISBe — instance segmentation via region labels

**Goal:** run [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) (v2.8.1) on FISBe *completely* and get
**instance** segmentations scored with the same FISBe metrics as PatchPerPix (PPP) and BiaPy.

**Repo root for all submits:** `/scratch/wmz2007/neuroinfo_fruitfly` (scripts use `SLURM_SUBMIT_DIR`).

---

## Method in one paragraph

nnU-Net is semantic-only. Its usual instance recipe (border–core, `acvl_utils`) fails on 1–3-voxel
neurites: even a perfect prediction scores S ≈ 0.47 on easy test volumes, because it splits each neuron
into many pieces. BiaPy's distance+watershed failed here for the same reason. So we train
**stock nnU-Net on nested region labels** built from `volumes/gt_instances`:

| label | meaning | region |
|---|---|---|
| 0 | background | — |
| 1 | neuron (in ≥ 1 GT instance) | neuron = {1,2,3} |
| 2 | contact (neuron voxel whose 26-neighbourhood touches a *different* instance) | contact = {2,3} |
| 3 | overlap (in ≥ 2 GT instances) | overlap = {3} |

nnU-Net trains three sigmoid heads (`regions_class_order [1,2,3]`). `decode_instances.py` turns the
probabilities into instances in two ways:
- **`cc`** — 26-connected components of `p_neuron > t_fg` (plain baseline).
- **`contact`** — remove `p_contact > t_c` from the foreground, take 26-connected cores, drop cores under
  `min_core` voxels, then flood each multi-core foreground component from its cores (geodesic nearest core).

Thresholds are tuned on the 5 val volumes (rm1500, MEAN S), then frozen and applied once to test.

---

## Artifact map

| What | Path |
|---|---|
| nnU-Net clone (v2.8.1, editable install, gitignored) | `nnUNet/` |
| Overlay / conda env | `env/nnunet.ext3` → `/ext3/miniforge3/envs/nnunet` (py 3.11, torch 2.8.0+cu128, torchvision 0.23.0, nnunetv2 2.8.1, zarr<3) |
| Converter (zarr → nnU-Net raw) | `nnunet_work_folder/fisbe_to_nnunet.py` |
| Custom trainer (`nnUNet_extTrainer`) | `nnunet_work_folder/trainers/nnUNetTrainer_ckpt10.py` (stock + checkpoint every 10 epochs) |
| Instance decoder | `nnunet_work_folder/decode_instances.py` |
| FISBe scorer (PPP-identical evalinstseg call) | `nnunet_work_folder/score_fisbe.py` |
| Val tuning grid / ranking | `nnunet_work_folder/val_tune_grid.txt`, `nnunet_work_folder/collect_scores.py` |
| Shared job settings | `sbatch/nnunet/nnunet_common.sh` |
| Jobs | `sbatch/nnunet/nnunet_{env_install,gpu_check,prep,benchmark,train,predict,decode_eval}_sbatch.sh`, `nnunet_submit_train.sh` |
| Logs | `sbatch/nnunet/logs/`, GPU util `gpu_log/nnunet/` |
| nnU-Net raw | `fisbe/nnUNet_raw/Dataset501_FISBeCompletely/` (imagesTr 23, labelsTr 23, imagesTs 7, `splits_final.json`) |
| nnU-Net preprocessed | `fisbe/nnUNet_preprocessed/Dataset501_FISBeCompletely/` |
| nnU-Net results | `metrics/nnunet/nnUNet_results/Dataset501_FISBeCompletely/nnUNetTrainer_ckpt10__nnUNetResEncUNetLPlans__3d_fullres/fold_0/` |
| Val tuning / test outputs | `metrics/nnunet/fisbe501_resencL_f0/{val_tune,test}/` |

## Data facts used

- FISBe spacing is **0.44 µm isotropic** (FISBe paper §2.1); the zarrs carry no spacing. Every image
  folder gets `<case>.json` = `{"spacing":[0.44,0.44,0.44]}`. Without it Tiff3DIO assumes (1,1,1) and
  nnU-Net would resample 2.27× per axis.
- All 3 MCFO channels kept (`_0000/_0001/_0002`, uint16), per-image z-score normalization.
- Fold 0 = FISBe split: 18 train / 5 val (`splits_final.json`, copied to preprocessed by the planner).
  7 test volumes are in `imagesTs/` and only used for prediction.
- `partly` (71 samples) is **not** used yet: zero there means "background or unlabelled neuron".
- Overlap voxels exist in 12 / 30 completely samples (≤ 1.7 % of foreground).

---

## Stages and commands

```text
S1 env install + GPU check   sbatch sbatch/nnunet/nnunet_env_install_sbatch.sh
                             sbatch --dependency=afterok:<id> sbatch/nnunet/nnunet_gpu_check_sbatch.sh
S2+S3 convert + plan/prep    sbatch sbatch/nnunet/nnunet_prep_sbatch.sh            (CPU, partition cs)
S4 benchmark (5 epochs)      sbatch sbatch/nnunet/nnunet_benchmark_sbatch.sh       (L40S, 32 CPUs)
S5 train fold 0, 1000 ep     bash sbatch/nnunet/nnunet_submit_train.sh 3            (chained 47 h jobs, --c)
                             -> also exports validation_final/ and validation_best/ probabilities
S6 val decode tuning         GRID=nnunet_work_folder/val_tune_grid.txt sbatch --export=ALL --array=1-14 \
                               sbatch/nnunet/nnunet_decode_eval_sbatch.sh
                             python3 nnunet_work_folder/collect_scores.py metrics/nnunet/fisbe501_resencL_f0/val_tune
S7 test                      CKPT=<final|best> sbatch --export=ALL sbatch/nnunet/nnunet_predict_sbatch.sh
                             SPLIT=test NAME=<best val setting> CKPT=.. MODE=.. T_FG=.. T_C=.. MIN_CORE=.. \
                               RM_LIST="1500 500 800" sbatch --export=ALL sbatch/nnunet/nnunet_decode_eval_sbatch.sh
```

Runtime activation (any job):

```bash
singularity exec --nv --overlay env/nnunet.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c 'source /ext3/env.sh; conda activate /ext3/miniforge3/envs/nnunet; nnUNetv2_train ...'
```

Set `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` and `nnUNet_extTrainer` as in
`sbatch/nnunet/nnunet_common.sh`. Host environment variables pass into the container.

## Numbers to beat (completely test, 7 volumes, 34 neurons)

| Baseline | rm | S mean / pooled | avF1 mean | C mean |
|---|---|---|---|---|
| PPP `opexp5-e1.3-rm1500` @ 23000 | 1500 | **0.3046 / 0.3028** | 0.1913 | 0.4180 |
| PPP same run | 500 | 0.2848 / 0.2691 | | |
| BiaPy v2 (approximate; outputs deleted) | 500 | 0.3193 / 0.2471 | 0.0641 | |
| FISBe leaderboard PPP | — | 0.34 | 0.29 | 0.40 |

## Gotchas

- nnU-Net ≥ 2.6 needs Python ≥ 3.10, so it cannot use the py3.9 `ppp` overlay.
- Pin torch **and** torchvision together. timm (pulled in via dynamic-network-architectures) otherwise
  upgrades torch, and resolving any `-tr` variant then crashes on `torchvision::nms`.
  torch 2.9.* is excluded by nnU-Net (3D-conv AMP regression).
- `nnUNetv2_predict` without `-f` tries folds 0–4 and fails: always pass `-f 0`.
- `nnUNetv2_train --val --val_best` overwrites `validation/`. The train job copies the folders out to
  `validation_final/` and `validation_best/`.
- The CLI does not set OMP/MKL threads; `nnunet_common.sh` exports them, and `nnUNet_n_proc_DA` explicitly.
  The default is capped by the node's 128 cores, not the SLURM allocation.
- `nnUNetTrainerBenchmark_5epochs_noDataLoading` does not support region labels (1-channel dummy target).
- evalinstseg silently centre-crops a prediction whose ZYX all differ from GT. The decoder and scorer
  assert exact shapes.
- Use 26-connectivity everywhere. The acvl helpers' 6-connectivity shatters thin neurites.
- cpu_short QOS caps a user at 120 GB in total; CPU jobs here use partition `cs` (QOS cpu48).

## Status

| Stage | Status |
|---|---|
| S1 env install + GPU check | DONE 2026-11-20 (23853809, 23853810): torch 2.8.0+cu128 on L40S; ResEnc-L-like 128³ bs2 step 0.14 s compiled, 6 GB |
| S2–S3 convert + plan/prep | DONE (23854296, 10 min): 23 train+val, 7 test; plan patch 128×224×256, bs 2, 0.44 iso, z-score ×3; preprocessed 11 GB; overlap counts match GT scan |
| Oracle decode check (val, GT-derived probs) | submitted: 23869029 → 23869030 (cc), 23869031 (contact) → `metrics/nnunet/oracle/val_tune/` |
| S4 benchmark | submitted 23869024 |
| S5 train fold 0 | chained 23869025 (afterok bench) → 23869027 → 23869028 |
| S6–S7 | pending |

## Next upgrades (after the first result)

1. Overlap-aware output: overlap voxels go to both neighbouring instances, written as layered channels.
2. Colour-aware merge of fragments across contact zones (MCFO chromaticity).
3. Skeleton Recall loss via `nnUNet_extTrainer`: upstream `skeleton-recall` plugin (needs nnunetv2 ≥ 2.8,
   plain classes, not regions). Do not install the stale local `Skeleton-Recall/` clone; it vendors nnunetv2 2.5.
4. `partly` data with an ignore label.
