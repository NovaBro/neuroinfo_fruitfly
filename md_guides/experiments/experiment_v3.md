# experiment_v3: Architecture Experiments (6-stage U-Net, S6 D2 W64)

Plan written 2026-11-17; decisions approved the same day (see Decisions). Not started. Status page
`md_guides/experiments/expv3_status.md` gets created at launch.

## Goal

Measure what the U-Net configuration from the paper below does to PatchPerPix (noenc). Pretrain on synthetic
1k_v1.1, finetune on real completely, then test on completely. Compare against the current architecture under
the same protocol.

## Reference paper

Huang et al., *Revisiting model scaling with a U-net benchmark for 3D medical image segmentation*, Sci. Rep. 15:29795
(2025), doi [10.1038/s41598-025-15617-1](https://www.nature.com/articles/s41598-025-15617-1). Local copy:
`papers/s41598-025-15617-1.pdf`. Code: [uni-medical/UNet-Benchmark](https://github.com/uni-medical/UNet-Benchmark)
(an nnU-Net v2.4.1 plans script).

- **Setup:** 18 nnU-Net variants on 42 public 3D medical datasets (CT, MRI, PET, US; 1 to 104 classes). The grid
  is stages S ∈ {4, 5, 6}, convs per stage D ∈ {2, 3} and first-stage width W ∈ {16, 32, 64}. Models range from
  1.4 M to 137 M params.
- **Definitions** (Methods + Table 1, matching the repo script):
  - S = number of resolution stages, i.e. S − 1 downsamplings.
  - D = conv layers per stage, in both encoder and decoder.
  - W = output channels of the first conv. Channels **double at each downsampling stage, capped at 512**, with a
    symmetric decoder and skip connections.
  - Table 1, S6D2W64: `(64, 128, 256, 512, 512, 512)`, **97.39 M params, 3.43 TFLOPs at 128³**.
- **Their training regime** (one fixed nnU-Net recipe for all variants):
  - blocks are conv + instance norm + leaky ReLU;
  - batch size 2;
  - 1000 epochs × 250 its = 250k iterations;
  - SGD with Nesterov momentum 0.99, weight decay 3e-5, lr 0.01, poly decay (1 − epoch/1000)^0.9;
  - Dice + CE loss;
  - nnU-Net augmentations, with 33% foreground oversampling;
  - nnU-Net auto-planned patch sizes;
  - inference by sliding window with mirror TTA.
- **Overall results:**
  - Mean rank by axis: D2 → D3 improves consistently (1.70 → 1.30). S4 → S5 helps a lot (2.58 → 1.73), S6 only
    marginally (1.73 → 1.68). W16 → W32 helps (2.53 → 1.90), W64 less so (1.90 → 1.57).
  - Median DSC across all variants is 0.82–0.87, so the gaps are small.
  - Larger models do not overfit more; the train/val gap is similar across sizes.
  - S6D2W64 and S6D3W64 consistently won on complex multi-organ abdominal CT (BTCV, FLARE22, WORD, AMOS), at
    every training-set size.
- **Cost (their Fig. 6):** doubling W roughly doubles inference time, +1 D adds 30–40%, +1 S adds 10–20%.
  Training time barely changed, because nnU-Net training was CPU / augmentation bound.
- **What we port:** only S, D and W. The other nnU-Net choices stay as PatchPerPix has them (see "Not ported"
  below).

### Paper guideline (Fig. 9) applied to FISBe

| Axis | Paper rule | FISBe | Guideline says | We use |
|---|---|---|---|---|
| S | spacing < 0.8 mm → S6 (S5 if resources are tight); larger S enlarges the receptive field | 0.44 µm isotropic; neurons span far more than any crop | S6 | **S6** |
| D | mean sphericity > 0.6 → D3; intricate shapes → D2 | thin, branching neurons, very low sphericity | D2 | **D2** |
| W | 1 class → W16; 2–10 classes → W32; > 10 classes → W64 (W32 if tight) | binary fg plus up to 2 overlapping instances, but a 343-channel patch-affinity output that has to encode local instance shape | W16–W32 by class count | **W64** |

The paper's own rule does not pick W64 for a binary-like task. Choosing W64 here assumes PatchPerPix's patch output
behaves more like a many-class task. Their mean-rank gain from W32 to W64 is also the smaller of the two width
steps. See Decision 6 for an optional W32 arm (S6D2W32: 60.08 M params, 0.88 TFLOPs at 128³, ~4× cheaper).

## Architecture change

Correction to the first draft of this plan. The draft read the config as "fmaps=2, fmap_inc_factors=1,
fmap_dec_factors=1". The paper's code doubles the width per stage, so the mapping onto PatchPerPix is:

| Paper | PatchPerPix `[model]` key | Current (expv2 / dsexp) | experiment_v3 |
|---|---|---|---|
| S = 6 stages | `downsample_factors` (S − 1 entries) | 3 × `[2,2,2]` (4 stages) | 5 × `[2,2,2]` (6 stages) |
| D = 2 convs / stage | `num_repetitions` | 2 | 2 (unchanged) |
| W = 64 | `num_fmaps` | 15 | 64 |
| ×2 per stage, cap 512 | `fmap_inc_factors` (per downsampling) | `[3, 3, 3]` | `[2, 2, 2, 1, 1]` |
| decoder mirrors encoder | `fmap_dec_factors` | `[1.2, 1.2, 1.2]` | `[2, 2, 2, 1, 1]` |
| encoder channels | | 15 / 45 / 135 / 405 | 64 / 128 / 256 / 512 / 512 / 512 |
| decoder channels (top first) | | 233 / 280 / 337 | 64 / 128 / 256 / 512 / 512 |
| padding | `train_padding`, `val_padding` | `valid` | **`same`** (decided 2026-11-17) |
| training crop | `train_input_shape_*` | 140³ valid → 52³ output | **160³ same → 160³ output** (128³ fallback) |
| parameters | | 24.1 M | 97.4 M (paper Table 1: 97.39 M) |
| theoretical receptive field | | 89 vox (39 µm) | 377 vox (166 µm) |

**Why same padding and 160³ instead of the ~178³ in the draft:**

- With valid padding, 2 convs per stage and 5 downsamplings, the smallest legal training input is 380³, which
  gives a 4³ output. A useful output needs 444³, which gives 68³. The shallowest completely volume is only 345
  voxels deep in z, so a 6-stage valid net does not fit the data.
- 178³ is not a legal size for 4, 5 or 6 stages under valid padding either.
- With same padding, the input must be divisible by 2⁵ = 32. The candidates near 178 are **160** (bottom level
  5³) and 192 (6³). Fallback is 128 (4³).
- Same padding is also what nnU-Net, and therefore the paper, uses.
- Caveat: the receptive field is 377 voxels, but a 160³ crop caps the context the net sees during training at
  160 voxels.

**Not ported from the paper's recipe** (stays as PatchPerPix has it):

- From the paper itself:
  - no normalization (they use instance norm);
  - ReLU (they use leaky ReLU);
  - Adam with 10% warmup + cosine at 4e-4 (they use SGD + Nesterov, lr 0.01, poly decay);
  - PatchPerPix loss and augmentations (they use Dice + CE and nnU-Net augmentations);
  - no mirror TTA at inference (8× predict cost).
- nnU-Net defaults the paper does not mention: strided-conv downsampling (we use max-pool) and deep supervision
  (we have none).

Each of these could become a follow-up arm. They are left out here so that only S, D and W change. Batch size 2
at 160³ does match the paper's bs 2.

Proposed overlay `PatchPerPix/experiments/flylight/setups/setup01/configs/exp/expv3/expv3_s6d2w64.toml`. It is not
created yet; batch sizes get set from the P3 smoke test.

```toml
# experiment_v3: S6 D2 W64 U-Net (Sci Rep 2025 U-Net benchmark), same padding.
[model]
num_fmaps = 64
fmap_inc_factors = [2, 2, 2, 1, 1]
fmap_dec_factors = [2, 2, 2, 1, 1]
downsample_factors = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
num_repetitions = 2
kernel_size = 3
train_padding = "same"
val_padding = "same"
train_input_shape_same = [160, 160, 160]   # multiple of 32; 128 if 160 does not fit bs >= 2
test_input_shape_same = [160, 160, 160]    # val crops + predict tile (see P2)

[training]
batch_size = 2          # from P3

[prediction]
batch_size = 4          # from P3 (current 36 is for the 156³ valid net)
```

## Cost estimate (computed, not measured; the P3 smoke test replaces these numbers)

Computed from the funlib UNet shape and FLOP logic. Memory is scaled from the current net's measured ~36 GB at bs18.

| | Current, 140³ valid | S6D2W64, 128³ same | S6D2W64, 160³ same |
|---|---|---|---|
| Output voxels per sample | 52³ (1×) | 128³ (15×) | 160³ (29×) |
| Forward TFLOP per sample | 1.38 | 3.90 (paper: 3.43) | 7.61 |
| Est. memory per sample | ~2 GB | ~10 GB | ~18 GB |
| Est. max batch on L40S (46 GB) | 18 (in use) | ~4 | ~2 |
| Loss voxels per iteration | 2.5 M (bs18) | 8.4 M (bs4) | 8.2 M (bs2) |
| Checkpoint size (weights + Adam) | ~0.3 GB | ~1.2 GB | ~1.2 GB |

- FLOPs here count 2 per multiply-add and include the 343-channel patch head, which the paper's network does
  not have. The paper's Table 1 is ~12% lower at 128³. Both counts give the same S6/S4 ratio (~1.06).
- The memory figures are probably optimistic. In same mode the 343-channel patch head and its loss run at full
  crop resolution, which the current valid net never does.
- **Speed:** scaling FLOPs from the measured 0.965 s/it (bs18, compile + fp16 raw) gives ~0.6 s/it at bs2 160³,
  so 75k iterations would take ~13 h. Treat this as a lower bound. The 512-channel layers at 5³–20³ and the
  full-resolution head use the GPU poorly.
- Each iteration sees ~3× more loss voxels than today but only 2 crops instead of 18. Iteration counts are
  therefore not directly comparable to earlier runs. Record crops seen and loss voxels seen as well.
- For scale: the paper trains every variant for 250k iterations at bs 2 (500k crops). 75k at bs 2 is 150k crops.

## Shared setup

| | value |
|---|---|
| Base / preset | `original_default.toml` + `our-noenc-c1k_v1.1` + `expv2_e1_common.toml` (compile, fp16 raw; its FMAPs 15 is overridden) + `expv3_s6d2w{64,32}.toml` (architecture, same padding) + a shape overlay (`expv3_crop160_bs1.toml` or `expv3_crop128_bs2.toml`) + `expv3_pre.toml` |
| Account / GPU | `torch_pr_61_general` ("tandon_general" in the draft), `l40s_public`, `--gres=gpu:1`. Fall back to `tandon_priority` if the queue is deep (see dsexp / OP-EXP 7 notes) |
| Pretrain data | 1k_v1.1 train, all 796 volumes (the "n=1000" convention) |
| Finetune data | completely train, 18 volumes |
| Validation | completely val, 5 volumes → `train/val_metrics.csv`. **Lowest val loss selects the checkpoint** in both pretrain and finetune (decided 2026-11-17). 1k_v1.1 val, first 5 volumes → `val_metrics_1k_v1.1.csv` (extra, not used for selection) |
| Test | completely test (7 volumes), sparse pred_affs pipeline (`pred_sparse_affs.toml`), `vi_th_0_3`, `eval_rm1500` (OP-EXP 5 best) |
| Metrics reported | S, avF1, C, tp, FS, FM, nPred / nGT, plus completely val loss at the selected checkpoint |
| Experiment ids | baseline finetune `expv3-e3.0-ft-baseline`; W64 `expv3-e3.1-pre-s6d2w64` → `expv3-e3.2-ft-s6d2w64`; W32 `expv3-e3.4-pre-s6d2w32` → `expv3-e3.5-ft-s6d2w32`; overlays in `configs/exp/expv3/` |
| Driver | `sbatch/ppp/expv3/expv3.py` (tick loop `expv3_tick.sh`, job name `expv3_tick`; state `metrics/ppp/expv3/state.json`; status page `md_guides/experiments/expv3_status.md`). W64 / W32 lanes wait in `hold` until `expv3.py release --lane W64 --iters N --time HH:MM:SS --overlay <shape overlay>` |

## Plan

- [ ] **P: Prerequisites**
  - [x] P1: Shape check (2026-11-17, CPU, `UnetModelWrapper` built from the merged config stack).
    - W64: 97.38 M UNet params; encoder 64/128/256/512/512/512, decoder (top first) 64/128/256/512/512.
    - W32: 60.07 M; encoder 32/64/128/256/512/512, decoder 32/64/128/256/512.
    - Both match the paper's Table 1 (97.39 / 60.08 M).
    - Same padding gives output = input at 64³ and 96³ in train and eval mode. 80³ is rejected (not a multiple
      of 32).
  - [x] P2: Same-padding inference. `predict_no_gp.py` treats a same-padding net as context = 0 and stitches tiles
    edge to edge. Every tile border was computed against zero padding, so `pred_affs` will have seams that
    vote_instances can split along.
    - Fix (approved 2026-11-17; show the exact edit list before changing code): a new `[model] test_same_margin = m`
      (e.g. 32). Predict drops m voxels on each side of every tile's output, so output = input − 2m and the existing
      valid-style context and tiling logic is reused unchanged. This is the same idea as nnU-Net's overlapping
      sliding window, which the paper uses. The predict tile can then be bigger than the training crop: 224³
      with m = 32 works, but 256³ tiles run out of memory on an L40S.
    - Check: predict one completely val volume both with and without the margin, then diff the `pred_affs` along
      tile borders.
    - Implemented 2026-11-17 in `predict_no_gp.py`; opt-in, and `test_same_margin` unset or 0 keeps the old
      behaviour. Test stack overlay `expv3_predict.toml`: 224³ tiles, margin 32, bs 1.
    - Seam tool: `sbatch/ppp/tools/compare_same_margin.py A.zarr B.zarr --tile T` reports |ΔP(fg)| and
      fg-call flips by distance to A's tile borders.
    - Functional test passed 2026-11-17 (job 23538692, smoke checkpoint, one completely val volume):
      - predict logged `test_same_margin 32: keeping output [160,160,160] of input [224,224,224]`, context 32;
      - it wrote 75 overlapping tiles and a complete `pred_affs` / `pred_numinst` of the right shape;
      - total time ~4 min.
    - Seam check: when the EXP 3.1 pretrain reached its 5k checkpoint, the driver ran
      `sbatch/ppp/expv3/p2_seam_check.sh`.
      - It predicted one completely val volume with margin 0 (abutting 160³ tiles) and margin 32 (production),
        both on the same 160³ output grid.
      - A planned margin-48 arm at 256³ ran out of memory (a 216 GiB allocation), so it and its afterok compare job
        were dropped. Margin 0 vs 32 was compared by hand on 2026-11-18.
      - **Done:** margin 32 is sufficient. The result table is under Results below; the report is
        `metrics/ppp/expv3/p2_seam_check.md`.
  - [ ] P3: Smoke / profile run, 300 iterations each (`expv3_smoke.toml`). Record:
    - steady s/it and peak memory;
    - finite loss at lr 4e-4 (the net has no normalization);
    - that compile, fp16 raw and fast_affinities all work with same padding;
    - that val runs on the same-padded val crops.
    - Then set the crop, batch size and pretrain length.
    - **Round 1 (jobs 23537164–72) failed in `torch.compile`.** Inductor cannot compile the loss kernel once the
      343-channel patch output has ≥ 2³¹ elements. At 2 × 160³ × 343 = 2.81e9, Triton fails on int64/uint32
      indexing. bs 2–4 at 160³ and bs 4 at 128³ all exceed the limit.
      - With compile, bs × crop³ must stay below 6.26 M voxels.
      - The eager loss is a full-size elementwise BCE too, at ~11 GB per fp32 intermediate for 2 × 160³.
      - Candidates are now **160³ bs 1** (4.10 M loss voxels/it) and **128³ bs 2** (4.19 M, the paper's
        regime). Both give ~1.6× today's 2.53 M per iteration.
      - The same shape overlay is used for W64 and W32, so only the width differs between the arms.
    - **Round 2 (2026-11-17):**
      - compiled: W64 160³ bs1 (23537705), W64 128³ bs2 (23537707), W32 160³ bs1 (23537708), W32 128³ bs2
        (23537710);
      - eager check: W64 160³ bs2 without compile (23537714).
    - Results, 300 its, L40S, compile + fp16 raw + fast_affinities. s/it is the median over iterations 51–300;
      memory is the nvidia-smi peak, including the val pass.

      | arm | s/it | first it (compile) | loss it1 → 300 | val loss @ 300 (completely / 1k_v1.1) | val time (completely / 1k_v1.1) | peak mem | loss voxels/s |
      |---|---|---|---|---|---|---|---|
      | W64 160³ bs1 | 0.415 | 51 s | 1.75 → 0.038 | 0.128 / 0.089 | 217 / 77 s | 28.4 GB | 9.9 M |
      | **W64 128³ bs2** | 0.425 | 59 s | 1.82 → 0.025 | 0.139 / 0.065 | 134 / 42 s | 28.9 GB | 9.9 M |
      | W32 160³ bs1 | 0.278 | 49 s | 1.79 → 0.066 | 0.138 / — | 222 / — s | 26.3 GB | 14.8 M |
      | **W32 128³ bs2** | 0.281 | 56 s | 1.78 → 0.027 | 0.140 / 0.070 | 131 / 41 s | 31.9 GB | 14.9 M |
      | W64 160³ bs2, eager | — | — | — | — | — | OOM (10.5 GB alloc at 37.5 GB used) | — |

    - Findings:
      - Same padding, compile, fp16 raw, fast_affinities and both val sets all work. The loss is finite at
        lr 4e-4.
      - At the same width, both shapes push the same loss voxels per second. W32 is 1.5× faster than W64.
      - The current net runs 2.5 M loss voxels/s (bs18 × 52³ / 0.965 s), so per voxel the same-padding nets
        train 4–6× faster.
      - Val loss at 300 is not comparable to the old net's: different crop sizes, and same-padded borders.
    - **Decision (approved rule, Decision 1):** 160³ cannot fit bs ≥ 2, so use the 128³ fallback,
      `expv3_crop128_bs2.toml`, for both arms. This is also the paper's regime and validates ~40% faster.
  - [x] P4: Length (2026-11-17): **75k iterations** (150k crops). One job per arm, no chaining needed; the driver
    still chains on timeout.
    - W64: 75k × 0.425 s ≈ 8.9 h, + 30 val passes × ~3 min ≈ 1.5 h, + startup → ~10.6 h; job limit 12 h.
    - W32: 75k × 0.281 s ≈ 5.9 h, + 1.5 h val → ~7.6 h; job limit 10 h.
    - Finetunes: 6k its ≈ 0.7 h + 6 val passes; job limit 3 h.
    - Launched 19:51 by `expv3.py release`: EXP 3.1 job 23538647, EXP 3.4 job 23538649 (`torch_pr_61_general`).
  - [ ] P5 (only if P3 shows divergence at 4e-4): a short check at 2e-4. Do not add normalization layers here;
    that would be a separate arm.
- [ ] **EXP 3.0: Baseline: current architecture, same protocol** (approved 2026-11-17)
  - Reuse `vince-data_n1000` (Dataset EXP 1.1: 1k_v1.1 n1000, 35k iterations, FMAPs 15) at its **best completely
    val** checkpoint, 5k (val loss 0.2296).
  - Finetune with `dsexp_ft.toml` + `dsexp_ft_1k_v1.1.toml`, which already match this plan: 6k iterations, peak lr
    5e-5, 10% warmup + cosine, ckpt + val every 1k.
  - Test at rm1500. Cost is one ~2 h finetune plus a test chain.
  - Differs from Dataset EXP 1.3, which selects the pretrain checkpoint by 1k_v1.1 val (25k).
  - Reference scores so far (completely test, rm1500): expv2 EXP 1.3 @ 23000 S 0.305 (best); OP-EXP 6.6 LoRA
    0.291; EXP 0 from scratch 0.259.
  - Launched 2026-11-17 19:33 by the driver: seeded from `vince-data_n1000` @ 5000; finetune job 23537352 (3.5 h
    limit), `expv3-e3.0-ft-baseline`. The test chain follows automatically.
- [ ] **EXP 3.1: Pretrain** S6D2W64 on 1k_v1.1 (796 volumes)
  - Peak lr 4e-4, 10% linear warmup + cosine to 0, Adam, AMP, length from P4.
  - Checkpoint + val every 2.5k (approved 2026-11-17, vs 5k in dsexp). With the old net, completely val loss was lowest at
    5k, the first eval point (EXP 1.1 and 1.2), so a coarse grid can miss the best checkpoint. 75k / 2.5k = 30
    checkpoints × 1.2 GB ≈ 36 GB; prune all but the best after selection.
  - Select the checkpoint with the lowest completely val loss. Also log the 1k_v1.1 val curve.
- [ ] **EXP 3.2: Finetune** EXP 3.1's best checkpoint on completely train (18 volumes)
  - Seed with `sbatch/ppp/expv2/seed_finetune_ckpt.py` (fresh Adam) and a generated overlay with
    `max_iterations = N + 6000`, `start_iter = N`.
  - 6k iterations, peak lr 5e-5, 10% warmup + cosine, ckpt + val every 1k, same crop and batch size as EXP 3.1.
  - Val: completely (selects) + 1k_v1.1 (extra).
  - Note: 6k iterations at bs2 is only ~12k crops, vs ~108k at bs18 in earlier finetunes.
- [ ] **EXP 3.3: Test** EXP 3.2's best completely-val checkpoint on completely test (7 volumes) at rm1500, using
  the P2 inference settings. Compare against EXP 3.0 and the reference scores above.
- [ ] **EXP 3.4–3.6: W32 arm** (S6D2W32; Decision 7, approved 2026-11-17). Same as 3.1–3.3 apart from the width
  (`expv3_s6d2w32.toml`): pretrain `expv3-e3.4-pre-s6d2w32` → finetune `expv3-e3.5-ft-s6d2w32` → test (3.6). It
  uses the same shape overlay and length as W64.

## Results (fill in)

| Exp | Run | Best ckpt (completely val loss) | S | avF1 | C | tp | FS | FM | nPred / nGT |
|---|---|---|---|---|---|---|---|---|---|
| 3.0 | `vince-data_n1000` @ 5000 → `expv3-e3.0-ft-baseline` | 6000 (0.1181) | 0.143 | 0.054 | 0.233 | 0.141 | 1.29 | 0.14 | 26 / 4.9 |
| 3.1 | `expv3-e3.1-pre-s6d2w64` | 55000 (0.0800) | — | — | — | — | — | — | — |
| 3.2 / 3.3 | `expv3-e3.2-ft-s6d2w64` | 57000 (0.0463) | **0.258** | 0.175 | 0.340 | 0.460 | 0.71 | 0.14 | 13 / 4.9 |
| 3.4 | `expv3-e3.4-pre-s6d2w32` | 12500 (0.0789) | — | — | — | — | — | — | — |
| 3.5 / 3.6 | `expv3-e3.5-ft-s6d2w32` | 14000 (0.0524) | 0.126 | 0.030 | 0.221 | 0.061 | 2.00 | 0.14 | 36 / 4.9 |
| 3.0 @ last | `expv3-e3.0-ft-baseline` @ 11000 (6k finetune steps, no selection) | — | 0.119 | 0.047 | 0.190 | 0.077 | 1.29 | 0.00 | 21 / 4.9 |
| 3.3 @ last | `expv3-e3.2-ft-s6d2w64` @ 61000 (6k finetune steps, no selection) | — | **0.257** | 0.187 | 0.327 | 0.408 | 1.00 | 0.14 | 11 / 4.9 |

clDiceTP: W64 0.725 (57000) / 0.737 (61000); W32 0.089; baseline 0.257 (6000) / 0.224 (11000). Driver finished
2026-11-18 11:38.

Reading (2026-11-18):
- W64 is robust to finetune checkpoint choice: 0.258 at the val-loss pick, 0.257 at the last checkpoint.
- The baseline is poor both ways (0.143 / 0.119), so its gap is not finetune-selection noise.
- The arms differ in how far their pretraining got: baseline 5k of 35k, W32 12.5k of 75k, W64 55k of 75k. The
  pretrain checkpoints come from the noisy completely-val-loss rule.
- The best old-net run, expv2 EXP 1.3 (0.305), started from a 20k pretrain. So W64's lead over the baseline and
  W32 may come partly from longer pretraining rather than architecture.

The last-checkpoint rows (driver `extra-test`, added 2026-11-18) separate the architecture effect from noise in
val-loss checkpoint selection. On 2026-11-18 the driver was also fixed: a finetune counts as finished once its log
shows the final iteration, since no checkpoint is written when max_it is off the 1k grid (W32: 12,500 + 6,000).

Status 2026-11-18 09:50:
- W64 (3.3) clDiceTP is 0.725 vs 0.257 for the baseline.
- The protocol-matched baseline is handicapped by the selection rule: completely val loss picked pretrain 5k
  (of 35k) and then finetune 6000, i.e. only 1k finetune steps. Its finetune aff_f1 was still rising (0.42 at
  6k → 0.48 at 9k).
- Pretrain completely val loss is flat and noisy after ~10k for both new nets (0.08–0.10). So the picks at
  55k (W64) and 12.5k (W32) are mostly noise.
- W32 finetune trained all 6k steps (to 18,500), but no checkpoint exists at 18,500: it is not a multiple of
  the 1k interval, because the pretrain pick was 12,500. The driver's completion check waited for that
  checkpoint and stopped the lane. That is a driver bug; the finetune results are intact.
- References (rm1500): expv2 EXP 1.3 0.305, OP-EXP 6.6 LoRA 0.291, Dataset EXP 0 (from scratch) 0.259.

**P2 seam check result** (W64 pretrain @ 5000, val volume R22C03, margin 0 vs margin 32; m48 run at 256³ OOMed
asking for 216 GiB, so 256³ predict tiles do not fit):

| distance to tile border | fg-call flips / fg voxels |
|---|---|
| 0–2 vox | 50.4% |
| 2–4 | 25.3% |
| 4–8 | 18.0% |
| 8–16 | 8.4% |
| 16–32 | 1.0% |
| ≥ 32 | 0.7% (noise floor) |

Abutting tiles produce real seams. The border effect has decayed to the noise floor by 16–32 voxels, so the
production margin of 32 is sufficient.

The driver's status page `md_guides/experiments/expv3_status.md` updates this live.

## Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Padding: same, crop 160³ (128³ fallback) | approved 2026-11-17 |
| 2 | Checkpoint selection: lowest completely val loss, pretrain and finetune | approved 2026-11-17 |
| 3 | P2: implement `test_same_margin` for same-padding inference | approved 2026-11-17 (edit list still shown before coding) |
| 4 | Run the EXP 3.0 baseline | approved 2026-11-17 |
| 5 | Crop / batch size and pretrain length: set from the P3 smoke (75k target; chain jobs or cap) | set 2026-11-17: 128³ bs 2, 75k its, one 12 h (W64) / 10 h (W32) job |
| 6 | Pretrain checkpoint + val every 2.5k | approved 2026-11-17 |
| 7 | W32 arm (S6D2W32, the paper's guideline width for few classes; ~4× fewer FLOPs) as EXP 3.4–3.6 | approved 2026-11-17 |
| 8 | Crop / batch size limited by the torch.compile 2³¹ limit to 160³ bs 1 or 128³ bs 2 (eager 160³ bs 2 OOMs) | 128³ bs 2 chosen 2026-11-17 (rule from Decision 1) |

## Risks and notes

- Same padding has never run end to end in this repo: train, val, predict, label and evaluate. P1–P3 exist for
  that reason.
- No normalization in a 97 M-param, 6-stage net at bs 2. If training is unstable, the fix to try is lr (P5).
  Adding instance norm would confound the comparison and is a separate arm.
- OP-EXP 4 and OP-EXP 7 showed completely val loss is a weak proxy for test S. Selection still uses it (decided
  2026-11-17), so if EXP 3.3 disappoints, check whether a different checkpoint would have scored better before
  concluding the architecture is worse.
- Finetune on completely only. OP-EXP 7 showed that mixing synthetic volumes into the finetune hurts.
- L40S speed settings: `md_guides/ppp/ppp_train_performance.md`. Its numbers are for the old valid net; re-measure
  in P3.
