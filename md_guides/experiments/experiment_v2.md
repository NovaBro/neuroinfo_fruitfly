## TODO
- [ ] Optimization Experiments:
  - use just 1k_v1.1 dataset, n=500 train samples, noenc, tandon advanced account, l40s 
  - Experiment Goal: Increase convergence speed and model performance
  - [x] OP-EXP 1: Find peak lr for lr scheduler, 30K Iterations, FMAPs=20, no lr scheduler
    - Experiment Goals: Finding the highest valid learning rate for lr scheduler later in OP-EXP 3.
    - [x] OP-EXP 1.1: lr=0.0001 
    - [x] **OP-EXP 1.2: lr=0.0004**
    - [x] OP-EXP 1.3: lr=0.0008
    - [x] OP-EXP 1.4: lr=0.001
    - [x] OP-EXP 1.5: test each experiment's best validation model on 1k_v1.1 test split
  - [x] OP-EXP 2: Modifying noenc model's fmaps, 30K Iterations, choose best lr from OP-EXP 1.5
    - Experiment Goals: See if we can get away with lower fmaps and save on compute costs
    - [x] OP-EXP 2.1: FMAPs=20 
    - [x] **OP-EXP 2.2: FMAPs=15** 
    - [x] OP-EXP 2.3: FMAPs=10
    - [x] OP-EXP 2.4: test each experiment's best validation model on 1k_v1.1 test split
  - [x] OP-EXP 3: Adding Learning Rate Scheduler, 60k iterations, choose best FMAPs from OP-EXP 2.4
    - Experiment Goals: Find best learning rate scheduler setup
    - [x] OP-EXP 3.1: baseline, choose best from OP-EXP 1.5, no scheduler
    - [x] OP-EXP 3.2: linear warmup (5% total iterations, OP-EXP 1's largest valid lr) + cosine 
    - [x] OP-EXP 3.3: linear warmup (1% total iterations, OP-EXP 1's largest valid lr) + cosine 
    - [x] **OP-EXP 3.4: linear warmup (10% total iterations, OP-EXP 1's largest valid lr) + cosine** 
    - [x] OP-EXP 3.5: test each experiment's best validation model on 1k_v1.1 test split
  - [x] Conclusion, use OP-EXP 1.2, OP-EXP 2.2, OP-EXP 3.4. Refer to md_guides/experiments/opexp_status.md results
  - [ ] OP-EXP 4: Hyperparameter tuning for fine tuning
    - Start from experiment metrics/ppp/expv2-e1.1-pre-n1000 for testing fine tune hyperparameters.
    - linear warmup 10% + cosine, fine tune 6k iterations
    - [x] **OP-EXP 4.1 max_lr=5e-5, test on completely test split**
    - [x] OP-EXP 4.2 max_lr=1e-5, test on completely test split
    - [x] OP-EXP 4.3 max_lr=5e-6, test on completely test split
    - Best validation still at early iterations! Overfitting.
    - Results (best completely-val ckpt = 21000 for all three, rm200, 7 completely test vols): S 4.1 0.275 / 4.2 0.281 / 4.3 0.272
      vs reference EXP 1.3/1.5 (max_lr 1e-4, 5% warmup) 0.282. No gain; lower lr leaves more fragments (nPred 134 / 227 / 278
      per volume vs 120) and lowers avF1. 4.1 has the lowest val loss (0.1087) but not the best S. Refer to md_guides/experiments/opexp45_status.md
  - [x] OP-EXP 5: Tuning metric threshold
    - Start from experiment metrics/ppp/expv2-e1.3-ft-n1000-completely for tuning metrics.
    - [x] OP-EXP 5.1 remove_small_components=500, test on completely test split
    - [x] OP-EXP 5.2 remove_small_components=1000, test on completely test split
    - [x] OP-EXP 5.3 remove_small_components=1500, test on completely test split
    - Results (EXP 1.3 @ 23000 labels): S rm200 0.282 / rm500 0.285 / rm1000 0.299 / **rm1500 0.305**; avF1 0.035 -> 0.191 and
      nPred 120 -> 13 per volume, C 0.530 -> 0.418; matched instances unchanged (tp, clDiceTP). S still rising at 1500.
      Refer to md_guides/experiments/opexp45_status.md
  - [x] OP-EXP 6: Fine Tuning with LoRA setup, hyperparameter search
    - Start from experiment metrics/ppp/expv2-e1.1-pre-n1000 for testing fine tune with LoRA.
    - Setup (2026-11-10): 6k its, linear warmup 10% + cosine, LoRA on encoder + decoder convs (l_conv, r_conv), output heads
      fully trainable; val on completely (selects ckpt) + 1k_v1.1 (n=5); test best ckpt with remove_small_components=1500;
      tandon_priority account. LoRA GPU unit test (sbatch/ppp/tools/test_lora.py) passed. Refer to md_guides/experiments/opexp6_status.md
    - [x] OP-EXP 6.1 r=16 a=16 max_lr=5e-5, test on completely test split
    - [x] OP-EXP 6.2 r=32 a=32 max_lr=5e-5, test on completely test split
    - [x] OP-EXP 6.3 r=64 a=64 max_lr=5e-5, test on completely test split
    - [x] OP-EXP 6.4 r=16 a=16 max_lr=5e-4, test on completely test split (added: LoRA usually needs a higher peak lr)
    - [x] OP-EXP 6.5 r=32 a=32 max_lr=5e-4, test on completely test split
    - [x] OP-EXP 6.6 r=64 a=64 max_lr=5e-4, test on completely test split
    - Results (best completely-val ckpt, rm1500, 7 completely test vols): S 6.1 0.284 / 6.2 0.288 / 6.3 0.266 / 6.4 0.287 /
      6.5 0.284 / **6.6 0.291** (avF1 0.15-0.17, C 0.39-0.42, nPred 16-18 / nGT 4.9). All below the full finetune EXP 1.3 at
      rm1500 (S 0.305) and above the from-scratch EXP 0 (0.259); no clear rank or lr trend (+-0.02 is within noise for 7 vols).
      Note: 6.4-6.6 validated on a different val-crop set (train.py change 2026-11-10 21:13, reverted 2026-11-11), so their
      val losses read ~0.025 high vs 6.1-6.3; test S is unaffected.
  - [x] OP-EXP 7: Fine Tuning with LoRA setup, hyperparameter search
    - Start from experiment metrics/ppp/expv2-e1.1-pre-n1000 for testing fine tune with LoRA.
    - Setup (2026-11-10): 6k its, linear warmup 10% + cosine, LoRA on encoder + decoder convs (l_conv, r_conv), output heads
      fully trainable; val on completely (selects ckpt) + 1k_v1.1 (n=5); test best ckpt with remove_small_components=1500; tandon_priority account.
    - Details (2026-11-11): alpha = r; 1k_v1.1 subset = first N sorted train volumes (nested 18 ⊂ 36 ⊂ 72), drawn uniformly per
      volume with the 18 completely volumes (real share 50% / 33% / 20%); new [data] max_train_samples_per_dataset. 3 h jobs,
      spread over general / tandon_advanced / tandon_priority while the l40s queue was ~600 deep. Refer to md_guides/experiments/opexp7_status.md
    - [x] OP-EXP 7.1 r=2 max_lr=5e-4, finetune on completely train split (n=18) and 1k_v1.1 train split (n=18), test on completely test split
    - [x] OP-EXP 7.2 r=4 max_lr=5e-4, finetune on completely train split (n=18) and 1k_v1.1 train split (n=18), test on completely test split
    - [x] OP-EXP 7.3 r=2 max_lr=5e-4, finetune on completely train split (n=18) and 1k_v1.1 train split (n=36), test on completely test split
    - [x] OP-EXP 7.4 r=4 max_lr=5e-4, finetune on completely train split (n=18) and 1k_v1.1 train split (n=36), test on completely test split
    - [x] OP-EXP 7.5 r=2 max_lr=8e-4, finetune on completely train split (n=18) and 1k_v1.1 train split (n=36), test on completely test split
    - [x] OP-EXP 7.6 r=2 max_lr=8e-4, finetune on completely train split (n=18) and 1k_v1.1 train split (n=72), test on completely test split
    - Results (best completely-val ckpt, rm1500): S 7.1 0.222 / 7.2 0.210 / 7.3 0.211 / 7.4 0.225 / **7.5 0.246** / 7.6 0.174.
      Every arm is below OP-EXP 6 (0.27-0.29) and even below the from-scratch EXP 0 (0.259); more synthetic volumes is worse
      (72 -> 0.174). Failure mode is fragmentation: C 0.25-0.33 (vs ~0.41), tp 0.13-0.40 (vs ~0.5), false splits 1.4-2.0
      (vs ~1.2), similar nPred (14-20). Test configs are identical to OP-EXP 6 apart from the training data.
    - Completely val loss did NOT predict this: 7.4 / 7.5 have the lowest val loss of any finetune (0.1034 / 0.1035 vs 0.1098
      for 6.2 and 0.1176 for EXP 1.3), and their voxel val metrics (fg_dice ~0.68, aff_f1 ~0.48-0.53) match OP-EXP 6.
      Conclusion: do not mix synthetic volumes into the finetune; val loss is a weak proxy for S (also seen in OP-EXP 4).
- [ ] Dataset Experiments
  - Experiment Goals: We want to see the effect of scaling synthetic training data on the performance of patchperpix training. And finetuning on the real completely dataset
  - Experiment Setup: tandon_general, l40s, remove_small_components=best from OP-EXP 5 (1500)
  - Note: It says train on n=1000 samples, but the datasets are only n=1000 samples total. Verify that we are not training on the test or val split for any dataset
    - Verified: train/val/test brain ids are disjoint. 1k_v1.1 = 796 / 109 / 95; SWC2IMG takes each brain's 1k_v1.1 split
      (788 / 105 / 93 of 986 renders, so no brain is in train for one dataset and val/test for the other); completely = 18 / 5 / 7.
      n=1000 therefore trains on all 796 (1k_v1.1) / 788 (SWC2IMG) train volumes; n=500 on the first 500 sorted.
  - Runs: since 2026-11-10 on torch_pr_61_tandon_priority (jobs on general sat ~420 deep in the l40s queue).
    Status / job ids: md_guides/experiments/dsexp_status.md
  - [x] EXP 0: Completely Dataset - (experiment base id: baseline_v1_*)
    - Experiment Setup: max_lr=4e-4, FMAPS=15, linear warmup 10% + cosine, train 100k, 
    - Ran 35k its / 10 h (user 2026-11-10, same budget as EXP 1/2) instead of 100k; completely val has 5 volumes (n=7 is the test split)
    - [x] EXP 0.1: train on completely train split n=18 samples, run validation with completely val split (n=7)
      - Best val loss at 10k (0.134); val loss then rises to 0.303 at 35k (overfits the 18 volumes)
    - [x] EXP 0.2: test with best completely val split on the completely test split
      - 10k ckpt, rm1500: S 0.259, avF1 0.169, C 0.348, nPred 14 / nGT 4.9 (EXP 1.3 at rm1500: S 0.305). md_guides/experiments/dsexp_e0_status.md
  - [ ] EXP 1: 1k_v1.1 Dataset - (experiment base id: vince-data_*)
    - Experiment Setup: FMAPS=15, linear warmup 10% + cosine, (pre train 75k with max_lr=4e-4), (fine tune 6k with max_lr=5e-5), 
    - Pretrain ran 35k its / 10 h (user 2026-11-09; was 50k / 20 h, then 55k / 16 h), ckpt + val every 5k
    - [x] EXP 1.1: train on n=1000 samples, run validation with 1k_v1.1 validation (n=5) and completely validation (n=5) split during training.
      - Best 1k_v1.1 val loss at 25k (0.0247); completely val loss best at 5k (0.230), 0.280 at 35k
    - [x] EXP 1.2: train on n=500 samples, run validation with 1k_v1.1 validation (n=5) and completely validation (n=5) split during training.
      - Best 1k_v1.1 val loss at 25k (0.0284); completely val loss best at 5k (0.216), 0.283 at 35k
    - Finetune configs ready (dsexp_ft.toml + dsexp_ft_1k_v1.1.toml: completely train, 6k its, max_lr 5e-5, 10% warmup + cosine,
      val on completely (selects ckpt) and 1k_v1.1 (n=5)); not started yet.
    - [ ] EXP 1.3: Take best model from EXP 1.1 with best 1k_v1.1 validation and finetune on completely
    - [ ] EXP 1.4: Take best model from EXP 1.2 with best 1k_v1.1 validation and finetune on completely
    - [ ] EXP 1.5: take best from EXP 1.3 on completely validation split and test on completely test split
    - [ ] EXP 1.6: take best from EXP 1.4 on completely validation split and test on completely test split
  - [ ] EXP 2: SWC2IMG Dataset - (experiment base id: gaurav-data_*)
    - NEW: Dataset located at /scratch/gw2467/data/, swc2img_1k_v1_part0.sqf swc2img_1k_v1_part1.sqf swc2img_1k_v1_part2.sqf swc2img_1k_v1_part3.sqf
    - Experiment Setup: FMAPS=15, linear warmup 10% + cosine, (pre train 75k with max_lr=4e-4), (fine tune 6k with max_lr=5e-5), 
    - Data prep (fisbe/swc2img_1k_v1, packed in fisbe/sqf/swc2img_1k_v1 as 15 images, 538 GB): raw_normalized =
      clip(raw - per-channel median, 0, 1500) / 1500 (renders sit on a ~2,500 pedestal), stored float16 (quota); train + val
      only (EXP 2 tests on completely). Pretrain 35k its / 10 h, same as EXP 1.
    - [x] EXP 2.1: train on n=1000 samples, run validation with SWC2IMG validation (n=5) and completely validation (n=5) split during training
    - [x] EXP 2.2: train on n=500 samples, run validation with SWC2IMG validation (n=5) and completely validation (n=5) split during training
      - Earlier attempts failed (too many image mounts; then float16 raw in intensity augmentation); both fixed.
      - Results (35k its, ~9.8 h each): SWC2IMG val loss 0.050 -> 0.044, best at 35k (2.1) / 20k (2.2), fg_dice plateaus at
        ~0.59 (1k_v1.1: ~0.83). Completely val loss best at 15k for both (~0.32) and noisy (0.32-0.57); EXP 1's best was
        ~0.22. EXP 2's completely val ran on the other val-crop set (~+0.025), which explains only part of the gap:
        SWC2IMG pretraining transfers worse to real data by this measure.
    - [ ] EXP 2.3: Take best model from EXP 2.1 with best SWC2IMG validation and finetune on completely
    - [ ] EXP 2.4: Take best model from EXP 2.2 with best SWC2IMG validation and finetune on completely
    - [ ] EXP 2.5: take best from EXP 2.3 on completely validation split (n=7) and test on completely test split
    - [ ] EXP 2.6: take best from EXP 2.4 on completely validation split (n=7) and test on completely test split
- NOTE: When running L40s training and testing jobs, use md_guides/ppp/ppp_train_performance.md to get fastest l40s setup