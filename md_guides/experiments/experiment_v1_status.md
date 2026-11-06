# experiment_v1 run log

Plan: [`experiment_v1.md`](experiment_v1.md). Started 2026-11-05 ~10:45. Meeting 2026-11-06 12:00.

## Decisions (2026-11-05, with user)

- **EXP 1 reuses** the existing runs `our-noenc-c1k_v1` and `our-noenc-c1k_v1_u8`. They were trained on L40S
  with fp32 raw transport, bs18, `max_train_samples=1000`. `1k_v1`/`1k_v1_u8` have only 796 train
  volumes (796/109/95), so n=1000 means all 796.
- **num_fmaps matches per dataset:** v1 → 20, u8 → 24 (as in the preset TOMLs). The n effect within a
  dataset is clean. v1 vs u8 is confounded by model size.
- **EXP 2 (n=500)** = the first 500 sorted train volumes (`exp_v1_n500.toml`), trained on H200
  (`exp_v1_h200.toml`: 50k iterations, ckpt every 2k, 14 workers, fp16 raw transport, snapshots every 25k).
- **EXP 3** starts from the best EXP 1 checkpoint (by completely val) as soon as EXP 1 val is done,
  without waiting for EXP 2. `sbatch/ppp/exp_v1_finetune.sh` copies the checkpoint (model + Adam
  state) into `exp3-…-ft<N>-completely/train/` and trains to N+50k on `completely` only.
- **EXP 3 is shortened for the meeting:** no extra GPU. It starts when an EXP 2 run frees an H200
  (~22:00), from the best EXP 1 ckpt validated by then. For the meeting, report EXP 3 at +30k
  fine-tune iterations (val first, then test). Training continues to +50k afterward.
- Priority for the val/test lanes before the meeting: EXP 1 val → EXP 1 test (best ckpt) →
  EXP 3 val (+30k) → EXP 2 val (30k–50k). Whatever doesn't fit runs after the meeting.
- **Only checkpoints >= 30k are validated/tested** (user, 2026-11-05: 10k/20k are rarely competitive).
  For EXP 3 this means >= 30k fine-tune iterations (N+30k, N+40k, N+50k), so the meeting gets EXP 3 at +30k
  (replaces the earlier +20k/+30k plan). The 20k/10k EXP 1 val pipelines were cancelled.
- The u8 test@56000 pipeline (21868139/40/48/50, general account) was left running.
- Val/test: `sbatch/ppp/ppp_submit_chain.sh` chains full array pipelines (`--pct 3`, tandon_advanced),
  each waiting `afterany` on the previous evaluate (which strips pred_affs). Two lanes run in parallel.

## Jobs

| What | Job(s) | Notes |
|---|---|---|
| EXP 2 train `exp2-noenc-c1k_v1-n500` | 21975068 | h200_public, 14 h limit |
| EXP 2 train `exp2-noenc-c1k_v1_u8-n500` | 21975069 | h200_public, 14 h limit |
| EXP 1 val lane A `our-noenc-c1k_v1` 50k→30k | eval 21975124, 21975132, **21975136** (last) | [`logs/exp1_val_our-noenc-c1k_v1.txt`](logs/exp1_val_our-noenc-c1k_v1.txt) |
| EXP 1 val lane B `our-noenc-c1k_v1_u8` 50k→30k | eval 21975180, 21975185, **21975203** (last) | [`logs/exp1_val_our-noenc-c1k_v1_u8.txt`](logs/exp1_val_our-noenc-c1k_v1_u8.txt) |

Results: `metrics/ppp/<exp>/val/evaluated/<ckpt>/patch_threshold_0_3/.../summary.csv`.

## Resume checklist (written 2026-11-05 ~12:45, session closed)

Check: `squeue -j 21975068,21975069,21975124,21975132,21975136,21975180,21975185,21975203,21868150 -o "%.10i %.32j %.8T %.10M %R"`
and `sacct -X -j <ids> -o JobID,JobName%35,State,Elapsed` for finished ones.

| Job | What | When done → next action |
|---|---|---|
| 21975068 | EXP 2 train v1 n500 (H200, started 12:17, 0.63 s/it, ETA ~21:15) | H200 free → submit EXP 3 via `exp_v1_finetune.sh` from best EXP 1 ckpt; then chain EXP 2 v1 val 30k/40k/50k |
| 21975069 | EXP 2 train u8 n500 (was held by per-user GPU cap at 12:35) | chain EXP 2 u8 val 30k/40k/50k |
| 21975124 / 21975132 / **21975136** | EXP 1 v1 val evaluate 50k / 40k / 30k (lane A) | after 21975136: pick best v1 ckpt, chain EXP 1 v1 test on it |
| 21975180 / 21975185 / **21975203** | EXP 1 u8 val evaluate 50k / 40k / 30k (lane B) | after 21975203: pick best u8 ckpt, chain EXP 1 u8 test on it |
| 21868150 | u8 test@56k evaluate (pre-existing, general account) | info only; frees ~190 GB |

Pipeline failure symptom: an evaluate stuck `PENDING (DependencyNeverSatisfied)` = a label task failed → lane blocked.

Per-user GPU cap: QOS gpu48 showed 6 GPUs in use and blocked a 7th (12:35). Val L40S tasks and H200 training compete;
`scontrol update nice=` is rejected on this cluster. Training jobs already have higher priority than val jobs.

## Update 2026-11-05 ~21:20

EXP 1 val (completely/val, mean of 5 vols; vi_th_0_3, eval_rm200). Best = 50k for both runs:

| Run | ckpt | avg_f1_cov | avF1 | Num Pred (GT 4.2) |
|---|---|---|---|---|
| v1 (fmaps 20) | 30k / 40k / **50k** | 0.052 / 0.064 / **0.083** | 0.0009 / 0.0010 / **0.0022** | 243 / 346 / 227 |
| u8 (fmaps 24) | 30k / 40k / **50k** | 0.033 / 0.067 / **0.083** | 0.0009 / 0.0012 / **0.0013** | 226 / 517 / 619 |

Both are near zero on real data (massive over-segmentation). Still improving at 50k.

| Job | What |
|---|---|
| eval 22023299 | EXP 1 test v1 @ 50k (completely/test) |
| eval 22023317 | EXP 1 test u8 @ 50k |
| 22025054 | EXP 3 fine-tune `exp3-noenc-c1k_v1-ft50000-completely` (user chose v1@50k), h200_public, 50k→100k |
| 22025056 / 22025057 | L40S real-data 300-it speed test, EXP 3 setup (completely, fmaps 20, fast_aff) + `raw_fp16.toml`, without / with channels_last |
| 21975068 | EXP 2 v1 train at 43k at 21:00 (0.63 s/it) |
| 21975069 | EXP 2 u8 train: still never started (h200_public: 3 nodes, ~1850 pending) |

L40S prediction before the test: 1.46–1.54 s/it if the fp16 hypothesis holds, else ~1.95–2.0 (H200: 0.63).

## Update 2026-11-06 ~00:10 — meeting plan (user decisions)

- Skip all further validation. Skip EXP 2 u8 n500 (21975069 cancelled at ~3.5k its; 2k ckpt kept, resumable with the same id).
- n=1000 is reported from the existing **56k** test results (v1 done: test mean avg_f1_cov 0.212, avF1 0.0055;
  u8 56k pipeline 21868139/40/48/50 still on torch_pr_61_general, 4/7 predicted; account change to tandon was rejected).
- EXP 1 50k test chains cancelled. L40S speed tests (22025056/57) cancelled; rerun after the meeting
  (overlay `raw_fp16.toml` + `profile_noenc_completely_l40s.toml`, ± `channels_last.toml`).
- EXP 3 overlay `exp3_ft_noenc-c1k_v1_50000.toml` set to `max_iterations = 80000` (+30k), so its test chains `afterok`.
  To continue to 100k later: `./sbatch/ppp/exp_v1_finetune.sh our-noenc-c1k_v1 our-noenc-c1k_v1 50000` (rewrites overlay to 100k, resumes).

| Job | What | Starts |
|---|---|---|
| 22025054 | EXP 3 train 50k→80k on completely (h200_public) | pending |
| eval **22040724** (pred 22040713) | EXP 2 v1 n500 test @ 50k | now |
| eval **22040734** (pred 22040725) | EXP 3 test @ 80k | afterok:22025054 |
| eval **22041776** (pred 22041772 = idx 4–6 only; labgraph 22041774, label 22041775) | EXP 1 u8 n1000 test @ 56k, moved to tandon_advanced (general chain 21868139/40/48/50 cancelled; idx 0–3 preds kept) | now |

## Update 2026-11-06 09:05

- EXP 2 v1 n500 @ 50k test DONE (eval 22040724): mean of 7 completely/test vols avg_f1_cov **0.2295**, avF1 **0.0070**, Num Pred 530 (GT 4.9).
  Compare n1000 v1 @ 56k: 0.2124 / 0.0055 / 604. n500 ≥ n1000 on 5 of 7 volumes; both near-zero avF1 (over-segmentation).
- EXP 1 u8 n1000 @ 56k: all predict/labgraph/label done; evaluate 22041776 running since 08:55 (~1 h).
- EXP 3 train 22025054 NEVER STARTED (h200_public ~1460 pending); its 80k test chain (22040725/26/27/34) waits on it.
  No EXP 3 result for the meeting.

## Update 2026-11-06 14:20 — EXP 3 cancelled, config reorg finished

- **EXP 3 cancelled** (user decision): train 22025054 and its test chain 22040725 (pred) / 22040726 (labgraph) /
  22040727 (label) / 22040734 (eval) were `scancel`led before starting. No EXP 3 jobs are queued.
- Seeded dir `metrics/ppp/exp3-noenc-c1k_v1-ft50000-completely/` removed. It held only a byte-identical copy of
  `our-noenc-c1k_v1/train/train_net_checkpoint_50000`, which is kept.
- Overlay moved to `setup01/configs/archive/exp3_ft_noenc-c1k_v1_50000.toml`.
- To restart EXP 3, rerun `./sbatch/ppp/exp_v1_finetune.sh our-noenc-c1k_v1 our-noenc-c1k_v1 <ckpt>`. It reseeds
  the dir and writes a fresh overlay to `configs/exp/`. Set `FT_ITERS` for a shorter run.
- setup01 configs now live in `setup01/configs/<role>/` (see its `README.md`). The temporary flat
  `setup01/*.toml` compat symlinks are deleted. Overlay names in this file still work as bare `--config` names,
  except `exp3_ft_noenc-c1k_v1_50000.toml`, which is archived and needs `--config archive/exp3_ft_noenc-c1k_v1_50000.toml`.
