# vesselFM finetune loop progress

Updated by the `/loop 60m` agent. Parent: [`vesselfm_fisbe.md`](vesselfm_fisbe.md).

## Loop

- Interval: **60m**
- Sentinel: `AGENT_LOOP_TICK_vesselfm_finetune`
- Prompt each tick: next unfinished stage → detailed plan → implement; sbatch only for smoke/testing.

## Status

| Stage | Status | Notes |
|-------|--------|-------|
| 0 Env | DONE | |
| 1 Images | DONE | |
| 2 Masks | DONE | job `17846666` |
| 3 Zero-shot | DONE | job `17353283`; mean Dice 0.0618 |
| 4 Patches | SMOKE DONE | job `17846712` (72/10/14); full `PATCHES_TRAIN=32` still TODO |
| 5 Config/ckpt | DONE | `eval_fisbe.yaml` + lightning ckpt |
| 6 Finetune | SMOKE PENDING | job `17847501` (`NUM_SHOTS=8`, `MAX_STEPS=50`) |
| 7 Eval | TODO | after Stage 6 smoke succeeds |

## Next tick priorities

1. Check job `17847501` (finetune smoke). If failed, fix and re-smoke. If OK, mark Stage 6 smoke DONE.
2. If Stage 6 smoke OK: either (a) submit Stage 4 **full** patches, or (b) Stage 7 export+infer smoke with the smoke ckpt.
3. Do **not** start a long full finetune unless patches are full and user/loop decides.
