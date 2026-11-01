# Web viewer: FISBe website metric names

The FISBe leaderboard and this repo’s web viewer display instance scores under
short **website names**. On disk (eval TOML / `summary.csv`) the same values use
longer **pipeline keys**. Aggregate heatmap and Scoring Metrics share one mapping.

Canonical reference:
[`evaluate-instance-segmentation/docs/METRICS.md`](../../evaluate-instance-segmentation/docs/METRICS.md)
(“Website leaderboard column mapping”).

## Conversion table

| Website name | Pipeline field | TOML location |
|---|---|---|
| **S** | `avg_f1_cov_score` | `[general]` |
| **avF1** | `avFscore19` | `[confusion_matrix]` (summary) |
| **C** | `avg_gt_skel_coverage` | `[general]` |
| **clDiceTP** | `avg_TP_05_cldice` | `[general]` |
| **tp** | `TP_05_rel` | `[general]` |
| **FS** | `FS` | `[general]` |
| **FM** | `FM` | `[general]` |
| **nPred** | `Num Pred` | `[general]` |
| **nGT** | `Num GT` | `[general]` |

**nPred** / **nGT** are instance counts (not ranking scores). On the Aggregate
heatmap they render as plain numbers (`color_scale: false`); other columns keep
per-column min→max colour scaling.

**S** formula: \(S = 0.5 \cdot \mathrm{avF1} + 0.5 \cdot C\).

METRICS.md documents `general.aggregate_score` for **S**, but
`evaluate-instance-segmentation` writes `general.avg_f1_cov_score` (same
formula). The viewer maps **S** ← `avg_f1_cov_score`.

## Where this lives in code

| Piece | Path |
|---|---|
| Shared ordered specs | [`web/server/services/fisbe_metrics.py`](../../web/server/services/fisbe_metrics.py) (`FISBE_LEADERBOARD_SPECS`, `leaderboard_from_toml`) |
| Aggregate heatmap | [`web/server/services/aggregate_metrics.py`](../../web/server/services/aggregate_metrics.py) aliases `METRIC_SPECS` / `PPP_INSTANCE_METRIC_SPECS` → shared specs |
| Per-sample API | [`web/server/services/metrics.py`](../../web/server/services/metrics.py) / [`ppp_metrics.py`](../../web/server/services/ppp_metrics.py) → payload field `leaderboard` |
| Scoring Metrics UI | [`web/client/src/components/MetricsPanel.tsx`](../../web/client/src/components/MetricsPanel.tsx) — **FISBe leaderboard** section first; raw General / CSV / thresholds keep pipeline keys |

## On-disk sources (unchanged)

- **BiaPy:** `tests/metrics/<stem>.zarr.toml` + optional `test_results_metrics.csv`
- **PPP instances:** `{split}/evaluated/.../<stem>_vote_instances_*.toml` (or `summary.csv` row)
- **PPP numinst:** voxel `_pred_metrics.csv` only — no instance leaderboard values

Do not rename keys inside eval outputs; only the UI/API display layer converts.
