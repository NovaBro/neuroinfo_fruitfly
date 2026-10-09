"""FISBe website leaderboard metric names (pipeline key → display label).

Official column mapping lives in
``evaluate-instance-segmentation/docs/METRICS.md`` ("Website leaderboard
column mapping"). Shared by Aggregate heatmap and per-sample Scoring Metrics.

Note: METRICS.md documents ``general.aggregate_score`` for **S**, but the
eval package writes ``general.avg_f1_cov_score`` (same formula
``0.5 * avF1 + 0.5 * C``). This viewer maps **S** ← ``avg_f1_cov_score``.
"""

from __future__ import annotations

# Ordered FISBe website columns: S · avF1 · C · clDiceTP · tp · FS · FM
#   source="toml-summary" -> toml["summary"][field]   (confusion_matrix avg)
#   source="toml-general" -> toml["general"][field]
# ``higher_is_better`` documents metric orientation for the client legend.
FISBE_LEADERBOARD_SPECS: tuple[dict, ...] = (
    {
        "key": "S",
        "label": "S",
        "source": "toml-general",
        "field": "avg_f1_cov_score",
        "higher_is_better": True,
    },
    {
        "key": "avF1",
        "label": "avF1",
        "source": "toml-summary",
        "field": "avFscore19",
        "higher_is_better": True,
    },
    {
        "key": "C",
        "label": "C",
        "source": "toml-general",
        "field": "avg_gt_skel_coverage",
        "higher_is_better": True,
    },
    {
        "key": "clDiceTP",
        "label": "clDiceTP",
        "source": "toml-general",
        "field": "avg_TP_05_cldice",
        "higher_is_better": True,
    },
    {
        "key": "tp",
        "label": "tp",
        "source": "toml-general",
        "field": "TP_05_rel",
        "higher_is_better": True,
    },
    {
        "key": "FS",
        "label": "FS",
        "source": "toml-general",
        "field": "FS",
        "higher_is_better": False,
    },
    {
        "key": "FM",
        "label": "FM",
        "source": "toml-general",
        "field": "FM",
        "higher_is_better": False,
    },
)


def _as_number(value) -> float | None:
    """Coerce a parsed metric cell to a float, or ``None`` when not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def leaderboard_from_toml(toml_data: dict | None) -> list[dict]:
    """Build ``[{key, label, value, higher_is_better}, ...]`` from a sample TOML.

    Missing fields become ``value: null``. When ``toml_data`` is ``None``, every
    entry still appears with a null value so the client can render the column
    headers consistently.
    """
    general = (toml_data or {}).get("general") or {}
    summary = (toml_data or {}).get("summary") or {}
    rows: list[dict] = []
    for spec in FISBE_LEADERBOARD_SPECS:
        if spec["source"] == "toml-summary":
            raw = summary.get(spec["field"])
        else:
            raw = general.get(spec["field"])
        rows.append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "value": _as_number(raw),
                "higher_is_better": spec["higher_is_better"],
            }
        )
    return rows
