"""Consolidate and plot BiaPy partitioned training logs.

Parses SLURM train ``.out`` files produced by ``run_biapy-py.py`` train mode,
where data is staged into sequential partitions that each restart epoch
numbering.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Log-line patterns from BiaPy / run_biapy-py.py train output
# ---------------------------------------------------------------------------
# Explicit partition banner: "Partition 3/27: 18 sample(s), ..."
_PARTITION_RE = re.compile(r"Partition\s+(\d+)\s*/\s*(\d+)\s*:")
# Fallback: staging path ".../part_2" before the Partition banner appears
_PART_DIR_RE = re.compile(r"/part_(\d+)(?:\s|/|$)")
# Per-iteration progress: "Epoch: [5]  [10/97]  ..."
_EPOCH_RE = re.compile(r"Epoch:\s*\[(\d+)\]")
# Partition train kickoff: "Start training in epoch 1 - Total: 30"
_START_TRAIN_RE = re.compile(r"Start training in epoch\s+(\d+)")
# End-of-epoch summaries (what we keep — not per-iteration lines)
_TRAIN_STATS_RE = re.compile(r"\[Train\]\s+averaged stats:\s*(.*)")
_VAL_STATS_RE = re.compile(r"\[Val\]\s+averaged stats:\s*(.*)")
_EARLY_STOP_RE = re.compile(r"Early stopping")
_FINISHED_RE = re.compile(r"Finished Training")

# Metric token inside averaged-stats blobs:
#   "loss: 2.0566 (2.8730)"  → last=2.0566, mean=2.8730 (we keep mean)
#   "lr: 0.000207"           → value only (no parentheses)
_METRIC_PAIR_RE = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9 ()/_-]*)\s*:\s*"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s*\("
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    r"\))?"
)

# Map BiaPy's human-readable metric labels → stable DataFrame column stems
_KNOWN_METRIC_ALIASES = {
    "loss": "loss",
    "lr": "lr",
    "iou (f channel)": "iou_f",
    "iou (c channel)": "iou_c",
    "l1 (db channel)": "l1_db",
    "l1 (dn channel)": "l1_dn",
}


def _normalize_metric_name(raw: str) -> str:
    """Turn a log metric label into a snake_case column stem.

    1. Lowercase / collapse whitespace so aliases match reliably.
    2. Prefer the known alias table (IoU / L1 channel names).
    3. Otherwise strip non-alphanumerics to underscores for unknown metrics.
    """
    # 1. Normalize casing and spacing
    key = " ".join(raw.strip().lower().split())
    # 2. Use canonical short name when we know this metric
    if key in _KNOWN_METRIC_ALIASES:
        return _KNOWN_METRIC_ALIASES[key]
    # 3. Generic fallback for any future channel / metric BiaPy adds
    cleaned = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return cleaned or "metric"


def _parse_averaged_stats(stats_blob: str) -> dict[str, float]:
    """Parse ``name: last (mean)`` or ``name: value`` into mean/value floats.

    1. Find every ``name: number (optional mean)`` token in the stats blob.
    2. Normalize the name to a column stem.
    3. Prefer the parenthesized epoch mean when present; else use the bare value
       (needed for ``lr``, which has no mean).
    """
    out: dict[str, float] = {}
    # 1–3. Walk all metric tokens in the averaged-stats suffix
    for match in _METRIC_PAIR_RE.finditer(stats_blob):
        name = _normalize_metric_name(match.group(1))
        last = float(match.group(2))
        mean = match.group(3)
        out[name] = float(mean) if mean is not None else last
    return out


def consolidate_training_log(path: str | Path) -> pd.DataFrame:
    """Parse a BiaPy partitioned train ``.out`` into one row per epoch.

    Columns include ``partition``, ``epoch``, ``global_step``, train/val
    metrics, ``lr``, and ``early_stopped``.

    Walk the log once, tracking which partition / epoch is active. A row is
    emitted only when a ``[Val] averaged stats`` line follows a pending
    ``[Train] averaged stats`` for the same epoch.
    """
    path = Path(path)

    # --- parser state -------------------------------------------------------
    rows: list[dict] = []
    partition: int | None = None          # 0-based active partition
    epoch: int | None = None              # epoch within that partition (1-based)
    pending_train: dict[str, float] | None = None  # train stats awaiting val
    early_stopped_partitions: set[int] = set()
    current_partition_early = False

    # 1. Stream the log line-by-line (files can be tens of thousands of lines)
    with path.open() as fh:
        for line in fh:
            # 2. Detect a new partition banner ("Partition N/M: ...")
            part_m = _PARTITION_RE.search(line)
            if part_m:
                # Carry early-stop flag from the partition we are leaving
                if partition is not None and current_partition_early:
                    early_stopped_partitions.add(partition)
                # Log uses 1-based N; store 0-based for indexing
                partition = int(part_m.group(1)) - 1
                epoch = None
                pending_train = None
                current_partition_early = False
            else:
                # 2b. Fallback: infer partition from staging path ".../part_K"
                #     only before any Partition banner has been seen
                part_dir_m = _PART_DIR_RE.search(line)
                if part_dir_m and partition is None:
                    partition = int(part_dir_m.group(1))

            # 3. Track the current epoch number inside this partition
            start_m = _START_TRAIN_RE.search(line)
            if start_m:
                epoch = int(start_m.group(1))

            epoch_m = _EPOCH_RE.search(line)
            if epoch_m:
                epoch = int(epoch_m.group(1))

            # 4. Cache train epoch means; wait for the matching val line
            train_m = _TRAIN_STATS_RE.search(line)
            if train_m:
                pending_train = _parse_averaged_stats(train_m.group(1))
                continue

            # 5. Pair val stats with pending train → one DataFrame row
            val_m = _VAL_STATS_RE.search(line)
            if val_m and pending_train is not None:
                val_stats = _parse_averaged_stats(val_m.group(1))
                # Defensive defaults if markers were missing (malformed log)
                if partition is None:
                    partition = 0
                if epoch is None:
                    epoch = len(rows) + 1

                row: dict = {
                    "partition": partition,
                    "epoch": epoch,
                }
                # Prefix train metrics; keep lr unprefixed (optimizer-wide)
                for key, value in pending_train.items():
                    if key == "lr":
                        row["lr"] = value
                    else:
                        row[f"train_{key}"] = value
                # Prefix val metrics the same way
                for key, value in val_stats.items():
                    if key == "lr":
                        continue
                    row[f"val_{key}"] = value
                rows.append(row)
                pending_train = None
                continue

            # 6. Note early stopping / end-of-partition for this chunk
            if _EARLY_STOP_RE.search(line) and partition is not None:
                current_partition_early = True
                early_stopped_partitions.add(partition)

            if _FINISHED_RE.search(line) and partition is not None:
                if current_partition_early:
                    early_stopped_partitions.add(partition)
                pending_train = None

    # 7. Empty log → schema-only DataFrame so callers can still inspect columns
    if not rows:
        return pd.DataFrame(
            columns=[
                "partition",
                "epoch",
                "global_step",
                "train_loss",
                "val_loss",
                "lr",
                "early_stopped",
            ]
        )

    # 8. Build the frame and add derived columns
    df = pd.DataFrame(rows)
    # Continuous x-axis across partitions (each partition restarts epoch at 1)
    df["global_step"] = range(len(df))
    # True for every row belonging to a partition that early-stopped
    df["early_stopped"] = df["partition"].isin(early_stopped_partitions)

    # 9. Stable column order: identifiers, common metrics, then any extras
    preferred = [
        "partition",
        "epoch",
        "global_step",
        "train_loss",
        "val_loss",
        "train_iou_f",
        "train_iou_c",
        "train_l1_db",
        "train_l1_dn",
        "val_iou_f",
        "val_iou_c",
        "val_l1_db",
        "val_l1_dn",
        "lr",
        "early_stopped",
    ]
    ordered = [c for c in preferred if c in df.columns]
    ordered.extend(c for c in df.columns if c not in ordered)
    return df[ordered]


def plot_training_across_partitions(
    df: pd.DataFrame,
    metrics: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot metrics vs ``global_step`` with vertical partition boundaries.

    1. Validate inputs / default to train+val loss.
    2. Draw each metric as a continuous curve over ``global_step``.
    3. Mark where each new partition begins with a vertical dashed line.
    4. Label partitions sparsely on a secondary top axis.
    5. Optionally write a PNG and return the figure.
    """
    # 1. Guard empty / missing columns
    if df.empty:
        raise ValueError("Cannot plot empty training DataFrame")

    metrics = tuple(metrics) if metrics is not None else ("train_loss", "val_loss")
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise KeyError(f"Metrics not in DataFrame: {missing}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4.5))
    else:
        fig = ax.figure

    # 2. One line per requested metric along the concatenated training axis
    for metric in metrics:
        ax.plot(
            df["global_step"],
            df[metric],
            label=metric,
            linewidth=1.5,
        )

    # 3. Partition boundaries = first global_step of each partition (skip 0)
    starts = (
        df.groupby("partition", sort=True)["global_step"]
        .min()
        .sort_index()
    )
    for step in starts.iloc[1:]:
        ax.axvline(step, color="0.7", linestyle="--", linewidth=0.8, zorder=0)

    # 4. Sparse top ticks so ~27 partitions stay readable
    n_parts = len(starts)
    label_every = max(1, n_parts // 12)
    top = ax.secondary_xaxis("top")
    tick_parts = starts.index[::label_every]
    top.set_xticks([starts[p] for p in tick_parts])
    top.set_xticklabels([str(p) for p in tick_parts])
    top.set_xlabel("partition")

    ax.set_xlabel("global_step")
    ax.set_ylabel("metric")
    ax.set_title(title or "Training across partitions")
    # Put legend outside the axes so it does not cover the curves
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.grid(True, alpha=0.3)
    # Leave room on the right for the external legend
    fig.tight_layout(rect=(0, 0, 0.82, 1))

    # 5. Optional disk write for CLI / notebooks
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig


def _build_argparser() -> argparse.ArgumentParser:
    """CLI: log path in, optional CSV + PNG out."""
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate BiaPy partitioned train .out logs into a DataFrame "
            "and optionally plot / save CSV."
        )
    )
    parser.add_argument(
        "log",
        type=Path,
        help="Path to a SLURM train .out (or BiaPy train log) file",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write the consolidated CSV",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional path to write a PNG of train/val loss across partitions",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics to plot (default: train_loss val_loss)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    """Run consolidate → optional CSV / plot from the command line.

    1. Parse args and build the epoch DataFrame from the log.
    2. Print a short summary (+ head) so the user can sanity-check.
    3. Write CSV and/or PNG when requested.
    """
    # 1. Parse + consolidate
    args = _build_argparser().parse_args(argv)
    df = consolidate_training_log(args.log)

    # 2. Summary for the terminal
    print(
        f"Parsed {len(df)} epoch(s) across "
        f"{df['partition'].nunique() if not df.empty else 0} partition(s) "
        f"from {args.log}"
    )
    if not df.empty:
        print(df.head().to_string(index=False))

    # 3a. Persist table
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"Wrote CSV → {args.csv}")

    # 3b. Persist figure (close so batch / CLI does not leak figures)
    if args.plot is not None:
        fig = plot_training_across_partitions(
            df,
            metrics=args.metrics,
            save_path=args.plot,
            title=args.log.name,
        )
        plt.close(fig)
        print(f"Wrote plot → {args.plot}")


if __name__ == "__main__":
    main()
