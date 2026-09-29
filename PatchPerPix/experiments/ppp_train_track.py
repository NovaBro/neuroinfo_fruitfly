"""Parse and plot PatchPerPix train logs from experiment ``run.log`` files.

Gunpowder emits one line per iteration::

    INFO:gunpowder.nodes.generic_train:Train process: iteration=N loss=X time=Y

Primary sources are ``metrics/ppp/<exp_id>/run.log`` (appended across SLURM
resumes). TensorBoard scalars are not assumed present.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import pandas as pd

_TRAIN_PROCESS_RE = re.compile(
    r"Train process: iteration=(\d+) "
    r"loss=([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?) "
    r"time=([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)


def parse_train_log(path: str | Path) -> pd.DataFrame:
    """Parse a PPP ``run.log`` (or train ``.out``) into one row per iteration.

    1. Stream the file and extract ``iteration`` / ``loss`` / ``time``.
    2. Keep the last row when the same iteration appears again (resume appends).
    3. Sort ascending by ``iteration``.

    Empty logs return a schema-only DataFrame.
    """
    path = Path(path)
    rows: list[dict] = []

    with path.open() as fh:
        for line in fh:
            m = _TRAIN_PROCESS_RE.search(line)
            if not m:
                continue
            rows.append(
                {
                    "iteration": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "time": float(m.group(3)),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["iteration", "loss", "time"])

    df = pd.DataFrame(rows)
    # File order preserved until drop; last duplicate wins after a resume.
    df = df.drop_duplicates(subset=["iteration"], keep="last")
    return df.sort_values("iteration", kind="mergesort").reset_index(drop=True)


def load_ppp_runs(
    run_map: Mapping[str, str | Path],
    *,
    smooth_window: int = 100,
) -> pd.DataFrame:
    """Parse several runs, tag with ``run``, and add a rolling-mean loss.

    ``smooth_window`` matches PPP ``len_epoch`` / ``val_log_step`` defaults.
    Raw ``loss`` is kept; ``loss_smooth`` is for plotting.
    """
    if smooth_window < 1:
        raise ValueError(f"smooth_window must be >= 1, got {smooth_window}")

    frames: list[pd.DataFrame] = []
    for run_name, path in run_map.items():
        df = parse_train_log(path)
        df.insert(0, "run", run_name)
        if df.empty:
            df["loss_smooth"] = pd.Series(dtype=float)
        else:
            df["loss_smooth"] = (
                df["loss"]
                .rolling(window=smooth_window, min_periods=1, center=True)
                .mean()
            )
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=["run", "iteration", "loss", "time", "loss_smooth"]
        )
    return pd.concat(frames, ignore_index=True)


def plot_train_loss(
    df: pd.DataFrame,
    *,
    use_smooth: bool = True,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Overlay loss vs iteration for each ``run`` in ``df``."""
    if df.empty:
        raise ValueError("Cannot plot empty training DataFrame")

    y_col = "loss_smooth" if use_smooth else "loss"
    if y_col not in df.columns:
        raise KeyError(f"Column not in DataFrame: {y_col}")
    if "run" not in df.columns or "iteration" not in df.columns:
        raise KeyError("DataFrame needs 'run' and 'iteration' columns")

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.5))

    for run_name, group in df.groupby("run", sort=False):
        ax.plot(
            group["iteration"],
            group[y_col],
            label=run_name,
            linewidth=1.5,
        )

    ax.set_xlabel("iteration")
    ax.set_ylabel("loss" + (" (smoothed)" if use_smooth else ""))
    ax.set_title(title or "PPP train loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150)

    return ax


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse PatchPerPix run.log train lines into a DataFrame "
            "and optionally plot / save CSV."
        )
    )
    parser.add_argument(
        "logs",
        nargs="+",
        type=Path,
        help="One or more run.log (or train .out) paths; label = parent dir name",
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
        help="Optional path to write a PNG of smoothed train loss",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=100,
        help="Rolling-mean window for loss_smooth (default: 100)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    run_map = {path.parent.name: path for path in args.logs}
    df = load_ppp_runs(run_map, smooth_window=args.smooth_window)

    for run_name, group in df.groupby("run", sort=False):
        if group.empty:
            print(f"{run_name}: 0 rows")
            continue
        print(
            f"{run_name}: {len(group)} iters "
            f"[{int(group['iteration'].min())}–{int(group['iteration'].max())}]"
        )
    if not df.empty:
        print(df.head().to_string(index=False))

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"Wrote CSV → {args.csv}")

    if args.plot is not None:
        ax = plot_train_loss(df, use_smooth=True, save_path=args.plot)
        plt.close(ax.figure)
        print(f"Wrote plot → {args.plot}")


if __name__ == "__main__":
    main()
