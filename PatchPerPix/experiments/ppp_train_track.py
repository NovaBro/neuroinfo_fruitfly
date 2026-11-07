"""Parse and plot PatchPerPix train / validation logs.

Train loss: gunpowder emits one line per iteration::

    INFO:gunpowder.nodes.generic_train:Train process: iteration=N loss=X time=Y

Validation (``setup01/validation.py``) every ``val_log_step`` iterations writes
``<exp>/train/val_metrics.csv`` and a log line::

    INFO:...validation:Validation: iteration=N n_samples=72 loss=... fg_dice=... aff_f1=...

Primary sources are ``metrics/ppp/<exp_id>/run.log`` (appended across SLURM
resumes) and ``metrics/ppp/<exp_id>/train/val_metrics.csv``. TensorBoard
scalars are not assumed present.

Legacy runs (before validation.py): gunpowder's TrainValProvider ran one val
batch at every ``iteration % val_log_step == 2`` and logged it as an ordinary
``Train process`` line. Pass ``legacy_val_log_step`` to drop those spikes from
the train curve and to plot them as a (single-batch, noisy) legacy val curve.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf"

_TRAIN_PROCESS_RE = re.compile(
    rf"Train process: iteration=(\d+) "
    rf"loss=({_NUM})"
    rf"(?: lr=(?:{_NUM}))? "
    rf"time=({_NUM})"
)
_VALIDATION_RE = re.compile(r"Validation: (iteration=\d+(?: \w+=\S+)*)")
_KV_RE = re.compile(r"(\w+)=(\S+)")

VAL_CSV_NAME = "val_metrics.csv"
DEFAULT_VAL_METRICS = ("loss", "fg_dice", "aff_f1", "numinst_acc_fg",
                       "overlap_recall", "loss_fg")


def parse_train_log(
    path: str | Path,
    *,
    legacy_val_log_step: int | None = None,
) -> pd.DataFrame:
    """Parse a PPP ``run.log`` (or train ``.out``) into one row per iteration.

    1. Stream the file and extract ``iteration`` / ``loss`` / ``time``.
    2. Keep the last row when the same iteration appears again (resume appends).
    3. Sort ascending by ``iteration``.
    4. Add ``is_val``. With ``legacy_val_log_step`` set, rows with
       ``iteration % step == 2`` that come before the first ``Validation:``
       line are the old interleaved val batches.

    Empty logs return a schema-only DataFrame.
    """
    path = Path(path)
    rows: list[dict] = []
    first_new_val: int | None = None

    with path.open() as fh:
        for line in fh:
            m = _TRAIN_PROCESS_RE.search(line)
            if m:
                rows.append(
                    {
                        "iteration": int(m.group(1)),
                        "loss": float(m.group(2)),
                        "time": float(m.group(3)),
                    }
                )
                continue
            if first_new_val is None:
                v = _VALIDATION_RE.search(line)
                if v:
                    first_new_val = int(_KV_RE.match(v.group(1)).group(2))

    if not rows:
        return pd.DataFrame(columns=["iteration", "loss", "time", "is_val"])

    df = pd.DataFrame(rows)
    # File order preserved until drop; last duplicate wins after a resume.
    df = df.drop_duplicates(subset=["iteration"], keep="last")
    df = df.sort_values("iteration", kind="mergesort").reset_index(drop=True)
    df["is_val"] = False
    if legacy_val_log_step:
        legacy = df["iteration"] % int(legacy_val_log_step) == 2
        if first_new_val is not None:
            legacy &= df["iteration"] < first_new_val
        df["is_val"] = legacy
    return df


def parse_val_log(path: str | Path) -> pd.DataFrame:
    """``Validation: iteration=N k=v ...`` lines of a run.log -> DataFrame."""
    rows: list[dict] = []
    with Path(path).open() as fh:
        for line in fh:
            m = _VALIDATION_RE.search(line)
            if not m:
                continue
            row = {}
            for k, v in _KV_RE.findall(m.group(1)):
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["iteration"])
    df = pd.DataFrame(rows)
    df["iteration"] = df["iteration"].astype(int)
    return df


def _dedupe_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["iteration"], keep="last")
    return df.sort_values("iteration", kind="mergesort").reset_index(drop=True)


def _exp_dir(path: str | Path) -> Path:
    """Accept ``<exp>``, ``<exp>/run.log`` or ``<exp>/train/val_metrics.csv``."""
    path = Path(path)
    if path.is_dir():
        return path
    if path.name == VAL_CSV_NAME:
        return path.parent.parent
    return path.parent


def load_val_metrics(path: str | Path) -> pd.DataFrame:
    """Validation metrics for one experiment.

    Prefers ``<exp>/train/val_metrics.csv``, else the ``Validation:`` lines
    of ``<exp>/run.log``. Deduped by iteration (last wins: resume re-runs).
    Adds ``source`` = ``csv`` / ``log``. Empty DataFrame if neither exists.
    """
    exp = _exp_dir(path)
    csv_path = exp / "train" / VAL_CSV_NAME
    if csv_path.exists() and csv_path.stat().st_size > 0:
        df = pd.read_csv(csv_path)
        df["source"] = "csv"
        return _dedupe_sort(df)
    log_path = Path(path) if Path(path).is_file() and Path(path).suffix != ".csv" \
        else exp / "run.log"
    if log_path.exists():
        df = parse_val_log(log_path)
        if not df.empty:
            df["source"] = "log"
            return _dedupe_sort(df)
    return pd.DataFrame(columns=["iteration", "source"])


def load_ppp_runs(
    run_map: Mapping[str, str | Path],
    *,
    smooth_window: int = 100,
    legacy_val_log_step: int | None = None,
) -> pd.DataFrame:
    """Parse several runs, tag with ``run``, and add a rolling-mean loss.

    Only train iterations are returned. Legacy interleaved val rows (see
    ``legacy_val_log_step``) are dropped before smoothing; get them from
    :func:`load_ppp_val_runs`. Raw ``loss`` is kept; ``loss_smooth`` is for
    plotting.
    """
    if smooth_window < 1:
        raise ValueError(f"smooth_window must be >= 1, got {smooth_window}")

    frames: list[pd.DataFrame] = []
    for run_name, path in run_map.items():
        df = parse_train_log(path, legacy_val_log_step=legacy_val_log_step)
        df = df[~df["is_val"].astype(bool)].drop(columns="is_val")
        df = df.reset_index(drop=True)
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


def load_ppp_val_runs(
    run_map: Mapping[str, str | Path],
    *,
    legacy_val_log_step: int | None = None,
) -> pd.DataFrame:
    """Validation rows for several runs, tagged with ``run``.

    ``run_map`` values may be run.log paths or experiment dirs. New-style
    metrics come from :func:`load_val_metrics`. With ``legacy_val_log_step``,
    old interleaved val batches from ``run.log`` are added (only ``loss``,
    ``source`` = ``legacy_1batch``) for iterations before the first new-style
    row.
    """
    frames: list[pd.DataFrame] = []
    for run_name, path in run_map.items():
        df = load_val_metrics(path)
        if legacy_val_log_step:
            log_path = Path(path) if Path(path).is_file() else Path(path) / "run.log"
            if log_path.exists():
                tr = parse_train_log(log_path, legacy_val_log_step=legacy_val_log_step)
                leg = tr.loc[tr["is_val"].astype(bool), ["iteration", "loss"]].copy()
                if not df.empty:
                    leg = leg[leg["iteration"] < df["iteration"].min()]
                leg["source"] = "legacy_1batch"
                df = pd.concat([leg, df], ignore_index=True) if not df.empty else leg
        df = df.copy()
        df.insert(0, "run", run_name)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["run", "iteration", "source"])
    out = pd.concat(frames, ignore_index=True)
    out["iteration"] = out["iteration"].astype(int)
    return out


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


def plot_val_metrics(
    val_df: pd.DataFrame,
    *,
    train_df: pd.DataFrame | None = None,
    metrics: Sequence[str] = DEFAULT_VAL_METRICS,
    ncols: int = 3,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Small multiples of validation metrics vs iteration, one line per run.

    On the ``loss`` panel the smoothed train loss from ``train_df`` (output of
    :func:`load_ppp_runs`) is drawn dashed in the same colour. Legacy
    single-batch val points (``source == legacy_1batch``) are drawn as faint
    markers. Metrics missing from ``val_df`` (or all-NaN) are skipped.
    """
    if val_df.empty:
        raise ValueError("Cannot plot empty validation DataFrame")
    metrics = [m for m in metrics
               if m in val_df.columns and val_df[m].notna().any()]
    if not metrics:
        raise ValueError("None of the requested metrics are in val_df")

    ncols = max(1, min(ncols, len(metrics)))
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows),
                             squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    runs = list(dict.fromkeys(val_df["run"]))
    if train_df is not None and not train_df.empty:
        runs += [r for r in dict.fromkeys(train_df["run"]) if r not in runs]
    color_of = {r: colors[i % len(colors)] for i, r in enumerate(runs)}
    source = val_df["source"] if "source" in val_df.columns else \
        pd.Series("csv", index=val_df.index)

    for ax, metric in zip(axes.flat, metrics):
        for run_name, group in val_df.groupby("run", sort=False):
            src = source.loc[group.index]
            new = group[(src != "legacy_1batch") & group[metric].notna()]
            leg = group[(src == "legacy_1batch") & group[metric].notna()]
            c = color_of[run_name]
            if not leg.empty:
                ax.plot(leg["iteration"], leg[metric], "o", ms=2.5, alpha=0.35,
                        color=c, label=f"{run_name} (legacy 1-batch)")
            if not new.empty:
                ax.plot(new["iteration"], new[metric], "-o", ms=3, lw=1.5,
                        color=c, label=f"{run_name} val")
        if metric == "loss" and train_df is not None and not train_df.empty:
            y = "loss_smooth" if "loss_smooth" in train_df.columns else "loss"
            for run_name, group in train_df.groupby("run", sort=False):
                ax.plot(group["iteration"], group[y], "--", lw=1, alpha=0.8,
                        color=color_of.get(run_name), label=f"{run_name} train")
        ax.set_title(metric)
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        axes.flat[0].legend(handles, labels, fontsize=7, loc="best")
    fig.suptitle(title or "PPP validation")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse PatchPerPix run.log train lines (and validation metrics) "
            "into DataFrames and optionally plot / save CSV."
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
    parser.add_argument(
        "--legacy-val-log-step",
        type=int,
        default=None,
        help=("val_log_step of runs from before validation.py: drops the "
              "interleaved val batches (iteration %% step == 2) from the "
              "train curve and reports them as legacy val points"),
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Also load validation metrics (train/val_metrics.csv or Validation: lines)",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=None,
        help="With --val: write the consolidated validation CSV here",
    )
    parser.add_argument(
        "--val-plot",
        type=Path,
        default=None,
        help="With --val: write a PNG of validation metrics here",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    run_map = {path.parent.name: path for path in args.logs}
    df = load_ppp_runs(run_map, smooth_window=args.smooth_window,
                       legacy_val_log_step=args.legacy_val_log_step)

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

    if args.val or args.val_csv or args.val_plot:
        vdf = load_ppp_val_runs(run_map,
                                legacy_val_log_step=args.legacy_val_log_step)
        for run_name, group in vdf.groupby("run", sort=False):
            srcs = ", ".join(f"{k}={v}" for k, v in
                             group["source"].value_counts().items())
            print(f"{run_name}: {len(group)} val points ({srcs})")
        if not vdf.empty:
            print(vdf.tail().to_string(index=False))
        if args.val_csv is not None:
            args.val_csv.parent.mkdir(parents=True, exist_ok=True)
            vdf.to_csv(args.val_csv, index=False)
            print(f"Wrote val CSV → {args.val_csv}")
        if args.val_plot is not None and not vdf.empty:
            fig = plot_val_metrics(vdf, train_df=df, save_path=args.val_plot)
            plt.close(fig)
            print(f"Wrote val plot → {args.val_plot}")


if __name__ == "__main__":
    main()
