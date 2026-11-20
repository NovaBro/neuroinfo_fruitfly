"""Parse and plot PatchPerPix train / validation logs.

Train loss: gunpowder emits one line per iteration::

    INFO:gunpowder.nodes.generic_train:Train process: iteration=N loss=X time=Y

With an lr scheduler, gunpowder's torch_train node also logs the lr it just
used (``get_last_lr()`` before ``lr_scheduler.step()``) on the line *before*
each ``Train process`` line::

    INFO:gunpowder.nodes.torch_train:stepping lr scheduler, current lr: [3e-06]

(newer numpy prints ``[np.float64(3e-06)]``). That lr belongs to the next
``Train process`` iteration and is parsed into the ``lr`` column.

Validation (``setup01/validation.py``) every ``val_log_step`` iterations writes
``<exp>/train/val_metrics.csv`` and a log line::

    INFO:...validation:Validation: iteration=N n_samples=72 loss=... fg_dice=... aff_f1=...

Primary sources are ``metrics/ppp/<exp_id>/run.log`` (appended across SLURM
resumes) and ``metrics/ppp/<exp_id>/train/val_metrics.csv``. TensorBoard
scalars are not assumed present.

Extra val sets (``[data] extra_val_datasets``, e.g. ``completely`` next to a
synthetic main val set) write ``<exp>/train/val_metrics_<name>.csv`` and
``Validation[<name>]: iteration=N ...`` lines. Val rows carry ``val_set`` (the
dataset name; the main set is named from ``[data] val_datasets`` in
``config.toml``) and ``val_role`` (``main`` / ``extra``). Checkpoint selection
(lowest val loss, the star in the plots) uses the main set only, as the
expv2 / opexp / dsexp drivers do.

Legacy runs (before validation.py): gunpowder's TrainValProvider ran one val
batch at every ``iteration % val_log_step == 2`` and logged it as an ordinary
``Train process`` line. Pass ``legacy_val_log_step`` to drop those spikes from
the train curve and to plot them as a (single-batch, noisy) legacy val curve.
Only lines logged before validation.py's setup line (``validation: N batches x
M samples every K its``) can be legacy; after it, iteration 2 etc. are ordinary
train steps.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf"

_TRAIN_PROCESS_RE = re.compile(
    rf"Train process: iteration=(\d+) "
    rf"loss=({_NUM})"
    rf"(?: lr=(?:{_NUM}))? "
    rf"time=({_NUM})"
)
# "Validation: ..." (main val set) or "Validation[<name>]: ..." (extra val set).
_VALIDATION_RE = re.compile(r"Validation(?:\[([^\]]+)\])?: (iteration=\d+(?: \w+=\S+)*)")
# torch_train: lr used for the step whose "Train process" line follows.
_LR_RE = re.compile(rf"stepping lr scheduler, current lr: \[(?:np\.float\d*\()?({_NUM})")
# validation.py setup line, logged once per (re)start before the first train step.
_VAL_SETUP_RE = re.compile(r"validation: \d+ batches x \d+ samples every \d+ its")
_KV_RE = re.compile(r"(\w+)=(\S+)")

VAL_CSV_NAME = "val_metrics.csv"
EXTRA_VAL_CSV_PREFIX = "val_metrics_"  # val_metrics_<name>.csv
# Line style per val set in the plots (by order of first appearance).
_VAL_SET_STYLES = (dict(ls="-", marker="o"), dict(ls="--", marker="s"),
                   dict(ls="-.", marker="^"), dict(ls=":", marker="D"))
DEFAULT_VAL_METRICS = ("loss", "fg_dice", "aff_f1", "numinst_acc_fg",
                       "overlap_recall", "loss_fg")
# Losses + the basic segmentation metrics of val_metrics.csv (3 x 3 grid).
BASIC_VAL_METRICS = ("loss", "loss_patch", "loss_fg",
                     "fg_dice", "fg_precision", "fg_recall",
                     "aff_f1", "numinst_acc_fg", "overlap_recall")


def parse_train_log(
    path: str | Path,
    *,
    legacy_val_log_step: int | None = None,
) -> pd.DataFrame:
    """Parse a PPP ``run.log`` (or train ``.out``) into one row per iteration.

    1. Stream the file and extract ``iteration`` / ``loss`` / ``time``, and
       ``lr`` from the preceding ``stepping lr scheduler`` line (NaN when the
       run has no lr scheduler, so nothing is logged).
    2. Keep the last row when the same iteration appears again (resume appends).
    3. Sort ascending by ``iteration``.
    4. Add ``is_val``. With ``legacy_val_log_step`` set, rows with
       ``iteration % step == 2`` logged before validation.py's setup line are
       the old interleaved val batches (fallback for logs without that line:
       before the first ``Validation:`` iteration).

    Empty logs return a schema-only DataFrame.
    """
    path = Path(path)
    rows: list[dict] = []
    first_new_val: int | None = None
    new_val_on = False  # validation.py active for the lines that follow
    pending_lr = float("nan")  # lr logged just before the next Train process line

    with path.open() as fh:
        for line in fh:
            m = _TRAIN_PROCESS_RE.search(line)
            if m:
                rows.append(
                    {
                        "iteration": int(m.group(1)),
                        "loss": float(m.group(2)),
                        "time": float(m.group(3)),
                        "lr": pending_lr,
                        "_new_val": new_val_on,
                    }
                )
                pending_lr = float("nan")
                continue
            lr_m = _LR_RE.search(line)
            if lr_m:
                pending_lr = float(lr_m.group(1))
                continue
            if not new_val_on and _VAL_SETUP_RE.search(line):
                new_val_on = True
                continue
            if first_new_val is None:
                v = _VALIDATION_RE.search(line)
                if v:
                    first_new_val = int(_KV_RE.match(v.group(2)).group(2))

    if not rows:
        return pd.DataFrame(columns=["iteration", "loss", "time", "lr", "is_val"])

    df = pd.DataFrame(rows)
    # File order preserved until drop; last duplicate wins after a resume.
    df = df.drop_duplicates(subset=["iteration"], keep="last")
    df = df.sort_values("iteration", kind="mergesort").reset_index(drop=True)
    df["is_val"] = False
    if legacy_val_log_step:
        legacy = df["iteration"] % int(legacy_val_log_step) == 2
        if new_val_on:
            legacy &= ~df["_new_val"]
        elif first_new_val is not None:
            legacy &= df["iteration"] < first_new_val
        df["is_val"] = legacy
    return df.drop(columns="_new_val")


def parse_val_log(path: str | Path) -> pd.DataFrame:
    """``Validation[<name>]: iteration=N k=v ...`` lines of a run.log -> DataFrame.

    ``val_set`` is the bracketed extra-val-set name, None for the main set
    (plain ``Validation:`` lines).
    """
    rows: list[dict] = []
    with Path(path).open() as fh:
        for line in fh:
            if "Validation" not in line:
                continue
            m = _VALIDATION_RE.search(line)
            if not m:
                continue
            row = {"val_set": m.group(1)}
            for k, v in _KV_RE.findall(m.group(2)):
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["iteration", "val_set"])
    df = pd.DataFrame(rows)
    df["iteration"] = df["iteration"].astype(int)
    return df


def _dedupe_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["iteration"], keep="last")
    return df.sort_values("iteration", kind="mergesort").reset_index(drop=True)


def _exp_dir(path: str | Path) -> Path:
    """Accept ``<exp>``, ``<exp>/run.log`` or ``<exp>/train/val_metrics*.csv``."""
    path = Path(path)
    if path.is_dir():
        return path
    if path.suffix == ".csv" and path.name.startswith("val_metrics"):
        return path.parent.parent
    return path.parent


def _val_set_names(exp: Path) -> tuple[str, list[str]]:
    """(main val set name, configured extra val sets) from ``<exp>/config.toml``.

    The main name is ``[data] val_datasets`` joined with ``+`` (``"val"`` if
    the config is missing or has none).
    """
    data: Mapping = {}
    cfg_path = exp / "config.toml"
    if cfg_path.exists():
        try:
            data = _load_toml(cfg_path).get("data") or {}
        except Exception:  # only used for labels
            data = {}

    def as_list(v) -> list[str]:
        return [v] if isinstance(v, str) else [str(x) for x in (v or [])]

    main = "+".join(as_list(data.get("val_datasets"))) or "val"
    return main, as_list(data.get("extra_val_datasets"))


def _nonempty(path: Path | None) -> bool:
    return path is not None and path.exists() and path.stat().st_size > 0


def load_val_metrics(path: str | Path) -> pd.DataFrame:
    """Validation metrics of every val set of one experiment.

    Per val set, prefers its CSV (``<exp>/train/val_metrics.csv`` for the main
    set, ``val_metrics_<name>.csv`` for an extra set), else its
    ``Validation[...]:`` lines in ``<exp>/run.log``. Deduped by iteration per
    set (last wins: resume re-runs). Adds ``source`` (``csv`` / ``log``),
    ``val_set`` (dataset name; main set from config.toml) and ``val_role``
    (``main`` / ``extra``); main set rows first. Empty DataFrame if nothing
    is found.
    """
    exp = _exp_dir(path)
    main_name, cfg_extras = _val_set_names(exp)
    csvs: dict[str | None, Path] = {None: exp / "train" / VAL_CSV_NAME}
    for p in sorted((exp / "train").glob(f"{EXTRA_VAL_CSV_PREFIX}*.csv")):
        csvs[p.stem[len(EXTRA_VAL_CSV_PREFIX):]] = p
    names: list[str | None] = [None] + list(dict.fromkeys(
        cfg_extras + [n for n in csvs if n is not None]))

    p = Path(path)
    log_path = p if p.is_file() and p.suffix != ".csv" else exp / "run.log"
    log_df = None
    if any(not _nonempty(csvs.get(n)) for n in names) and log_path.exists():
        log_df = parse_val_log(log_path)
        names += [n for n in dict.fromkeys(log_df["val_set"].dropna()) if n not in names]

    frames = []
    for name in names:
        if _nonempty(csvs.get(name)):
            df = pd.read_csv(csvs[name])
            df["source"] = "csv"
        elif log_df is not None:
            sel = log_df["val_set"].isna() if name is None else log_df["val_set"] == name
            df = log_df[sel].drop(columns="val_set")
            df["source"] = "log"
        else:
            continue
        if df.empty:
            continue
        df = _dedupe_sort(df)
        df["val_set"] = main_name if name is None else name
        df["val_role"] = "main" if name is None else "extra"
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["iteration", "source", "val_set", "val_role"])
    return pd.concat(frames, ignore_index=True)


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
            columns=["run", "iteration", "loss", "time", "lr", "loss_smooth"]
        )
    return pd.concat(frames, ignore_index=True)


def load_ppp_val_runs(
    run_map: Mapping[str, str | Path],
    *,
    legacy_val_log_step: int | None = None,
) -> pd.DataFrame:
    """Validation rows for several runs, tagged with ``run``.

    ``run_map`` values may be run.log paths or experiment dirs. New-style
    metrics come from :func:`load_val_metrics` (all val sets, see
    ``val_set`` / ``val_role``). With ``legacy_val_log_step``, old
    interleaved val batches from ``run.log`` are added to the main set (only
    ``loss``, ``source`` = ``legacy_1batch``) for iterations before its first
    new-style row.
    """
    frames: list[pd.DataFrame] = []
    for run_name, path in run_map.items():
        df = load_val_metrics(path)
        if legacy_val_log_step:
            log_path = Path(path) if Path(path).is_file() else Path(path) / "run.log"
            if log_path.exists():
                tr = parse_train_log(log_path, legacy_val_log_step=legacy_val_log_step)
                leg = tr.loc[tr["is_val"].astype(bool), ["iteration", "loss"]].copy()
                main = df[df["val_role"] == "main"]
                if not main.empty:
                    leg = leg[leg["iteration"] < main["iteration"].min()]
                leg["source"] = "legacy_1batch"
                leg["val_set"] = _val_set_names(_exp_dir(path))[0]
                leg["val_role"] = "main"
                df = pd.concat([leg, df], ignore_index=True) if not df.empty else leg
        df = df.copy()
        df.insert(0, "run", run_name)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["run", "iteration", "source", "val_set", "val_role"])
    out = pd.concat(frames, ignore_index=True)
    out["iteration"] = out["iteration"].astype(int)
    return out


def _new_val_rows(val_df: pd.DataFrame) -> pd.DataFrame:
    """validation.py rows only (drops ``legacy_1batch`` points)."""
    if "source" not in val_df.columns:
        return val_df
    return val_df[val_df["source"] != "legacy_1batch"]


def _main_val_rows(val_df: pd.DataFrame) -> pd.DataFrame:
    """Rows of each run's main val set (the checkpoint-selection set)."""
    if "val_role" not in val_df.columns:
        return val_df
    return val_df[val_df["val_role"] == "main"]


def _val_groups(val_df: pd.DataFrame):
    """Yield ``(run, val_set, rows)``; ``val_set`` is None if untagged."""
    if "val_set" not in val_df.columns:
        for run_name, group in val_df.groupby("run", sort=False):
            yield run_name, None, group
        return
    for (run_name, vset), group in val_df.groupby(["run", "val_set"], sort=False):
        yield run_name, vset, group


def _val_set_styles(val_df: pd.DataFrame) -> dict[str, dict]:
    """val set -> line style kwargs (main sets first, then by first appearance)."""
    if "val_set" not in val_df.columns:
        return {}
    sets = list(dict.fromkeys(list(_main_val_rows(val_df)["val_set"])
                              + list(val_df["val_set"])))
    return {s: _VAL_SET_STYLES[i % len(_VAL_SET_STYLES)] for i, s in enumerate(sets)}


def _val_label(run_name: str, vset: str | None) -> str:
    return f"{run_name} val" + (f" [{vset}]" if vset else "")


def _legend_by_run(handles, labels, run_of: Mapping[str, str], runs: Sequence[str]):
    """Order legend entries by run (stable within a run)."""
    order = {r: i for i, r in enumerate(runs)}
    pairs = sorted(zip(handles, labels),
                   key=lambda hl: order.get(run_of.get(hl[1]), len(order)))
    return [h for h, _ in pairs], [l for _, l in pairs]


def best_val_iterations(val_df: pd.DataFrame, by: str = "loss") -> dict[str, int]:
    """run -> iteration with the lowest ``by`` on the run's main val set
    (validation.py rows only; extra val sets are ignored).

    Same rule the expv2 / opexp / dsexp drivers use to pick the checkpoint to
    test.
    """
    out = {}
    for run_name, group in _new_val_rows(_main_val_rows(val_df)).groupby("run", sort=False):
        g = group[group[by].notna()] if by in group.columns else group.iloc[0:0]
        if not g.empty:
            out[run_name] = int(g.loc[g[by].idxmin(), "iteration"])
    return out


def val_summary(
    val_df: pd.DataFrame,
    *,
    metrics: Sequence[str] = BASIC_VAL_METRICS,
    by: str = "loss",
) -> pd.DataFrame:
    """One row per (run, val set), main set first. validation.py rows only.

    ``best_iter`` is the selected checkpoint (lowest ``by`` on the run's
    *main* val set) and ``best_<metric>`` this set's metrics at that
    iteration, so an extra set's row shows how the selected checkpoint does
    on it. ``own_best_iter`` is this set's own lowest-``by`` iteration
    (= ``best_iter`` on the main set); ``last_<by>`` the last val point's.
    """
    new = _new_val_rows(val_df)
    best = best_val_iterations(val_df, by=by)
    rows = []
    for run_name, vset, group in _val_groups(new):
        if run_name not in best:
            continue
        at = group[group["iteration"] == best[run_name]]
        g = group[group[by].notna()] if by in group.columns else group.iloc[0:0]
        last = group.iloc[-1]
        row = {"run": run_name, "val_set": vset or "val",
               "val_role": group["val_role"].iloc[0] if "val_role" in group.columns else "main",
               "n_val": len(group),
               "iters": f"{int(group['iteration'].min())}–{int(group['iteration'].max())}",
               "best_iter": best[run_name]}
        for m in metrics:
            if m in group.columns and group[m].notna().any():
                row[f"best_{m}"] = at[m].iloc[-1] if not at.empty else float("nan")
        row["own_best_iter"] = int(g.loc[g[by].idxmin(), "iteration"]) if not g.empty else None
        row["last_iter"] = int(last["iteration"])
        row[f"last_{by}"] = last.get(by)
        rows.append(row)
    return pd.DataFrame(rows).set_index(["run", "val_set"]) if rows else pd.DataFrame()


def plot_train_loss(
    df: pd.DataFrame,
    *,
    use_smooth: bool = True,
    val_df: pd.DataFrame | None = None,
    logy: bool = False,
    mark_best: bool = True,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Overlay loss vs iteration for each ``run`` in ``df``.

    With ``val_df`` (from :func:`load_ppp_val_runs`), the validation.py val
    loss of every val set is drawn on top in the run's colour, one line style
    per val set (legend: ``<run> val [<set>]``), with a star at the selected
    checkpoint (lowest main-val loss) on each set's curve (``mark_best=False``
    turns the stars off).
    """
    if df.empty:
        raise ValueError("Cannot plot empty training DataFrame")

    y_col = "loss_smooth" if use_smooth else "loss"
    if y_col not in df.columns:
        raise KeyError(f"Column not in DataFrame: {y_col}")
    if "run" not in df.columns or "iteration" not in df.columns:
        raise KeyError("DataFrame needs 'run' and 'iteration' columns")

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.5))

    color_of: dict[str, str] = {}
    run_of: dict[str, str] = {}  # legend label -> run
    for run_name, group in df.groupby("run", sort=False):
        label = f"{run_name} train" if val_df is not None else run_name
        (line,) = ax.plot(
            group["iteration"],
            group[y_col],
            label=label,
            linewidth=1.5 if val_df is None else 1.0,
            alpha=1.0 if val_df is None else 0.6,
        )
        color_of[run_name] = line.get_color()
        run_of[label] = run_name

    if val_df is not None and not val_df.empty and "loss" in val_df.columns:
        best = best_val_iterations(val_df) if mark_best else {}
        new = _new_val_rows(val_df)
        styles = _val_set_styles(new)
        for run_name, vset, group in _val_groups(new):
            g = group[group["loss"].notna()]
            if g.empty:
                continue
            kw = {"color": color_of[run_name]} if run_name in color_of else {}
            label = _val_label(run_name, vset)
            (line,) = ax.plot(g["iteration"], g["loss"], ms=4, lw=1.8, label=label,
                              **styles.get(vset, _VAL_SET_STYLES[0]), **kw)
            color_of.setdefault(run_name, line.get_color())
            run_of[label] = run_name
            if run_name in best:
                b = g[g["iteration"] == best[run_name]]
                ax.plot(b["iteration"], b["loss"], "*", ms=14, mec="k", mew=0.8,
                        color=line.get_color())

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss" + (" (smoothed)" if use_smooth else ""))
    ax.set_title(title or ("PPP train / val loss"
                           + ("  (★ = selected ckpt: lowest main-val loss)" if mark_best else "")
                           if val_df is not None else "PPP train loss"))
    if val_df is not None:  # several entries per run: keep the curves uncovered
        handles, labels = _legend_by_run(*ax.get_legend_handles_labels(), run_of,
                                         list(color_of))
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    else:
        ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_val_metrics(
    val_df: pd.DataFrame,
    *,
    train_df: pd.DataFrame | None = None,
    metrics: Sequence[str] = DEFAULT_VAL_METRICS,
    ncols: int = 3,
    logy_loss: bool = True,
    mark_best: bool = True,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Small multiples of validation metrics vs iteration.

    Colour = run, line style = val set (e.g. solid ``1k_v1.1``, dashed
    ``completely``); one shared legend right of the panels. On the ``loss``
    panel the smoothed train loss from ``train_df`` (output of
    :func:`load_ppp_runs`) is drawn as a thin faint line in the run's colour.
    Legacy single-batch val points (``source == legacy_1batch``) are drawn as
    faint markers. Metrics missing from ``val_df`` (or all-NaN) are skipped.
    ``logy_loss``: log y-axis on ``loss*`` panels (val losses of different
    sets / pretrain vs finetune differ ~10x). ``mark_best``: star the selected
    checkpoint (lowest main-val loss) on every curve of every panel.
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
    best = best_val_iterations(val_df) if mark_best and "loss" in val_df.columns else {}
    styles = _val_set_styles(val_df)

    run_of: dict[str, str] = {}  # legend label -> run
    entries: dict[str, object] = {}  # legend label -> handle (first panel that has it)
    for ax, metric in zip(axes.flat, metrics):
        for run_name, vset, group in _val_groups(val_df):
            src = source.loc[group.index]
            new = group[(src != "legacy_1batch") & group[metric].notna()]
            leg = group[(src == "legacy_1batch") & group[metric].notna()]
            c = color_of[run_name]
            if not leg.empty:
                label = f"{run_name} (legacy 1-batch)"
                h, = ax.plot(leg["iteration"], leg[metric], "o", ms=2.5, alpha=0.35,
                             color=c, label=label)
                entries.setdefault(label, h)
                run_of[label] = run_name
            if not new.empty:
                label = _val_label(run_name, vset)
                h, = ax.plot(new["iteration"], new[metric], ms=3.5, lw=1.5, color=c,
                             label=label, **styles.get(vset, _VAL_SET_STYLES[0]))
                entries.setdefault(label, h)
                run_of[label] = run_name
                b = new[new["iteration"] == best.get(run_name)]
                if not b.empty:
                    ax.plot(b["iteration"], b[metric], "*", ms=12, mec="k",
                            mew=0.7, color=c)
        if metric == "loss" and train_df is not None and not train_df.empty:
            y = "loss_smooth" if "loss_smooth" in train_df.columns else "loss"
            for run_name, group in train_df.groupby("run", sort=False):
                label = f"{run_name} train" + (" (smoothed)" if y == "loss_smooth" else "")
                h, = ax.plot(group["iteration"], group[y], "-", lw=0.9, alpha=0.45,
                             color=color_of.get(run_name), label=label)
                entries.setdefault(label, h)
                run_of[label] = run_name
        if logy_loss and metric.startswith("loss"):
            ax.set_yscale("log")
        ax.set_title(metric + (" (log)" if logy_loss and metric.startswith("loss") else ""))
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle((title or "PPP validation")
                 + ("  (★ = selected ckpt: lowest main-val loss)" if best else ""))
    fig.tight_layout()
    if entries:
        handles, labels = _legend_by_run(list(entries.values()), list(entries), run_of, runs)
        fig.legend(handles, labels, fontsize=8, loc="upper left",
                   bbox_to_anchor=(1.0, 0.95), borderaxespad=0.0)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------------- learning rate
# Primary source: the lr logged per step in run.log (``lr`` column of
# parse_train_log). Runs without an lr scheduler log nothing; for those, and for
# the not-yet-trained part of a run, the lr is rebuilt from config.toml with the
# same formula as setup01/train.py. The rebuild is exact unless the config
# changed mid-run (a resume with a new max_iterations / lr reshapes it).

def _load_toml(path: Path) -> dict:
    try:
        import tomllib
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except ModuleNotFoundError:  # Python < 3.11
        import toml
        return toml.load(path)


def lr_schedule_from_config(cfg: Mapping) -> tuple[pd.DataFrame, dict]:
    """lr for every planned iteration of a run, from its merged config.toml.

    Mirrors setup01/train.py: ``[optimizer.optimizer] lr`` is the peak;
    ``[optimizer.lr_scheduler] lr_scheduler = "WarmupCosine"`` is
    ``make_warmup_cosine`` (linear warmup to the peak over ``warmup_iters`` or
    ``warmup_frac`` of the schedule, then cosine to ``min_lr_ratio`` * peak;
    counted from ``start_iter``); no scheduler = constant lr. Iteration i
    (1-based, as in run.log) uses factor(i - 1). Other torch schedulers are
    not rebuilt (lr = NaN, see ``info["note"]``).

    Returns (DataFrame[iteration, lr], info dict).
    """
    opt = (cfg.get("optimizer") or {}).get("optimizer") or {}
    sc = (cfg.get("optimizer") or {}).get("lr_scheduler") or {}
    peak = float(opt.get("lr", float("nan")))
    end = int((cfg.get("training") or {}).get("max_iterations", 0))
    name = sc.get("lr_scheduler")
    start = int(sc.get("start_iter", 0))
    its = pd.RangeIndex(start + 1, end + 1)
    info = {"scheduler": name or "constant", "peak_lr": peak, "start_iter": start,
            "end_iter": end, "note": ""}
    if not name:
        lr = pd.Series(peak, index=its, dtype=float)
    elif name == "WarmupCosine":
        total = int(sc.get("total_iters", end - start))
        warm = int(sc["warmup_iters"]) if "warmup_iters" in sc else \
            int(round(float(sc.get("warmup_frac", 0.05)) * total))
        min_ratio = float(sc.get("min_lr_ratio", 0.0))
        warm, total = max(0, warm), max(warm + 1, total)
        step = pd.Series(its - 1 - start, index=its, dtype=float).clip(lower=0)
        prog = ((step - warm) / (total - warm)).clip(0.0, 1.0)
        cos = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + np.cos(np.pi * prog))
        factor = cos.where(step >= warm, (step + 1) / max(warm, 1))
        lr = peak * factor
        info.update(warmup_iters=warm, total_iters=total, min_lr_ratio=min_ratio)
    else:
        lr = pd.Series(float("nan"), index=its, dtype=float)
        info["note"] = f"{name}: not reconstructed (torch scheduler)"
    return pd.DataFrame({"iteration": its, "lr": lr.to_numpy()}), info


def load_ppp_lr_runs(
    run_map: Mapping[str, str | Path],
    *,
    train_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """lr per iteration for several runs.

    Trained iterations use the lr logged in run.log (``source = "log"``); runs
    without logged lr, and planned iterations not trained yet, use the
    config.toml rebuild (``source = "config"``). ``lr_config`` keeps the rebuild
    for every row so the two can be compared.

    Returns ``(lr_df, info_df)``: ``lr_df`` has run / iteration / lr /
    lr_config / source / ``trained``; ``info_df`` has one row per run
    (scheduler, peak lr, warmup, range, lr source, max |log - config| / peak).
    ``train_df`` is accepted for backward compatibility and not needed.
    """
    frames, infos = [], []
    for run_name, path in run_map.items():
        exp = _exp_dir(path)
        log_path = Path(path) if Path(path).is_file() else exp / "run.log"
        logged = pd.DataFrame(columns=["iteration", "lr"])
        if log_path.exists():
            tr = parse_train_log(log_path)
            logged = tr.loc[tr["lr"].notna(), ["iteration", "lr"]]
        cfg_path = exp / "config.toml"
        if cfg_path.exists():
            planned, info = lr_schedule_from_config(_load_toml(cfg_path))
        else:
            planned = pd.DataFrame(columns=["iteration", "lr"])
            info = {"note": f"no {cfg_path}"}
        if logged.empty and planned.empty:
            infos.append({"run": run_name, **info, "lr_source": "none"})
            continue
        df = planned.rename(columns={"lr": "lr_config"}).merge(
            logged, on="iteration", how="outer").sort_values("iteration")
        df["source"] = "log"
        no_log = df["lr"].isna()
        df.loc[no_log, "source"] = "config"
        df.loc[no_log, "lr"] = df.loc[no_log, "lr_config"]
        last = int(logged["iteration"].max()) if not logged.empty else None
        if last is None and log_path.exists():  # no scheduler: trained range from loss lines
            tr = parse_train_log(log_path)
            last = int(tr["iteration"].max()) if not tr.empty else None
        df["trained"] = df["iteration"] <= last if last is not None else False
        both = df[(df["source"] == "log") & df["lr_config"].notna()]
        peak = info.get("peak_lr") or float("nan")
        diff = float((both["lr"] - both["lr_config"]).abs().max() / peak) if len(both) else float("nan")
        df.insert(0, "run", run_name)
        frames.append(df.reset_index(drop=True))
        infos.append({"run": run_name, **info, "last_trained_iter": last,
                      "lr_source": "log" if not logged.empty else "config",
                      "n_logged": len(logged), "max_rel_diff_log_vs_config": diff})
    lr_df = pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=["run", "iteration", "lr", "lr_config", "source", "trained"])
    return lr_df, pd.DataFrame(infos).set_index("run")


def plot_lr(
    lr_df: pd.DataFrame,
    *,
    val_df: pd.DataFrame | None = None,
    logy: bool = False,
    mark_best: bool = True,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Axes:
    """lr vs iteration per run: solid = trained so far (logged lr where the run
    logs it), dotted = planned rest (config rebuild).

    With ``val_df``, stars mark the lowest-val-loss iteration (the tested
    checkpoint) on each run's lr curve (``mark_best=False`` turns them off).
    """
    if lr_df.empty:
        raise ValueError("Cannot plot empty lr DataFrame")
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    best = (best_val_iterations(val_df)
            if mark_best and val_df is not None and not val_df.empty else {})
    for run_name, g in lr_df.groupby("run", sort=False):
        done, rest = g[g["trained"]], g[~g["trained"]]
        (line,) = ax.plot(done["iteration"], done["lr"], lw=1.6, label=run_name)
        if not rest.empty:
            ax.plot(rest["iteration"], rest["lr"], ":", lw=1.2, color=line.get_color())
        b = g[g["iteration"] == best.get(run_name)]
        if not b.empty:
            ax.plot(b["iteration"], b["lr"], "*", ms=13, mec="k", mew=0.8,
                    color=line.get_color())
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("learning rate")
    ax.set_title(title or ("PPP learning rate (logged in run.log; dotted = planned, from config.toml)"
                           + ("; ★ = lowest val loss" if best else "")))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
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
        for run_name, vset, group in _val_groups(vdf):
            srcs = ", ".join(f"{k}={v}" for k, v in
                             group["source"].value_counts().items())
            print(f"{run_name} [{vset}]: {len(group)} val points ({srcs})")
        if not vdf.empty:
            print(vdf.tail().to_string(index=False))
            vs = val_summary(vdf)
            if not vs.empty:
                print(vs.to_string(float_format=lambda x: f"{x:.4f}"))
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
