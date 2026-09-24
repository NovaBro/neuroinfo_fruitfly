"""Write BiaPy trial YAMLs for Ray Tune (scheduler-only, nested sbatch).

Stage 1: deep-copy a base config, apply safe train hparam overrides, and write
``biapy_work_folder/configs/trials/<trial_id>.yaml`` for later sbatch submit.

Run from repo root::

    python biapy_work_folder/raytune/write_trial_config.py \\
      --base biapy-aug-zarr-seunet-FDb-skel \\
      --trial-id smoke_lr1e3 \\
      --lr 1e-3 --batch-size 30 --skel-weight 0.1 \\
      --search-epochs 40 --search-patience 10

Manual train handoff (Stage 2 will automate)::

    sbatch --account=torch_pr_61_general \\
      --job-name=biapy-aug-zarr-seunet-FDb-skel-rsmoke_lr1e3-train \\
      --cpus-per-task=16 --time=24:00:00 --mem=384g \\
      --gres=gpu:1 --constraint='h200' \\
      sbatch/biapy/biapy-py_sbatch.sh \\
      trials/smoke_lr1e3.yaml train biapy-aug-zarr-seunet-FDb-skel smoke_lr1e3

Always pass job_name ``$3`` = base stem (not ``trials``).
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path("biapy_work_folder/configs")
TRIALS_SUBDIR = "trials"

# Flat hparam key -> nested YAML path under the BiaPy config dict.
HPARAM_MAP: dict[str, tuple[str, ...]] = {
    "lr": ("TRAIN", "LR"),
    "batch_size": ("TRAIN", "BATCH_SIZE"),
    "skel_weight": ("LOSS", "SKELETON_RECALL", "WEIGHT"),
    "scheduler": ("TRAIN", "LR_SCHEDULER", "NAME"),
    "min_lr": ("TRAIN", "LR_SCHEDULER", "MIN_LR"),
    "warmup_cosine_epochs": ("TRAIN", "LR_SCHEDULER", "WARMUP_COSINE_DECAY_EPOCHS"),
    "channel_weights": ("PROBLEM", "INSTANCE_SEG", "DATA_CHANNEL_WEIGHTS"),
    "epochs": ("TRAIN", "EPOCHS"),
    "patience": ("TRAIN", "PATIENCE"),
}

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_trial_id(trial_id: str, *, max_len: int = 64) -> str:
    """Make a trial id safe for filesystem and SLURM job-name fragments."""
    cleaned = _SAFE_ID_RE.sub("_", str(trial_id).strip()).strip("._-")
    if not cleaned:
        cleaned = "rt"
    if cleaned[0].isdigit() or cleaned[0] in ".-":
        cleaned = f"rt_{cleaned}"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("._-")
    return cleaned or "rt"


def resolve_base_yaml(base_yaml: Path | str) -> Path:
    """Resolve a base config path or stem under ``CONFIG_DIR``."""
    raw = Path(base_yaml)
    candidates: list[Path] = []
    if raw.is_file():
        return raw.resolve()
    candidates.append(raw)
    if not raw.suffix:
        candidates.append(Path(f"{raw}.yaml"))
    stem = raw.name
    if stem.endswith(".yaml"):
        candidates.append(CONFIG_DIR / stem)
    else:
        candidates.append(CONFIG_DIR / f"{stem}.yaml")
        candidates.append(CONFIG_DIR / stem)

    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Base YAML not found for {base_yaml!r}; tried: "
        + ", ".join(str(p) for p in candidates)
    )


def apply_train_mode_flags(cfg: dict) -> dict:
    """Mirror ``run_biapy-py.apply_mode_overrides(..., mode='train')``."""
    cfg.setdefault("TRAIN", {})
    cfg.setdefault("TEST", {})
    cfg.setdefault("MODEL", {})
    cfg["TRAIN"]["ENABLE"] = True
    cfg["TEST"]["ENABLE"] = False
    cfg["MODEL"]["LOAD_CHECKPOINT"] = False
    return cfg


def skeleton_recall_enabled(cfg: dict) -> bool:
    """True when ``LOSS.SKELETON_RECALL.ENABLE`` is set (skel bases, not dice)."""
    loss = cfg.get("LOSS")
    if not isinstance(loss, dict):
        return False
    skel = loss.get("SKELETON_RECALL")
    if not isinstance(skel, dict):
        return False
    return bool(skel.get("ENABLE"))


def dice_loss_enabled(cfg: dict) -> bool:
    """True when ``DATA_CHANNELS_LOSSES`` includes ``dice`` (FDb-dice bases)."""
    try:
        losses = cfg["PROBLEM"]["INSTANCE_SEG"]["DATA_CHANNELS_LOSSES"]
    except (KeyError, TypeError):
        return False
    if not isinstance(losses, (list, tuple)):
        return False
    return any(str(x).lower() == "dice" for x in losses)


def base_uses_skeleton_recall(base_yaml: Path | str) -> bool:
    """Load base YAML and report whether skeleton-recall loss is enabled."""
    path = resolve_base_yaml(base_yaml)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        return False
    return skeleton_recall_enabled(cfg)


def base_uses_dice_loss(base_yaml: Path | str) -> bool:
    """Load base YAML and report whether instance Dice loss is configured."""
    path = resolve_base_yaml(base_yaml)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        return False
    return dice_loss_enabled(cfg)


# Virtual keys expanded in ``normalize_hparams`` before ``HPARAM_MAP`` apply.
_VIRTUAL_HPARAMS = frozenset({"f_weight", "db_weight"})
DEFAULT_DB_WEIGHT_FOR_F = 2.0  # matches FDb-dice YAML DATA_CHANNEL_WEIGHTS Db


def normalize_hparams(hparams: dict) -> dict:
    """Expand virtual keys (``f_weight`` → ``channel_weights``) and copy.

    ``f_weight`` becomes ``channel_weights = [f_weight, db_weight]`` with
    ``db_weight`` defaulting to ``DEFAULT_DB_WEIGHT_FOR_F`` (2.0).
    """
    out = dict(hparams)
    if "f_weight" not in out:
        out.pop("db_weight", None)
        return out
    f_w = float(out.pop("f_weight"))
    db_w = float(out.pop("db_weight", DEFAULT_DB_WEIGHT_FOR_F))
    out["channel_weights"] = [f_w, db_w]
    return out


def _set_nested(cfg: dict, path: tuple[str, ...], value: Any) -> None:
    """Set ``cfg[a][b][c] = value``, creating intermediate dicts as needed."""
    node: dict = cfg
    for key in path[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise TypeError(
                f"Cannot set {'.'.join(path)}: {key!r} is {type(child).__name__}, "
                f"expected dict"
            )
        node = child
    node[path[-1]] = value


def apply_hparams(cfg: dict, hparams: dict) -> dict:
    """Apply known flat hparams; raise on unknown keys.

    Skips ``skel_weight`` when skeleton-recall loss is not enabled so dice
    (and other non-skel) bases do not grow an orphan ``LOSS.SKELETON_RECALL``.
    Accepts virtual ``f_weight`` / ``db_weight`` (see ``normalize_hparams``).
    """
    resolved = normalize_hparams(hparams)
    unknown = sorted(set(resolved) - set(HPARAM_MAP))
    if unknown:
        raise ValueError(
            f"Unknown hparam key(s): {unknown}. "
            f"Allowed: {sorted(HPARAM_MAP)} (+ virtual {sorted(_VIRTUAL_HPARAMS)})"
        )
    skel_on = skeleton_recall_enabled(cfg)
    for key, value in resolved.items():
        if key == "skel_weight" and not skel_on:
            print(
                "apply_hparams: skipping skel_weight "
                "(LOSS.SKELETON_RECALL.ENABLE is not True)"
            )
            continue
        _set_nested(cfg, HPARAM_MAP[key], value)
    return cfg


def _format_header(
    trial_id: str,
    base_path: Path,
    hparams: dict,
    search_epochs: int | None,
    search_patience: int | None,
) -> str:
    payload = {
        "trial_id": trial_id,
        "base": base_path.name,
        "hparams": hparams,
        "search_epochs": search_epochs,
        "search_patience": search_patience,
    }
    # Compact single-line JSON keeps the YAML header readable.
    return (
        "# Generated by biapy_work_folder/raytune/write_trial_config.py\n"
        f"# raytune {json.dumps(payload, sort_keys=True, default=str)}\n"
    )


_DATA_CHANNELS_BLOCK_RE = re.compile(
    r"^([ \t]*)DATA_CHANNELS:\n(?:\1- .+\n)+",
    re.MULTILINE,
)


def _flow_data_channels_line(indent: str, channels: list) -> str:
    """Single-line ``DATA_CHANNELS: ['F', 'Db']`` for biapy-py_sbatch.sh grep."""
    inner = ", ".join(repr(str(c)) for c in channels)
    return f"{indent}DATA_CHANNELS: [{inner}]\n"


def preserve_data_channels_flow(text: str, cfg: dict) -> str:
    """Rewrite block-style DATA_CHANNELS so staging grep still matches Db/Dc."""
    try:
        channels = list(cfg["PROBLEM"]["INSTANCE_SEG"]["DATA_CHANNELS"])
    except (KeyError, TypeError):
        return text

    def _repl(match: re.Match[str]) -> str:
        return _flow_data_channels_line(match.group(1), channels)

    return _DATA_CHANNELS_BLOCK_RE.sub(_repl, text, count=1)


def write_trial_config(
    base_yaml: Path | str,
    hparams: dict,
    trial_id: str,
    *,
    search_epochs: int | None = 40,
    search_patience: int | None = 10,
    out_dir: Path | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> Path:
    """Write ``configs/trials/<trial_id>.yaml``; return path relative to CONFIG_DIR.

    ``search_epochs`` / ``search_patience`` overwrite ``TRAIN.EPOCHS`` /
    ``TRAIN.PATIENCE`` when not None (win over hparam ``epochs`` / ``patience``).

    If the trial YAML already exists and ``overwrite`` is False, leave it alone
    and return the existing relative path (audit-friendly for Ray resume).
    """
    base_path = resolve_base_yaml(base_yaml)
    run_id = sanitize_trial_id(trial_id)
    trials_dir = out_dir if out_dir is not None else CONFIG_DIR / TRIALS_SUBDIR
    rel_path = Path(TRIALS_SUBDIR) / f"{run_id}.yaml"
    out_path = trials_dir / f"{run_id}.yaml"

    if out_path.is_file() and not overwrite and not dry_run:
        print(f"write_trial_config: keeping existing {out_path}")
        return rel_path

    with open(base_path) as f:
        cfg = copy.deepcopy(yaml.safe_load(f))
    if not isinstance(cfg, dict):
        raise ValueError(f"Base YAML root must be a mapping: {base_path}")

    apply_train_mode_flags(cfg)
    apply_hparams(cfg, dict(hparams))

    cfg.setdefault("TRAIN", {})
    if search_epochs is not None:
        cfg["TRAIN"]["EPOCHS"] = int(search_epochs)
    if search_patience is not None:
        cfg["TRAIN"]["PATIENCE"] = int(search_patience)

    body = yaml.safe_dump(
        cfg,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    # safe_dump expands lists to block style; sbatch staging greps for
    # DATA_CHANNELS: [...'Db'...] / [...'Dc'...] on one line.
    body = preserve_data_channels_flow(body, cfg)
    text = _format_header(
        run_id, base_path, dict(hparams), search_epochs, search_patience
    ) + body

    if dry_run:
        print(text, end="" if text.endswith("\n") else "\n")
        return rel_path

    trials_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return rel_path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Write a BiaPy trial YAML under biapy_work_folder/configs/trials/ "
            "(run from repo root)."
        )
    )
    p.add_argument(
        "--base",
        default="biapy-aug-zarr-seunet-FDb-skel",
        help="Base config stem or path (default: FDb-skel)",
    )
    p.add_argument("--trial-id", required=True, help="Trial / run id")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--skel-weight", type=float, default=None)
    p.add_argument("--scheduler", type=str, default=None)
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--warmup-cosine-epochs", type=int, default=None)
    p.add_argument(
        "--channel-weights",
        type=str,
        default=None,
        help='JSON list, e.g. "[1.0, 1.0]"',
    )
    p.add_argument(
        "--f-weight",
        type=float,
        default=None,
        help="Virtual → channel_weights=[f_weight, db_weight]",
    )
    p.add_argument(
        "--db-weight",
        type=float,
        default=None,
        help="Db weight with --f-weight (default 2.0)",
    )
    p.add_argument("--epochs", type=int, default=None, help="Hparam EPOCHS")
    p.add_argument("--patience", type=int, default=None, help="Hparam PATIENCE")
    p.add_argument(
        "--search-epochs",
        type=int,
        default=40,
        help="Overwrite TRAIN.EPOCHS (default 40; use negative to skip)",
    )
    p.add_argument(
        "--search-patience",
        type=int,
        default=10,
        help="Overwrite TRAIN.PATIENCE (default 10; use negative to skip)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print YAML to stdout; do not write a file",
    )
    return p


def _hparams_from_args(args: argparse.Namespace) -> dict:
    hparams: dict[str, Any] = {}
    if args.lr is not None:
        hparams["lr"] = args.lr
    if args.batch_size is not None:
        hparams["batch_size"] = args.batch_size
    if args.skel_weight is not None:
        hparams["skel_weight"] = args.skel_weight
    if args.scheduler is not None:
        hparams["scheduler"] = args.scheduler
    if args.min_lr is not None:
        hparams["min_lr"] = args.min_lr
    if args.warmup_cosine_epochs is not None:
        hparams["warmup_cosine_epochs"] = args.warmup_cosine_epochs
    if args.channel_weights is not None:
        weights = json.loads(args.channel_weights)
        if not isinstance(weights, (list, tuple)):
            raise ValueError("--channel-weights must be a JSON list")
        hparams["channel_weights"] = list(weights)
    if getattr(args, "f_weight", None) is not None:
        hparams["f_weight"] = args.f_weight
    if getattr(args, "db_weight", None) is not None:
        hparams["db_weight"] = args.db_weight
    if args.epochs is not None:
        hparams["epochs"] = args.epochs
    if args.patience is not None:
        hparams["patience"] = args.patience
    return hparams


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    search_epochs = args.search_epochs if args.search_epochs >= 0 else None
    search_patience = args.search_patience if args.search_patience >= 0 else None
    rel = write_trial_config(
        args.base,
        _hparams_from_args(args),
        args.trial_id,
        search_epochs=search_epochs,
        search_patience=search_patience,
        dry_run=args.dry_run,
    )
    # Relative path for sbatch $1 (even on dry-run so scripts can capture it).
    print(rel.as_posix())


if __name__ == "__main__":
    main()
