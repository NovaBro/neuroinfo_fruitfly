"""Ray Tune entrypoint for BiaPy nested-sbatch trials (Stages 3 + 6).

Requires Ray (+ Optuna for ``--search optuna``) in the environment (Stage 5).

Fixed smoke (one-point space)::

    python biapy_work_folder/raytune/run_tune.py \\
      --exp biapy_fdb_skel_smoke --num-samples 1 --dry-run \\
      --lr 1e-3 --batch-size 30 --skel-weight 0.1

Bayesian Optuna sweep (dry-run samples configs without GPU)::

    python biapy_work_folder/raytune/run_tune.py --search optuna --gpu h200 \\
      --exp biapy_fdb_skel_optuna_dry --num-samples 4 --dry-run

Controller (real nested GPU)::

    sbatch sbatch/biapy/raytune_controller_sbatch.sh \\
      --search optuna --gpu h200 --exp biapy_fdb_skel_optuna \\
      --num-samples 8 --max-concurrent 1 \\
      --search-epochs 40 --search-patience 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from biapy_work_folder.raytune.search_space import (
    SPACE_PRESETS,
    build_search_space,
    constraint_for_gpu,
    mem_for_gpu,
)
from biapy_work_folder.raytune.slurm_trial import (
    DEFAULT_ACCOUNT,
    DEFAULT_CONSTRAINT,
    DEFAULT_CPUS,
    DEFAULT_GRES,
    DEFAULT_MEM,
    DEFAULT_METRIC,
    DEFAULT_MODE,
    DEFAULT_POLL_S,
    DEFAULT_TIME,
)
from biapy_work_folder.raytune.trainable import (
    _RAY_INSTALL_HINT,
    make_biapy_sbatch_trainable,
)
from biapy_work_folder.raytune.winner import write_winner_config
from biapy_work_folder.raytune.write_trial_config import (
    HPARAM_MAP,
    _VIRTUAL_HPARAMS,
    base_uses_dice_loss,
    base_uses_skeleton_recall,
)

STORAGE_PATH = Path("metrics/biapy/raytune")

_DEFAULT_SMOKE_HPARAMS = {
    "lr": 1e-3,
    "batch_size": 30,
    "skel_weight": 0.1,
}

_OPTUNA_HINT = (
    "OptunaSearch requires optuna (and ray[tune]). "
    "Install into BiaPy_env (Stage 5): pip install \"ray[tune]\" optuna"
)


def _require_ray():
    try:
        import ray
        from ray import tune
    except ImportError as exc:
        raise ImportError(_RAY_INSTALL_HINT) from exc
    return ray, tune


def _init_ray(ray) -> None:
    """Pin Ray to the SLURM allocation when present.

    Without this, ``Tuner.fit()`` auto-init sees the whole host (e.g. 128 CPUs)
    and tries to prestart that many workers inside a small cgroup/Singularity
    job, which hangs before any nested sbatch runs.
    """
    if ray.is_initialized():
        return
    cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get(
        "SLURM_JOB_CPUS_PER_NODE"
    )
    if not cpus:
        # Interactive / non-SLURM: leave uninitialized so Tuner.fit() auto-inits.
        return
    n = int(str(cpus).split("(")[0].split(",")[0])
    print(f"ray.init under SLURM: num_cpus={n}, object_store_memory=2GiB")
    ray.init(
        num_cpus=n,
        object_store_memory=2 * 1024**3,
        include_dashboard=False,
    )


def _require_optuna_search():
    try:
        from ray.tune.search.optuna import OptunaSearch
    except ImportError as exc:
        raise ImportError(_OPTUNA_HINT) from exc
    return OptunaSearch


def _hparams_from_args(args: argparse.Namespace) -> dict[str, Any]:
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
    allowed = set(HPARAM_MAP) | set(_VIRTUAL_HPARAMS)
    unknown = set(hparams) - allowed
    if unknown:
        raise ValueError(f"Unknown hparam keys: {sorted(unknown)}")
    return hparams


def build_param_space_fixed(hparams: dict[str, Any] | None) -> dict[str, Any]:
    """One-point param space for Stage 3 / ``--search fixed`` smoke."""
    if not hparams:
        hparams = dict(_DEFAULT_SMOKE_HPARAMS)
    return dict(hparams)


def _resolve_constraint(argv: Sequence[str], gpu: str, constraint: str) -> str:
    if "--constraint" in argv:
        return constraint
    return constraint_for_gpu(gpu)


def _resolve_mem(argv: Sequence[str], gpu: str, mem: str) -> str:
    if "--mem" in argv:
        return mem
    return mem_for_gpu(gpu)


def _resolve_num_samples(search: str, num_samples: int | None) -> int:
    if num_samples is not None:
        return int(num_samples)
    return 8 if search == "optuna" else 1


def _write_artifacts(
    results: Any,
    exp_dir: Path,
    *,
    metric: str,
    mode: str,
    base: str,
    search_epochs: int | None,
    search_patience: int | None,
) -> dict[str, str]:
    """Write results_table.csv + best_config.yaml under exp_dir."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    table_path = exp_dir / "results_table.csv"
    try:
        df = results.get_dataframe()
        # Prefer a stable column order when present.
        preferred = [
            c
            for c in (
                "trial_id",
                metric,
                "config/lr",
                "config/batch_size",
                "config/skel_weight",
                "config/scheduler",
                "slurm_job_id",
                "out_path",
                "run_id",
                "config_relpath",
                "state",
            )
            if c in df.columns
        ]
        extra = [c for c in df.columns if c not in preferred]
        df.loc[:, preferred + extra].to_csv(table_path, index=False)
    except Exception as exc:  # noqa: BLE001
        table_path.write_text(f"# failed to build results table: {exc}\n")
    paths["results_table"] = str(table_path)

    best_path = exp_dir / "best_config.yaml"
    payload: dict[str, Any] = {
        "metric": metric,
        "mode": mode,
        "base": base,
        "search_epochs": search_epochs,
        "search_patience": search_patience,
    }
    try:
        best = results.get_best_result(metric=metric, mode=mode)
        payload["value"] = (
            best.metrics.get(metric) if best.metrics else None
        )
        payload["hparams"] = dict(best.config) if best.config else None
        if best.metrics:
            for key in (
                "slurm_job_id",
                "out_path",
                "run_id",
                "config_relpath",
                "state",
            ):
                if key in best.metrics:
                    payload[key] = best.metrics[key]
        payload["ray_trial_path"] = str(best.path) if best.path else None
    except Exception as exc:  # noqa: BLE001
        payload["best_error"] = str(exc)

    best_path.write_text(
        yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)
    )
    paths["best_config"] = str(best_path)
    return paths


def _count_user_train_jobs() -> int:
    """Count current user SLURM jobs whose name looks like a BiaPy train job."""
    import getpass
    import subprocess

    user = getpass.getuser()
    proc = subprocess.run(
        ["squeue", "-u", user, "-h", "-o", "%j"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        # squeue unavailable (e.g. dry local test) — do not block.
        return 0
    count = 0
    for line in proc.stdout.splitlines():
        name = line.strip()
        if "-train" in name or name.endswith("train"):
            count += 1
    return count


def _guard_user_trains(max_user_trains: int, *, dry_run: bool) -> None:
    if dry_run or max_user_trains < 0:
        return
    n = _count_user_train_jobs()
    if n >= max_user_trains:
        raise SystemExit(
            f"Refusing to start: {n} user jobs matching '*-train' already "
            f"in squeue (limit --max-user-trains={max_user_trains}). "
            f"Wait for them to finish or raise the limit."
        )


def _exp_dir_nonempty(exp_dir: Path) -> bool:
    if not exp_dir.exists():
        return False
    try:
        next(exp_dir.iterdir())
        return True
    except StopIteration:
        return False


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Ray Tune for BiaPy nested-sbatch trials "
            "(--search fixed|optuna; repo root cwd)."
        )
    )
    p.add_argument("--base", default="biapy-aug-zarr-seunet-FDb-skel")
    p.add_argument(
        "--job-name",
        default=None,
        help="BiaPy job name (default: base stem)",
    )
    p.add_argument("--exp", default="biapy_fdb_skel_smoke", help="Ray experiment name")
    p.add_argument(
        "--search",
        choices=("fixed", "optuna"),
        default="fixed",
        help="fixed=one-point CLI/defaults; optuna=Bayesian search_space",
    )
    p.add_argument(
        "--gpu",
        default="h200",
        choices=("h200", "l40", "l40s"),
        help="Search-space branch + default SLURM constraint",
    )
    p.add_argument(
        "--space",
        default="legacy",
        choices=SPACE_PRESETS,
        help="Optuna search-space preset (legacy keeps in-flight sweeps stable; "
        "wc_nopat = warmupcosine + warmup epochs + loss weights, no batch search)",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Trial count (default: 1 for fixed, 8 for optuna)",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Nested GPU jobs in flight (prefer 1–2 for Optuna)",
    )
    p.add_argument(
        "--max-user-trains",
        type=int,
        default=4,
        help="Refuse to start if this many *-train jobs already in squeue "
        "(-1 disables; skipped for --dry-run)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Restore Tuner from metrics/biapy/raytune/<exp>/",
    )
    p.add_argument("--metric", default=DEFAULT_METRIC)
    p.add_argument("--mode", choices=("min", "max"), default=DEFAULT_MODE)
    p.add_argument("--search-epochs", type=int, default=40)
    p.add_argument("--search-patience", type=int, default=10)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass dry_run to Stage 2 (print sbatch; no GPU job)",
    )
    p.add_argument("--account", default=DEFAULT_ACCOUNT)
    p.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
    p.add_argument("--mem", default=DEFAULT_MEM)
    p.add_argument("--time", dest="time_limit", default=DEFAULT_TIME)
    p.add_argument("--constraint", default=DEFAULT_CONSTRAINT)
    p.add_argument("--gres", default=DEFAULT_GRES)
    p.add_argument("--poll-s", type=float, default=DEFAULT_POLL_S)
    p.add_argument("--timeout-s", type=float, default=None)
    # Optional one-point hparams (--search fixed)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--skel-weight", type=float, default=None)
    p.add_argument("--scheduler", type=str, default=None)
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--warmup-cosine-epochs", type=int, default=None)
    p.add_argument("--channel-weights", type=str, default=None)
    p.add_argument(
        "--f-weight",
        type=float,
        default=None,
        help="Virtual hparam → channel_weights=[f_weight, db_weight]",
    )
    p.add_argument(
        "--db-weight",
        type=float,
        default=None,
        help="Db channel weight when using --f-weight (default 2.0)",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    os.chdir(_REPO_ROOT)
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args = _build_argparser().parse_args(argv_list)
    ray, tune = _require_ray()
    _init_ray(ray)

    _guard_user_trains(args.max_user_trains, dry_run=args.dry_run)

    search_epochs = args.search_epochs if args.search_epochs >= 0 else None
    search_patience = args.search_patience if args.search_patience >= 0 else None
    num_samples = _resolve_num_samples(args.search, args.num_samples)
    constraint = _resolve_constraint(argv_list, args.gpu, args.constraint)
    mem = _resolve_mem(argv_list, args.gpu, args.mem)

    search_alg = None
    include_skel: bool | None = None
    include_f: bool | None = None
    if args.search == "optuna":
        OptunaSearch = _require_optuna_search()
        include_skel = base_uses_skeleton_recall(args.base)
        include_f = base_uses_dice_loss(args.base) and not include_skel
        print(
            f"build_search_space: preset={args.space!r} "
            f"include_skel_weight={include_skel} include_f_weight={include_f} "
            f"search_epochs={search_epochs} (base={args.base})"
        )
        param_space = build_search_space(
            gpu=args.gpu,
            include_skel_weight=include_skel,
            include_f_weight=include_f,
            search_epochs=search_epochs,
            preset=args.space,
        )
        search_alg = OptunaSearch(metric=args.metric, mode=args.mode)
    else:
        param_space = build_param_space_fixed(_hparams_from_args(args))

    objective = make_biapy_sbatch_trainable(
        base_yaml=args.base,
        job_name=args.job_name,
        metric=args.metric,
        mode=args.mode,
        search_epochs=search_epochs,
        search_patience=search_patience,
        dry_run=args.dry_run,
        skip_finished=True,
        poll_s=args.poll_s,
        timeout_s=args.timeout_s,
        account=args.account,
        cpus=args.cpus,
        mem=mem,
        time_limit=args.time_limit,
        constraint=constraint,
        gres=args.gres,
    )
    trainable = tune.with_resources(objective, {"cpu": 1})

    STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    exp_dir = STORAGE_PATH / args.exp

    if args.resume:
        if not exp_dir.exists():
            raise SystemExit(
                f"--resume requires existing experiment dir: {exp_dir}"
            )
        try:
            tuner = tune.Tuner.restore(
                str(exp_dir.resolve()),
                trainable=trainable,
                resume_unfinished=True,
                resume_errored=True,
            )
        except TypeError:
            # Older Ray may use different kw-only names.
            try:
                tuner = tune.Tuner.restore(
                    str(exp_dir.resolve()),
                    trainable=trainable,
                )
            except Exception as exc:  # noqa: BLE001
                raise SystemExit(
                    f"Tuner.restore failed for {exp_dir}: {exc}"
                ) from exc
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Tuner.restore failed for {exp_dir}: {exc}") from exc
    else:
        if _exp_dir_nonempty(exp_dir):
            raise SystemExit(
                f"Experiment dir already exists and is non-empty: {exp_dir}\n"
                f"Pass --resume to continue, or choose a new --exp name."
            )
        tune_kwargs: dict[str, Any] = {
            "num_samples": num_samples,
            "max_concurrent_trials": args.max_concurrent,
            "metric": args.metric,
            "mode": args.mode,
        }
        if search_alg is not None:
            tune_kwargs["search_alg"] = search_alg

        tuner = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(**tune_kwargs),
            run_config=tune.RunConfig(
                name=args.exp,
                storage_path=str(STORAGE_PATH.resolve()),
            ),
        )

    results = tuner.fit()

    artifact_paths = _write_artifacts(
        results,
        exp_dir,
        metric=args.metric,
        mode=args.mode,
        base=args.base,
        search_epochs=search_epochs,
        search_patience=search_patience,
    )

    payload: dict[str, Any] = {
        "exp": args.exp,
        "search": args.search,
        "gpu": args.gpu,
        "constraint": constraint,
        "mem": mem,
        "num_samples": num_samples,
        "max_concurrent": args.max_concurrent,
        "resume": args.resume,
        "dry_run": args.dry_run,
        "ray_version": getattr(ray, "__version__", "unknown"),
        "artifacts": artifact_paths,
        "exp_dir": str(exp_dir),
    }
    if args.search == "fixed":
        payload["param_space"] = param_space
    else:
        payload["param_space"] = (
            "build_search_space(gpu=%r, include_skel_weight=%r, "
            "include_f_weight=%r, search_epochs=%r, preset=%r)"
            % (args.gpu, include_skel, include_f, search_epochs, args.space)
        )

    best_config = None
    try:
        best = results.get_best_result(metric=args.metric, mode=args.mode)
        payload["best_metrics"] = dict(best.metrics) if best.metrics else None
        best_config = dict(best.config) if best.config else None
        payload["best_config"] = best_config
        payload["best_path"] = str(best.path) if best.path else None
    except Exception as exc:  # noqa: BLE001
        payload["best_error"] = str(exc)
        payload["num_errors"] = (
            results.num_errors if hasattr(results, "num_errors") else None
        )

    if best_config:
        winner_rel, chain_cmd = write_winner_config(
            base_yaml=args.base,
            hparams=best_config,
            exp=args.exp,
            search_epochs=None,
            search_patience=None,
        )
        pointer = {
            "winner_config_relpath": winner_rel.as_posix(),
            "chain_command": chain_cmd,
            "hparams": best_config,
            "metric": args.metric,
            "mode": args.mode,
            "value": (payload.get("best_metrics") or {}).get(args.metric),
        }
        pointer_path = exp_dir / "winner_pointer.yaml"
        exp_dir.mkdir(parents=True, exist_ok=True)
        pointer_path.write_text(
            yaml.safe_dump(pointer, default_flow_style=False, sort_keys=False)
        )
        artifact_paths["winner_pointer"] = str(pointer_path)
        artifact_paths["winner_config"] = str(
            Path("biapy_work_folder/configs") / winner_rel
        )
        payload["artifacts"] = artifact_paths
        payload["winner_config"] = winner_rel.as_posix()
        payload["chain_command"] = chain_cmd

    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
