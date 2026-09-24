"""Submit / wait / parse BiaPy train SLURM jobs for Ray Tune (Stage 2).

Assumes cwd is the repo root. Reuses ``sbatch/biapy/biapy-py_sbatch.sh`` and
``biapy_train_track.consolidate_training_log``.

Offline parse (no GPU)::

    python biapy_work_folder/raytune/slurm_trial.py parse \\
      --out sbatch/biapy/biapy-aug-zarr-seunet-FDb-skel-rlr_cosine-train-17344918.out

Full submit + wait + parse::

    python biapy_work_folder/raytune/slurm_trial.py run \\
      --config trials/smoke_lr1e3.yaml \\
      --job-name biapy-aug-zarr-seunet-FDb-skel \\
      --run-id smoke_lr1e3

Dry-run (print sbatch argv, do not submit)::

    python biapy_work_folder/raytune/slurm_trial.py run \\
      --config trials/smoke_lr1e3.yaml \\
      --job-name biapy-aug-zarr-seunet-FDb-skel \\
      --run-id smoke_lr1e3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

# Allow ``python biapy_work_folder/raytune/slurm_trial.py`` from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from biapy_work_folder.biapy_train_track import consolidate_training_log
from biapy_work_folder.raytune.write_trial_config import CONFIG_DIR

SBATCH_SCRIPT = Path("sbatch/biapy/biapy-py_sbatch.sh")
OUT_DIR = Path("sbatch/biapy")

DEFAULT_ACCOUNT = "torch_pr_61_general"
DEFAULT_CPUS = 16
DEFAULT_MEM = "384g"
DEFAULT_TIME = "24:00:00"
DEFAULT_CONSTRAINT = "h200"
DEFAULT_GRES = "gpu:1"
DEFAULT_POLL_S = 60.0
DEFAULT_METRIC = "val_loss"
DEFAULT_MODE = "min"

_SUCCESS_STATE = "COMPLETED"
_FAILURE_STATES = frozenset(
    {
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "REVOKED",
    }
)


@dataclass
class TrialJobResult:
    job_id: str
    out_path: Path
    metric: float
    state: str
    exit_code: str | None = None


def slurm_job_name(job_name: str, run_id: str) -> str:
    """SLURM ``--job-name`` / ``%x`` fragment: ``{job_name}-r{run_id}-train``."""
    return f"{job_name}-r{run_id}-train"


def find_train_out(job_name: str, run_id: str, job_id: str) -> Path:
    """Resolve ``sbatch/biapy/{job_name}-r{run_id}-train-{job_id}.out``."""
    return OUT_DIR / f"{slurm_job_name(job_name, run_id)}-{job_id}.out"


def find_train_err(job_name: str, run_id: str, job_id: str) -> Path:
    """Resolve matching ``.err`` beside the train ``.out``."""
    return OUT_DIR / f"{slurm_job_name(job_name, run_id)}-{job_id}.err"


def find_existing_train_outs(job_name: str, run_id: str) -> list[Path]:
    """Glob ``sbatch/biapy/{job}-r{run_id}-train-*.out``, newest first."""
    pattern = f"{slurm_job_name(job_name, run_id)}-*.out"
    paths = sorted(OUT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return paths


def _job_id_from_out_path(out_path: Path, job_name: str, run_id: str) -> str:
    prefix = f"{slurm_job_name(job_name, run_id)}-"
    stem = out_path.name
    if stem.startswith(prefix) and stem.endswith(".out"):
        return stem[len(prefix) : -len(".out")]
    return "unknown"


def try_reuse_finished_trial(
    job_name: str,
    run_id: str,
    *,
    metric: str = DEFAULT_METRIC,
    mode: str = DEFAULT_MODE,
) -> TrialJobResult | None:
    """Return a result if an existing train ``.out`` parses to a usable metric."""
    for out_path in find_existing_train_outs(job_name, run_id):
        try:
            value = metric_from_out(out_path, metric=metric, mode=mode)
        except (RuntimeError, FileNotFoundError, ValueError):
            continue
        job_id = _job_id_from_out_path(out_path, job_name, run_id)
        return TrialJobResult(
            job_id=job_id,
            out_path=out_path,
            metric=value,
            state="REUSED",
            exit_code="0:0",
        )
    return None


def build_sbatch_argv(
    config_relpath: str | Path,
    job_name: str,
    run_id: str,
    *,
    account: str = DEFAULT_ACCOUNT,
    cpus: int = DEFAULT_CPUS,
    mem: str = DEFAULT_MEM,
    time_limit: str = DEFAULT_TIME,
    constraint: str = DEFAULT_CONSTRAINT,
    gres: str = DEFAULT_GRES,
) -> list[str]:
    """Build ``sbatch --parsable …`` argv (no submit)."""
    config_relpath = Path(config_relpath).as_posix()
    return [
        "sbatch",
        "--parsable",
        f"--account={account}",
        f"--job-name={slurm_job_name(job_name, run_id)}",
        f"--cpus-per-task={cpus}",
        f"--time={time_limit}",
        f"--mem={mem}",
        f"--gres={gres}",
        f"--constraint={constraint}",
        str(SBATCH_SCRIPT),
        config_relpath,
        "train",
        job_name,
        run_id,
    ]


def _require_submit_paths(config_relpath: str | Path) -> Path:
    if not SBATCH_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing sbatch script: {SBATCH_SCRIPT}")
    cfg = CONFIG_DIR / Path(config_relpath)
    if not cfg.is_file():
        raise FileNotFoundError(
            f"Trial config not found: {cfg} "
            f"(expected under {CONFIG_DIR}/)"
        )
    return cfg


def submit_train_job(
    config_relpath: str | Path,
    job_name: str,
    run_id: str,
    *,
    account: str = DEFAULT_ACCOUNT,
    cpus: int = DEFAULT_CPUS,
    mem: str = DEFAULT_MEM,
    time_limit: str = DEFAULT_TIME,
    constraint: str = DEFAULT_CONSTRAINT,
    gres: str = DEFAULT_GRES,
    dry_run: bool = False,
) -> str:
    """Submit GPU train via ``biapy-py_sbatch.sh``; return SLURM job id.

    If ``dry_run``, print the command and return ``DRY_RUN`` without submitting.
    """
    _require_submit_paths(config_relpath)
    argv = build_sbatch_argv(
        config_relpath,
        job_name,
        run_id,
        account=account,
        cpus=cpus,
        mem=mem,
        time_limit=time_limit,
        constraint=constraint,
        gres=gres,
    )
    if dry_run:
        print(shlex.join(argv))
        return "DRY_RUN"

    # Nested sbatch must not inherit the controller job's allocation env
    # (partition/mem/cpus), or GPU requests can be rejected.
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("SLURM_") and key != "SLURM_CONF":
            env.pop(key, None)

    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout}"
        )
    line = (proc.stdout or "").strip().splitlines()[0] if proc.stdout.strip() else ""
    if not line:
        raise RuntimeError("sbatch --parsable returned empty stdout")
    # Tolerate "jobid;cluster"
    job_id = line.split(";", 1)[0].strip()
    if not job_id:
        raise RuntimeError(f"Could not parse job id from sbatch output: {line!r}")
    return job_id


def _normalize_state(raw: str) -> str:
    return raw.strip().rstrip("+*").upper()


def _sacct_state_exit(job_id: str) -> tuple[str | None, str | None]:
    """Return (state, exit_code) from sacct, or (None, None) if unavailable."""
    proc = subprocess.run(
        [
            "sacct",
            "-j",
            job_id,
            "-n",
            "-X",
            "-o",
            "State,ExitCode",
            "-P",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None, None
    # First data line: STATE|ExitCode
    line = proc.stdout.strip().splitlines()[0]
    parts = line.split("|")
    if len(parts) < 2:
        return _normalize_state(parts[0]) if parts else None, None
    return _normalize_state(parts[0]), parts[1].strip()


def _squeue_state(job_id: str) -> str | None:
    proc = subprocess.run(
        ["squeue", "-j", job_id, "-h", "-o", "%T"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return _normalize_state(proc.stdout.strip().splitlines()[0])


def _tail_file(path: Path, n: int = 50) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def wait_for_job(
    job_id: str,
    *,
    poll_s: float = DEFAULT_POLL_S,
    timeout_s: float | None = None,
) -> tuple[str, str]:
    """Poll until a terminal SLURM state; return ``(state, exit_code)``.

    On ``timeout_s``, ``scancel`` the job and raise ``TimeoutError``.
    """
    if job_id == "DRY_RUN":
        return _SUCCESS_STATE, "0:0"

    t0 = time.monotonic()
    last_state = "UNKNOWN"
    last_exit = ""
    while True:
        state, exit_code = _sacct_state_exit(job_id)
        if state is None:
            sq = _squeue_state(job_id)
            if sq is not None:
                state = sq
                exit_code = exit_code or ""
            else:
                # Job vanished from both — treat as unknown failure after a beat
                state = last_state if last_state != "UNKNOWN" else "UNKNOWN"
                exit_code = last_exit

        if state:
            last_state = state
        if exit_code is not None:
            last_exit = exit_code

        if state == _SUCCESS_STATE:
            return state, exit_code or "0:0"
        if state in _FAILURE_STATES:
            return state, exit_code or ""

        if timeout_s is not None and (time.monotonic() - t0) >= timeout_s:
            subprocess.run(
                ["scancel", job_id],
                capture_output=True,
                text=True,
                check=False,
            )
            time.sleep(min(5.0, poll_s))
            raise TimeoutError(
                f"Timed out after {timeout_s}s waiting for job {job_id} "
                f"(last state={last_state}, exit={last_exit}); issued scancel"
            )

        time.sleep(poll_s)


def metric_from_out(
    out_path: Path | str,
    metric: str = DEFAULT_METRIC,
    mode: str = DEFAULT_MODE,
) -> float:
    """Parse train ``.out`` with ``consolidate_training_log``; aggregate metric."""
    out_path = Path(out_path)
    if not out_path.is_file():
        raise FileNotFoundError(f"Train log not found: {out_path}")
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

    df = consolidate_training_log(out_path)
    if df.empty:
        raise RuntimeError(f"no usable metrics in {out_path} (empty DataFrame)")
    if metric not in df.columns:
        raise RuntimeError(
            f"metric {metric!r} not in {out_path}; columns={list(df.columns)}"
        )
    series = df[metric].dropna()
    if series.empty:
        raise RuntimeError(f"metric {metric!r} all-NaN in {out_path}")
    value = float(series.min() if mode == "min" else series.max())
    return value


_OOM_MARKERS = (
    "cuda out of memory",
    "outofmemory",
    "out of memory",
    "oom",
)


def _looks_like_oom(*paths: Path) -> bool:
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(errors="replace").lower()
        if any(m in text for m in _OOM_MARKERS):
            return True
    return False


def _raise_job_failure(
    job_id: str,
    state: str,
    exit_code: str,
    job_name: str,
    run_id: str,
) -> None:
    err_path = find_train_err(job_name, run_id, job_id)
    out_path = find_train_out(job_name, run_id, job_id)
    tail = _tail_file(err_path)
    oom = state == "OUT_OF_MEMORY" or _looks_like_oom(err_path, out_path)
    prefix = "OOM: " if oom else ""
    msg = (
        f"{prefix}Train job {job_id} ended with state={state} exit_code={exit_code}"
    )
    if tail:
        msg += f"\n--- tail {err_path} ---\n{tail}"
    raise RuntimeError(msg)


def run_train_trial(
    config_relpath: str | Path,
    job_name: str,
    run_id: str,
    *,
    metric: str = DEFAULT_METRIC,
    mode: str = DEFAULT_MODE,
    poll_s: float = DEFAULT_POLL_S,
    timeout_s: float | None = None,
    dry_run: bool = False,
    skip_finished: bool = True,
    account: str = DEFAULT_ACCOUNT,
    cpus: int = DEFAULT_CPUS,
    mem: str = DEFAULT_MEM,
    time_limit: str = DEFAULT_TIME,
    constraint: str = DEFAULT_CONSTRAINT,
    gres: str = DEFAULT_GRES,
) -> TrialJobResult:
    """Submit → wait → parse; raise on failed / empty logs.

    If ``skip_finished``, reuse an existing parseable ``.out`` for this
    ``job_name``/``run_id`` instead of submitting again.
    """
    if skip_finished and not dry_run:
        reused = try_reuse_finished_trial(
            job_name, run_id, metric=metric, mode=mode
        )
        if reused is not None:
            print(
                f"skip_finished: reusing {reused.out_path} "
                f"(job_id={reused.job_id}, {metric}={reused.metric})"
            )
            return reused

    job_id = submit_train_job(
        config_relpath,
        job_name,
        run_id,
        account=account,
        cpus=cpus,
        mem=mem,
        time_limit=time_limit,
        constraint=constraint,
        gres=gres,
        dry_run=dry_run,
    )
    if dry_run:
        out_path = find_train_out(job_name, run_id, "JOBID")
        return TrialJobResult(
            job_id=job_id,
            out_path=out_path,
            metric=float("nan"),
            state="DRY_RUN",
            exit_code=None,
        )

    state, exit_code = wait_for_job(job_id, poll_s=poll_s, timeout_s=timeout_s)
    ok_exit = exit_code in ("0:0", "0", "")
    if state != _SUCCESS_STATE or not ok_exit:
        _raise_job_failure(job_id, state, exit_code, job_name, run_id)

    out_path = find_train_out(job_name, run_id, job_id)
    if not out_path.is_file():
        raise FileNotFoundError(
            f"Expected train log missing after COMPLETED: {out_path}"
        )
    try:
        value = metric_from_out(out_path, metric=metric, mode=mode)
    except RuntimeError as exc:
        if _looks_like_oom(out_path, find_train_err(job_name, run_id, job_id)):
            raise RuntimeError(f"OOM: {exc}") from exc
        raise
    return TrialJobResult(
        job_id=job_id,
        out_path=out_path,
        metric=value,
        state=state,
        exit_code=exit_code,
    )


def _add_resource_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--account", default=DEFAULT_ACCOUNT)
    p.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
    p.add_argument("--mem", default=DEFAULT_MEM)
    p.add_argument("--time", dest="time_limit", default=DEFAULT_TIME)
    p.add_argument("--constraint", default=DEFAULT_CONSTRAINT)
    p.add_argument("--gres", default=DEFAULT_GRES)
    p.add_argument("--poll-s", type=float, default=DEFAULT_POLL_S)
    p.add_argument("--timeout-s", type=float, default=None)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit/wait/parse BiaPy train SLURM jobs (repo root cwd)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    parse_p = sub.add_parser("parse", help="Parse metric from an existing .out")
    parse_p.add_argument("--out", required=True, type=Path, help="Train .out path")
    parse_p.add_argument("--metric", default=DEFAULT_METRIC)
    parse_p.add_argument("--mode", choices=("min", "max"), default=DEFAULT_MODE)

    run_p = sub.add_parser("run", help="Submit train, wait, parse metric")
    run_p.add_argument(
        "--config",
        required=True,
        help="Config path relative to biapy_work_folder/configs/",
    )
    run_p.add_argument("--job-name", required=True, help="BiaPy job name (base stem)")
    run_p.add_argument("--run-id", required=True)
    run_p.add_argument("--metric", default=DEFAULT_METRIC)
    run_p.add_argument("--mode", choices=("min", "max"), default=DEFAULT_MODE)
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch command; do not submit",
    )
    _add_resource_args(run_p)
    return parser


def _print_result(payload: dict) -> None:
    print(json.dumps(payload, indent=2, default=str))


def main(argv: Sequence[str] | None = None) -> None:
    # CLI always operates from repo root (relative configs / sbatch paths).
    import os

    os.chdir(_REPO_ROOT)
    args = _build_argparser().parse_args(argv)
    if args.command == "parse":
        value = metric_from_out(args.out, metric=args.metric, mode=args.mode)
        _print_result(
            {
                "out_path": str(args.out),
                "metric_name": args.metric,
                "mode": args.mode,
                "metric": value,
            }
        )
        return

    if args.command == "run":
        result = run_train_trial(
            args.config,
            args.job_name,
            args.run_id,
            metric=args.metric,
            mode=args.mode,
            poll_s=args.poll_s,
            timeout_s=args.timeout_s,
            dry_run=args.dry_run,
            account=args.account,
            cpus=args.cpus,
            mem=args.mem,
            time_limit=args.time_limit,
            constraint=args.constraint,
            gres=args.gres,
        )
        _print_result(asdict(result) | {"metric_name": args.metric, "mode": args.mode})
        return

    raise ValueError(f"Unknown command {args.command!r}")


if __name__ == "__main__":
    main()
