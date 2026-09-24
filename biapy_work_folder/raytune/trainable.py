"""Ray Tune trainable wrapping Stage 1 YAML write + Stage 2 SLURM train.

Ray is imported lazily so ``import biapy_work_folder.raytune.write_trial_config``
never requires Ray. Install Ray in the controller env (Stage 5) before running
``run_tune.py``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
    run_train_trial,
)
from biapy_work_folder.raytune.write_trial_config import (
    HPARAM_MAP,
    _VIRTUAL_HPARAMS,
    sanitize_trial_id,
    write_trial_config,
)

_RAY_INSTALL_HINT = (
    "Ray is required for Stage 3 Tune trials but is not installed. "
    "Install into the controller env (Stage 5), e.g. "
    '`pip install "ray[tune]"` inside the Singularity overlay.'
)


def _require_ray_tune():
    try:
        from ray import tune
    except ImportError as exc:
        raise ImportError(_RAY_INSTALL_HINT) from exc
    return tune


def _ensure_repo_root() -> None:
    os.chdir(_REPO_ROOT)


def _default_job_name(base_yaml: str | Path) -> str:
    stem = Path(base_yaml).name
    if stem.endswith(".yaml"):
        stem = stem[: -len(".yaml")]
    return stem


def _trial_id() -> str:
    tune = _require_ray_tune()
    # Ray 2.7+: tune.get_context(); older: session / trial_id via air
    get_context = getattr(tune, "get_context", None)
    if get_context is not None:
        ctx = get_context()
        get_trial_id = getattr(ctx, "get_trial_id", None)
        if get_trial_id is not None:
            return str(get_trial_id())
    try:
        from ray.train import get_context as train_get_context

        return str(train_get_context().get_trial_id())
    except Exception:
        pass
    return "local_trial"


def make_biapy_sbatch_trainable(
    *,
    base_yaml: str | Path = "biapy-aug-zarr-seunet-FDb-skel",
    job_name: str | None = None,
    metric: str = DEFAULT_METRIC,
    mode: str = DEFAULT_MODE,
    search_epochs: int | None = 40,
    search_patience: int | None = 10,
    dry_run: bool = False,
    skip_finished: bool = True,
    poll_s: float = DEFAULT_POLL_S,
    timeout_s: float | None = None,
    account: str = DEFAULT_ACCOUNT,
    cpus: int = DEFAULT_CPUS,
    mem: str = DEFAULT_MEM,
    time_limit: str = DEFAULT_TIME,
    constraint: str = DEFAULT_CONSTRAINT,
    gres: str = DEFAULT_GRES,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a Ray Tune objective: hparam config -> metric dict.

    ``config`` should only contain keys from ``HPARAM_MAP``; extras are ignored.
    Nested GPU work runs via ``run_train_trial`` (Stage 2); this callable itself
    should be wrapped with CPU-only resources (see ``run_tune.py``).
    """
    resolved_job = job_name or _default_job_name(base_yaml)

    def biapy_sbatch_trainable(config: dict[str, Any]) -> dict[str, Any]:
        _ensure_repo_root()
        run_id = sanitize_trial_id(_trial_id())
        allowed = set(HPARAM_MAP) | set(_VIRTUAL_HPARAMS)
        hparams = {k: v for k, v in config.items() if k in allowed}
        rel = write_trial_config(
            base_yaml,
            hparams,
            run_id,
            search_epochs=search_epochs,
            search_patience=search_patience,
            overwrite=False,
        )
        result = run_train_trial(
            rel,
            resolved_job,
            run_id,
            metric=metric,
            mode=mode,
            poll_s=poll_s,
            timeout_s=timeout_s,
            dry_run=dry_run,
            skip_finished=skip_finished,
            account=account,
            cpus=cpus,
            mem=mem,
            time_limit=time_limit,
            constraint=constraint,
            gres=gres,
        )
        return {
            metric: result.metric,
            "slurm_job_id": result.job_id,
            "out_path": str(result.out_path),
            "run_id": run_id,
            "config_relpath": rel.as_posix(),
            "state": result.state,
        }

    return biapy_sbatch_trainable
