"""Export full-epoch winner YAML after a Ray Tune sweep (Stage 7)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from biapy_work_folder.raytune.write_trial_config import (
    CONFIG_DIR,
    HPARAM_MAP,
    _VIRTUAL_HPARAMS,
    resolve_base_yaml,
    sanitize_trial_id,
    write_trial_config,
)


def chain_command(stem: str, run_id: str) -> str:
    """Shell command to retrain+test the winner with the normal chain script."""
    return (
        f"./sbatch/biapy/biapy-py_sbatch_chain.sh "
        f"-r {run_id} {stem} train test"
    )


def write_winner_config(
    *,
    base_yaml: str | Path,
    hparams: dict[str, Any],
    exp: str,
    run_id: str | None = None,
    search_epochs: int | None = None,
    search_patience: int | None = None,
) -> tuple[Path, str]:
    """Write full-epoch winner YAML; return (rel path under configs/, chain cmd).

    ``search_epochs`` / ``search_patience`` default to None so base EPOCHS /
    PATIENCE are kept (no search budget cap).
    """
    allowed = set(HPARAM_MAP) | set(_VIRTUAL_HPARAMS)
    filtered = {k: v for k, v in hparams.items() if k in allowed}
    winner_id = sanitize_trial_id(run_id or f"{sanitize_trial_id(exp)}_winner")
    rel = write_trial_config(
        base_yaml,
        filtered,
        winner_id,
        search_epochs=search_epochs,
        search_patience=search_patience,
        overwrite=True,
    )
    base_path = resolve_base_yaml(base_yaml)
    stem = base_path.stem
    cmd = chain_command(stem, winner_id)
    print(f"winner config: {CONFIG_DIR / rel}")
    print(f"retrain with: {cmd}")
    return rel, cmd
