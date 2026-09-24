"""Ray Tune helpers for BiaPy YAML trial configs and nested SLURM trials.

``make_biapy_sbatch_trainable`` / ``build_search_space`` do not import Ray until
used. Prefer ``python biapy_work_folder/raytune/run_tune.py`` for fits.
"""

from .search_space import build_search_space, constraint_for_gpu, mem_for_gpu
from .slurm_trial import (
    TrialJobResult,
    find_train_out,
    metric_from_out,
    run_train_trial,
    submit_train_job,
    wait_for_job,
)
from .trainable import make_biapy_sbatch_trainable
from .winner import write_winner_config
from .write_trial_config import sanitize_trial_id, write_trial_config

__all__ = [
    "TrialJobResult",
    "build_search_space",
    "constraint_for_gpu",
    "find_train_out",
    "make_biapy_sbatch_trainable",
    "mem_for_gpu",
    "metric_from_out",
    "run_train_trial",
    "sanitize_trial_id",
    "submit_train_job",
    "wait_for_job",
    "write_trial_config",
    "write_winner_config",
]
