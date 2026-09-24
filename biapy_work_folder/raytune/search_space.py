"""GPU-conditioned Ray Tune search spaces for BiaPy (Stage 6).

``build_search_space`` imports ``ray.tune`` lazily so importing this module
without Ray installed still works for ``constraint_for_gpu``.

Presets
-------
``legacy``
    Historical space: ``lr``, ``batch_size``, ``scheduler`` choice, optional
    ``skel_weight``. Kept so in-flight / resumed Optuna runs keep the same
    dimensions.

``wc_nopat``
    Dice-vs-skel plan (``md_guides/raytune_dice_vs_skel_fdb.md``): fixed
    ``warmupcosine``, no ``batch_size`` search, ``warmup_cosine_epochs`` capped
    by ``search_epochs``, plus ``skel_weight`` and/or ``f_weight``.
"""

from __future__ import annotations

from typing import Any, Literal

SpacePreset = Literal["legacy", "wc_nopat"]
SPACE_PRESETS: tuple[str, ...] = ("legacy", "wc_nopat")


def constraint_for_gpu(gpu: str) -> str:
    """Map a logical GPU label to a SLURM ``--constraint`` value."""
    key = gpu.strip().lower()
    if key == "h200":
        return "h200"
    if key in ("l40", "l40s"):
        return "l40s"
    raise ValueError(
        f"Unknown gpu {gpu!r}; expected 'h200', 'l40', or 'l40s'"
    )


def mem_for_gpu(gpu: str) -> str:
    """Default ``--mem`` for nested train jobs on this GPU type.

    L40S public nodes reject 384g (``GPU job setup is not valid``); 250g fits.
    H200 keeps the historical 384g train footprint from ``biapy-py_sbatch_chain``.
    """
    key = gpu.strip().lower()
    if key == "h200":
        return "384g"
    if key in ("l40", "l40s"):
        return "250g"
    raise ValueError(
        f"Unknown gpu {gpu!r}; expected 'h200', 'l40', or 'l40s'"
    )


def _lr_for_gpu(tune: Any, gpu: str) -> Any:
    key = gpu.strip().lower()
    if key == "h200":
        return tune.loguniform(3e-4, 3e-3)
    if key in ("l40", "l40s"):
        return tune.loguniform(1e-4, 1e-3)
    raise ValueError(
        f"Unknown gpu {gpu!r}; expected 'h200', 'l40', or 'l40s'"
    )


def _batch_size_for_gpu(tune: Any, gpu: str) -> Any:
    key = gpu.strip().lower()
    if key == "h200":
        return tune.choice([24, 30])
    if key in ("l40", "l40s"):
        return tune.choice([12, 15])
    raise ValueError(
        f"Unknown gpu {gpu!r}; expected 'h200', 'l40', or 'l40s'"
    )


def build_search_space(
    *,
    gpu: str = "h200",
    include_skel_weight: bool = True,
    include_f_weight: bool = False,
    search_epochs: int | None = 40,
    preset: SpacePreset | str = "legacy",
) -> dict[str, Any]:
    """Return an Optuna/Ray Tune param space for BiaPy train sweeps.

    Keys are a subset of ``HPARAM_MAP`` in ``write_trial_config``, plus the
    virtual key ``f_weight`` (converted to ``channel_weights`` on write).

    ``skel_weight`` is included only when ``include_skel_weight`` is True.
    ``f_weight`` is included only when ``include_f_weight`` is True (dice plan).
    """
    try:
        from ray import tune
    except ImportError as exc:
        raise ImportError(
            "Ray is required to build a Tune search space. "
            "Install into BiaPy_env (Stage 5): pip install \"ray[tune]\" optuna"
        ) from exc

    key = preset.strip().lower()
    if key not in SPACE_PRESETS:
        raise ValueError(
            f"Unknown space preset {preset!r}; expected one of {SPACE_PRESETS}"
        )

    if key == "legacy":
        space: dict[str, Any] = {
            "lr": _lr_for_gpu(tune, gpu),
            "batch_size": _batch_size_for_gpu(tune, gpu),
            "scheduler": tune.choice(["warmupcosine", "onecycle"]),
        }
        if include_skel_weight:
            space["skel_weight"] = tune.uniform(0.05, 0.3)
        return space

    # preset == "wc_nopat"
    epochs = 40 if search_epochs is None else int(search_epochs)
    if epochs < 3:
        raise ValueError(
            f"wc_nopat requires search_epochs >= 3 (got {epochs}); "
            "warmup_cosine_epochs needs a non-empty integer range"
        )
    # tune.randint upper bound is exclusive → inclusive max is ``epochs``.
    warmup_hi = epochs + 1
    space = {
        "lr": _lr_for_gpu(tune, gpu),
        "scheduler": "warmupcosine",
        "warmup_cosine_epochs": tune.randint(3, warmup_hi),
    }
    if include_skel_weight:
        space["skel_weight"] = tune.uniform(0.05, 0.3)
    if include_f_weight:
        # Converted to DATA_CHANNEL_WEIGHTS=(f_weight, 2.0) in apply_hparams.
        space["f_weight"] = tune.uniform(0.5, 2.0)
    return space
