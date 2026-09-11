"""Phase D checks: load FDb-skel YAML and confirm Skeleton Recall plumbing.

Example
-------
    singularity exec --overlay env/BiaPy_env.ext3:ro \\
      /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \\
      /bin/bash -c 'source /ext3/env.sh; conda activate BiaPy_env; \\
      export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}; \\
      python biapy_work_folder/check_skeleton_phase_d.py'
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from yacs.config import CfgNode as CN

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "BiaPy-novabro"))

from biapy.config import Config  # noqa: E402
from biapy.engine.check_configuration import (  # noqa: E402
    check_configuration,
    convert_old_model_cfg_to_current_version,
)

YAML_PATH = REPO_ROOT / "biapy_work_folder" / "configs" / "biapy-aug-zarr-seunet-FDb-skel.yaml"


def _load_yaml_cfg(path: Path):
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg_manager = Config(job_dir="/tmp/biapy_sk_phase_d", job_identifier="sk_phase_d")
    temp_cfg = CN(convert_old_model_cfg_to_current_version(raw))
    cfg_manager._C.merge_from_other_cfg(temp_cfg)
    return cfg_manager.get_cfg_defaults()


def _check_yaml_plumbing() -> None:
    assert YAML_PATH.is_file(), f"missing config: {YAML_PATH}"
    cfg = _load_yaml_cfg(YAML_PATH)
    assert cfg.LOSS.SKELETON_RECALL.ENABLE is True
    assert cfg.LOSS.SKELETON_RECALL.WEIGHT == 1.0
    assert cfg.LOSS.SKELETON_RECALL.TUBE_DILATIONS == 1
    assert cfg.LOSS.SKELETON_RECALL.TARGET_CHANNEL == "F"

    check_configuration(cfg, "sk_phase_d", check_data_paths=False)
    chs = list(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS)
    assert "Sk" in chs and "I" in chs, f"expected Sk+I auto-append, got {chs}"
    assert chs.index("Sk") < chs.index("I")
    extra = cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS[0]
    assert extra.get("Sk", {}).get("dilation") == 1, f"Sk opts: {extra.get('Sk')}"
    losses = list(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_LOSSES)
    weights = list(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS)
    assert len(losses) == 2, f"losses should ignore Sk/I: {losses}"
    assert len(weights) == 2, f"weights should ignore Sk/I: {weights}"
    print(
        f"  [PASS] YAML plumbing: ENABLE={cfg.LOSS.SKELETON_RECALL.ENABLE} "
        f"channels={chs} losses={losses} Sk.dilation={extra['Sk']['dilation']}"
    )


def _check_train_track_aliases() -> None:
    sys.path.insert(0, str(REPO_ROOT / "biapy_work_folder"))
    from biapy_train_track import _KNOWN_METRIC_ALIASES, _normalize_metric_name  # noqa: WPS433

    assert _KNOWN_METRIC_ALIASES.get("skeleton recall") == "skel_recall"
    assert _normalize_metric_name("Skeleton Recall") == "skel_recall"
    print("  [PASS] train-track alias: 'Skeleton Recall' -> skel_recall")


def main() -> None:
    print("Phase D checks:")
    _check_yaml_plumbing()
    _check_train_track_aliases()
    print("\nAll Phase D checks passed.")
    print(
        "Launch (when ready):\n"
        "  ./sbatch/biapy/biapy-py_sbatch_chain.sh biapy-aug-zarr-seunet-FDb-skel preprocessing train test"
    )


if __name__ == "__main__":
    main()
