"""Phase B checks: config auto-append, exclusions, regen registration, bin norm.

Example
-------
    singularity exec --overlay env/BiaPy_env.ext3:ro \\
      /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \\
      /bin/bash -c 'source /ext3/env.sh; conda activate BiaPy_env; \\
      export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}; \\
      python biapy_work_folder/check_skeleton_phase_b.py'
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "BiaPy-novabro"))

from biapy.config import Config  # noqa: E402
from biapy.data.norm import normalize_mask  # noqa: E402
from biapy.data.pre_processing.channel_layout import (  # noqa: E402
    channel_physical_offsets,
    instance_channel_needs_regen,
)
from biapy.data.pre_processing.labels_to_channels import labels_into_channels  # noqa: E402
from biapy.engine.check_configuration import check_configuration  # noqa: E402


def _minimal_cfg(
    data_channels,
    extra_opts,
    *,
    enable_skel: bool = False,
    target_channel: str = "F",
    ndim: str = "3D",
    weights=(1.0, 2.0),
):
    """Build a clone of BiaPy defaults with enough fields for check_configuration."""
    cfg_obj = Config(job_dir="/tmp/biapy_sk_phase_b", job_identifier="sk_phase_b")
    cfg = cfg_obj.get_cfg_defaults()
    cfg.defrost()
    cfg.PROBLEM.TYPE = "INSTANCE_SEG"
    cfg.PROBLEM.NDIM = ndim
    cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS = list(data_channels)
    cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS = [dict(extra_opts)]
    cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS = weights
    cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_LOSSES = []
    cfg.LOSS.SKELETON_RECALL.ENABLE = enable_skel
    cfg.LOSS.SKELETON_RECALL.TARGET_CHANNEL = target_channel
    cfg.LOSS.SKELETON_RECALL.TUBE_DILATIONS = 1
    cfg.TRAIN.ENABLE = True
    cfg.TEST.ENABLE = False
    cfg.DATA.PATCH_SIZE = (80, 128, 128, 1) if ndim == "3D" else (128, 128, 1)
    cfg.DATA.TRAIN.PATH = "/tmp/biapy_sk_phase_b/train/raw"
    cfg.DATA.TRAIN.GT_PATH = "/tmp/biapy_sk_phase_b/train/label"
    cfg.DATA.VAL.PATH = "/tmp/biapy_sk_phase_b/val/raw"
    cfg.DATA.VAL.GT_PATH = "/tmp/biapy_sk_phase_b/val/label"
    cfg.DATA.VAL.FROM_TRAIN = False
    cfg.DATA.TEST.PATH = "/tmp/biapy_sk_phase_b/test/raw"
    cfg.DATA.TEST.GT_PATH = "/tmp/biapy_sk_phase_b/test/label"
    cfg.MODEL.ARCHITECTURE = "unet"
    cfg.MODEL.FEATURE_MAPS = [16, 32, 64, 128]  # 3 downsample levels; patch divisible by 8
    cfg.MODEL.Z_DOWN = [1, 1, 1]
    cfg.freeze()
    return cfg


def _check_config_plumbing() -> None:
    cfg = _minimal_cfg(
        ["F", "Db"],
        {"Db": {"mask_values": False}},
        enable_skel=True,
        weights=(1.0, 2.0),
    )
    check_configuration(cfg, "sk_phase_b", check_data_paths=False)
    chs = list(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS)
    assert "Sk" in chs, f"Sk not auto-appended: {chs}"
    assert "I" in chs, f"I not auto-appended for Sk regen: {chs}"
    assert chs.index("Sk") < chs.index("I"), f"I must stay last: {chs}"
    extra = cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS[0]
    assert extra.get("Sk", {}).get("dilation") == 1, f"unexpected Sk opts: {extra.get('Sk')}"
    losses = list(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_LOSSES)
    assert len(losses) == 2, f"losses should ignore Sk/I, got {losses}"
    weights = list(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS)
    assert len(weights) == 2, f"weights should ignore Sk/I, got {weights}"
    print(f"  [PASS] config plumbing: channels={chs} losses={losses} Sk.dilation={extra['Sk']['dilation']}")


def _check_b3_raises() -> None:
    cfg = _minimal_cfg(
        ["Db"],
        {"Db": {"mask_values": False}},
        enable_skel=True,
        target_channel="F",
        weights=(1.0,),
    )
    try:
        check_configuration(cfg, "sk_phase_b_b3", check_data_paths=False)
    except AssertionError as e:
        assert "TARGET_CHANNEL" in str(e) or "Skeleton Recall" in str(e) or "F" in str(e)
        print(f"  [PASS] B3 guard raises without F/M: {e}")
        return
    raise AssertionError("expected AssertionError when ENABLE without F/M")


def _check_needs_regen() -> None:
    assert instance_channel_needs_regen("Sk", {}) is True
    assert instance_channel_needs_regen("F", {}) is False
    print("  [PASS] instance_channel_needs_regen('Sk') is True")


def _check_regen_mask_map_logic() -> None:
    """Simulate the generator's regen_mask_map construction for F+Db+Sk+I."""
    mode = ["F", "Db", "Sk", "I"]
    extra = {"Db": {"mask_values": False, "val_type": "norm", "act": ""}, "Sk": {"dilation": 1}, "F": {"erosion": 0, "dilation": 0}}
    # Synthetic mask channels: F, Db, Sk binary-ish, I labels — match normalize_mask tagging.
    rng = np.random.default_rng(0)
    z, y, x = 8, 16, 16
    labels = np.zeros((z, y, x, 1), dtype=np.int32)
    labels[2:6, 4:12, 4:12, 0] = 1
    labels[2:6, 4:8, 8:12, 0] = 2
    full = labels_into_channels(labels, mode=mode, channel_extra_opts=extra)
    # Build a fake per_channel_info like the generator: bin for F/Sk/I-as-label, no_bin for Db.
    per = {
        0: {"type": "bin"},
        1: {"type": "no_bin"},
        2: {"type": "bin"},
        3: {"type": "label"},
    }
    offsets = channel_physical_offsets(mode, extra)
    col = offsets["Sk"]
    mpos = sum(1 for k in range(col) if per[k]["type"] not in ("no_bin", "flow"))
    assert col == 2, f"Sk physical col expected 2, got {col}"
    assert mpos == 1, f"Sk mask-group pos expected 1 (F before it; Db in heat), got {mpos}"
    # Regeneration replaces mask[..., mpos] from regen[..., col]
    regen = labels_into_channels(labels, mode=mode, channel_extra_opts=extra)
    assert np.array_equal(full[..., col] > 0, regen[..., col] > 0)
    print(f"  [PASS] regen_mask_map logic: Sk col={col} mpos={mpos}")


def _check_bin_norm() -> None:
    labels = np.zeros((8, 16, 16, 1), dtype=np.uint16)
    labels[2:6, 4:12, 4:12, 0] = 1
    mask = labels_into_channels(
        labels,
        mode=["F", "Db", "Sk"],
        channel_extra_opts={"Db": {"mask_values": False}, "Sk": {"dilation": 1}},
    )
    sk_idx = 2
    assert np.unique(mask[..., sk_idx]).size <= 2
    nm = {"target_type": "mask"}
    _, info = normalize_mask(mask, norm_module=nm, instance_problem=True, apply_norm=False)
    sk_type = info["per_channel_info"][sk_idx]["type"]
    assert sk_type == "bin", f"Sk should be 'bin', got {sk_type!r} ({info['per_channel_info']})"
    print("  [PASS] normalize_mask tags Sk as 'bin'")


def main() -> None:
    print("Phase B remaining checks:")
    _check_needs_regen()
    _check_config_plumbing()
    _check_b3_raises()
    _check_regen_mask_map_logic()
    _check_bin_norm()
    print("\nAll Phase B checks passed.")


if __name__ == "__main__":
    main()
