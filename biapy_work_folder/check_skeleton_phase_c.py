"""Phase C checks: Sk workflow skips, multiple_metrics filter, loss wiring.

Example
-------
    singularity exec --overlay env/BiaPy_env.ext3:ro \\
      /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \\
      /bin/bash -c 'source /ext3/env.sh; conda activate BiaPy_env; \\
      export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}; \\
      python biapy_work_folder/check_skeleton_phase_c.py'
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "BiaPy-novabro"))

from biapy.config import Config  # noqa: E402
from biapy.engine.check_configuration import check_configuration  # noqa: E402
from biapy.engine.metrics import instance_segmentation_loss, multiple_metrics  # noqa: E402

INSTANCE_SEG_PATH = REPO_ROOT / "BiaPy-novabro" / "biapy" / "engine" / "instance_seg.py"


def _minimal_cfg(enable_skel: bool = True):
    cfg_obj = Config(job_dir="/tmp/biapy_sk_phase_c", job_identifier="sk_phase_c")
    cfg = cfg_obj.get_cfg_defaults()
    cfg.defrost()
    cfg.PROBLEM.TYPE = "INSTANCE_SEG"
    cfg.PROBLEM.NDIM = "3D"
    cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS = ["F", "Db"]
    cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS = [{"Db": {"mask_values": False}}]
    cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS = (1.0, 2.0)
    cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_LOSSES = []
    cfg.LOSS.SKELETON_RECALL.ENABLE = enable_skel
    cfg.LOSS.SKELETON_RECALL.WEIGHT = 0.5
    cfg.LOSS.SKELETON_RECALL.TARGET_CHANNEL = "F"
    cfg.LOSS.SKELETON_RECALL.TUBE_DILATIONS = 1
    cfg.TRAIN.ENABLE = True
    cfg.TEST.ENABLE = False
    cfg.DATA.PATCH_SIZE = (80, 128, 128, 1)
    cfg.DATA.TRAIN.PATH = "/tmp/biapy_sk_phase_c/train/raw"
    cfg.DATA.TRAIN.GT_PATH = "/tmp/biapy_sk_phase_c/train/label"
    cfg.DATA.VAL.PATH = "/tmp/biapy_sk_phase_c/val/raw"
    cfg.DATA.VAL.GT_PATH = "/tmp/biapy_sk_phase_c/val/label"
    cfg.DATA.VAL.FROM_TRAIN = False
    cfg.DATA.TEST.PATH = "/tmp/biapy_sk_phase_c/test/raw"
    cfg.DATA.TEST.GT_PATH = "/tmp/biapy_sk_phase_c/test/label"
    cfg.MODEL.ARCHITECTURE = "unet"
    cfg.MODEL.FEATURE_MAPS = [16, 32, 64, 128]
    cfg.MODEL.Z_DOWN = [1, 1, 1]
    cfg.freeze()
    return cfg


def _check_skip_sites() -> None:
    text = INSTANCE_SEG_PATH.read_text()
    # Head routing list
    assert re.search(r'channel in \["I", "We", "Sk"\]', text), "head routing missing Sk"
    # Activations continue
    assert re.search(r'elif channel == "Sk":\s*\n\s*#.*\n\s*continue', text) or re.search(
        r'elif channel == "Sk":\s*\n\s*continue', text
    ), "activations missing Sk continue"
    # Metrics continue
    assert 'elif channel == "Sk":' in text and "define_metrics" in text
    # Count Sk continues / list membership in the three known regions
    sk_guards = len(re.findall(r'"Sk"', text))
    assert sk_guards >= 4, f"expected Sk mentioned in skips + loss kwargs, found {sk_guards}"
    assert "skeleton_recall_weight=self.cfg.LOSS.SKELETON_RECALL.WEIGHT" in text
    assert "skeleton_target_channel=self.cfg.LOSS.SKELETON_RECALL.TARGET_CHANNEL" in text
    print("  [PASS] instance_seg.py skip sites + loss kwargs present")


def _check_multiple_metrics_filter() -> None:
    m = multiple_metrics(
        num_classes=1,
        metric_names=["IoU (F channel)", "L1 (Db channel)"],
        device=torch.device("cpu"),
        out_channels=["F", "Db", "Sk"],
        ndim=2,
    )
    assert m.out_channels == ["F", "Db"], f"expected ['F','Db'], got {m.out_channels}"
    print(f"  [PASS] multiple_metrics filters Sk: {m.out_channels}")


def _check_loss_wiring() -> None:
    cfg = _minimal_cfg(enable_skel=True)
    check_configuration(cfg, "sk_phase_c", check_data_paths=False)

    # Predicted GT width: F + Db (I dropped by generator; Sk is aux counted inside the loss).
    pred_channels = [c for c in cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS if c not in ("We", "I", "Sk")]
    gt_channels_expected = len(pred_channels)

    loss_fn = instance_segmentation_loss(
        channel_weights=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS,
        ndim=3,
        out_channels=list(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS),
        losses_to_use=list(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_LOSSES),
        channel_extra_opts=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS[0],
        gt_channels_expected=gt_channels_expected,
        skeleton_recall_weight=cfg.LOSS.SKELETON_RECALL.WEIGHT,
        skeleton_target_channel=cfg.LOSS.SKELETON_RECALL.TARGET_CHANNEL,
        device=torch.device("cpu"),
    )
    assert loss_fn.use_skeleton_recall
    assert loss_fn.skeleton_recall_weight == 0.5
    assert loss_fn.skeleton_target_channel == "F"
    assert loss_fn.gt_channels_expected == 3  # F, Db, Sk (I never reaches loss)

    torch.manual_seed(0)
    y_pred = torch.randn(2, 2, 8, 16, 16, requires_grad=True)
    y_true = torch.zeros(2, 3, 8, 16, 16)
    y_true[:, 0, 2:6, 4:12, 4:12] = 1.0
    y_true[:, 1, 2:6, 4:12, 4:12] = 0.4
    y_true[:, 2, 3:5, 6:10, 6:10] = 1.0  # Sk
    v = loss_fn(y_pred, y_true)
    assert torch.isfinite(v), f"loss not finite: {v}"
    v.backward()
    assert y_pred.grad is not None and torch.any(y_pred.grad != 0)
    print(
        f"  [PASS] loss wiring: weight={loss_fn.skeleton_recall_weight} "
        f"target={loss_fn.skeleton_target_channel} loss={v.item():.6f}"
    )


def main() -> None:
    print("Phase C checks:")
    _check_skip_sites()
    _check_multiple_metrics_filter()
    _check_loss_wiring()
    print("\nAll Phase C checks passed.")


if __name__ == "__main__":
    main()
