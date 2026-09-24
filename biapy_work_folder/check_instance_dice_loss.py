"""Unit checks for Stage A instance-seg soft Dice loss.

Runs on CPU; does not load FISBe volumes or touch training. Imports the local
BiaPy-novabro checkout so the new ``dice`` path is found without an install.

Example
-------
    # Prefer conda's libstdc++ so scipy/skimage C++ extensions resolve inside the SIF:
    singularity exec --overlay env/BiaPy_env.ext3:ro \\
      /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \\
      /bin/bash -c 'source /ext3/env.sh; conda activate BiaPy_env; \\
      export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}; \\
      python biapy_work_folder/check_instance_dice_loss.py'
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "BiaPy-novabro"))

from biapy.config import Config  # noqa: E402
from biapy.engine.check_configuration import check_configuration  # noqa: E402
from biapy.engine.metrics import instance_segmentation_loss  # noqa: E402


def _minimal_cfg(data_channels, losses, extra_opts, weights):
    """Build a clone of BiaPy defaults with enough fields for check_configuration."""
    cfg_obj = Config(job_dir="/tmp/biapy_dice_stage_a", job_identifier="dice_a")
    cfg = cfg_obj.get_cfg_defaults()
    cfg.defrost()
    cfg.PROBLEM.TYPE = "INSTANCE_SEG"
    cfg.PROBLEM.NDIM = "3D"
    cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS = list(data_channels)
    cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS = [dict(extra_opts)]
    cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNEL_WEIGHTS = weights
    cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_LOSSES = list(losses)
    cfg.TRAIN.ENABLE = True
    cfg.TEST.ENABLE = False
    cfg.DATA.PATCH_SIZE = (80, 128, 128, 1)
    cfg.DATA.TRAIN.PATH = "/tmp/biapy_dice_stage_a/train/raw"
    cfg.DATA.TRAIN.GT_PATH = "/tmp/biapy_dice_stage_a/train/label"
    cfg.DATA.VAL.PATH = "/tmp/biapy_dice_stage_a/val/raw"
    cfg.DATA.VAL.GT_PATH = "/tmp/biapy_dice_stage_a/val/label"
    cfg.DATA.VAL.FROM_TRAIN = False
    cfg.DATA.TEST.PATH = "/tmp/biapy_dice_stage_a/test/raw"
    cfg.DATA.TEST.GT_PATH = "/tmp/biapy_dice_stage_a/test/label"
    cfg.MODEL.ARCHITECTURE = "unet"
    cfg.MODEL.FEATURE_MAPS = [16, 32, 64, 128]
    cfg.MODEL.Z_DOWN = [1, 1, 1]
    cfg.freeze()
    return cfg


def _check_config_allows_dice_on_f() -> None:
    cfg = _minimal_cfg(
        ["F", "Db"],
        ["dice", "l1"],
        {"Db": {"mask_values": False}},
        (1.0, 1.0),
    )
    check_configuration(cfg, "dice_a_ok", check_data_paths=False)
    losses = list(cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_LOSSES)
    assert losses == ["dice", "l1"], f"unexpected losses after check: {losses}"
    print(f"  [PASS] config allows dice on F: losses={losses}")


def _check_config_rejects_dice_on_db() -> None:
    cfg = _minimal_cfg(
        ["F", "Db"],
        ["bce", "dice"],
        {"Db": {"mask_values": False}},
        (1.0, 1.0),
    )
    try:
        check_configuration(cfg, "dice_a_bad", check_data_paths=False)
    except ValueError as exc:
        msg = str(exc)
        assert "dice" in msg.lower() and "Db" in msg, f"unexpected error: {msg}"
        print(f"  [PASS] config rejects dice on Db: {msg.splitlines()[0]}")
        return
    raise AssertionError("expected ValueError when dice is paired with Db")


def _make_loss() -> instance_segmentation_loss:
    return instance_segmentation_loss(
        channel_weights=(1.0, 1.0),
        ndim=3,
        out_channels=["F", "Db"],
        losses_to_use=["dice", "l1"],
        channel_extra_opts={"Db": {"mask_values": False}},
        gt_channels_expected=2,
    )


def _check_forward_backward() -> None:
    """Random 3D batch: finite scalar loss and nonzero grad on F logits."""
    loss_fn = _make_loss()
    # Pred / GT: (B, C=2, D, H, W)
    pred = torch.randn(2, 2, 8, 16, 16, requires_grad=True)
    gt = torch.zeros(2, 2, 8, 16, 16)
    gt[:, 0, 2:6, 4:12, 4:12] = 1.0  # F
    gt[:, 1] = torch.rand(2, 8, 16, 16) * gt[:, 0]  # Db sparse on FG
    loss = loss_fn(pred, gt)
    assert loss.ndim == 0 and torch.isfinite(loss), f"bad loss: {loss}"
    loss.backward()
    assert pred.grad is not None, "pred.grad is None"
    assert torch.any(pred.grad[:, 0] != 0), "expected nonzero grad on F channel"
    print(f"  [PASS] forward/backward: loss={loss.item():.6f}")


def _check_overlap_ordering() -> None:
    """Large positive F logits on FG -> lower Dice than all-negative F logits."""
    loss_fn = _make_loss()
    gt = torch.zeros(1, 2, 8, 16, 16)
    gt[:, 0, 2:6, 4:12, 4:12] = 1.0
    gt[:, 1] = 0.0

    # Keep Db logits at 0 so L1 does not dominate; only F Dice differs.
    pred_good = torch.zeros(1, 2, 8, 16, 16)
    pred_good[:, 0] = -10.0
    pred_good[:, 0][gt[:, 0] == 1] = 20.0

    pred_bad = torch.zeros(1, 2, 8, 16, 16)
    pred_bad[:, 0] = -20.0

    loss_good = loss_fn(pred_good, gt)
    loss_bad = loss_fn(pred_bad, gt)
    assert torch.isfinite(loss_good) and torch.isfinite(loss_bad)
    assert loss_good.item() < loss_bad.item(), (
        f"good overlap ({loss_good.item():.6f}) should be < bad ({loss_bad.item():.6f})"
    )
    # Pure Dice on a well-covered FG should be near 0 (Db L1 is ~0 here).
    assert loss_good.item() < 0.2, f"expected near-zero good loss, got {loss_good.item():.6f}"
    print(
        f"  [PASS] overlap ordering: good={loss_good.item():.6f} < bad={loss_bad.item():.6f}"
    )


def main() -> None:
    print("Instance-seg Stage A Dice checks:")
    _check_config_allows_dice_on_f()
    _check_config_rejects_dice_on_db()
    _check_forward_backward()
    _check_overlap_ordering()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
