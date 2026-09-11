"""CPU checks for Phase A2: SoftSkeletonRecall wired into instance_segmentation_loss.

Constructs ``instance_segmentation_loss`` directly (no config / training). Imports the
local BiaPy-novabro checkout.

Example
-------
    singularity exec --overlay env/BiaPy_env.ext3:ro \\
      /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \\
      /bin/bash -c 'source /ext3/env.sh; conda activate BiaPy_env; \\
      export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}; \\
      python biapy_work_folder/check_instance_skel_loss.py'
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "BiaPy-novabro"))

from biapy.engine.metrics import instance_segmentation_loss  # noqa: E402


def _base_batch(h: int = 16, w: int = 16):
    """Shared F/Db prediction and GT (no aux channels)."""
    torch.manual_seed(0)
    y_pred = torch.randn(2, 2, h, w)  # F, Db logits / values
    y_true = torch.zeros(2, 2, h, w)
    y_true[:, 0, 4:12, 4:12] = 1.0  # F
    y_true[:, 1, 4:12, 4:12] = 0.5  # Db
    skel = torch.zeros(2, 1, h, w)
    skel[:, :, 6:10, 6:10] = 1.0
    return y_pred, y_true, skel


def _check_sk_changes_loss() -> None:
    y_pred, y_true, skel = _base_batch()
    loss_base = instance_segmentation_loss(
        channel_weights=(1.0, 1.0),
        ndim=2,
        out_channels=["F", "Db"],
        losses_to_use=["bce", "l1"],
        gt_channels_expected=2,
        channel_extra_opts={"Db": {"mask_values": False}},
    )
    loss_sk = instance_segmentation_loss(
        channel_weights=(1.0, 1.0),
        ndim=2,
        out_channels=["F", "Db", "Sk"],
        losses_to_use=["bce", "l1"],
        gt_channels_expected=2,
        channel_extra_opts={"Db": {"mask_values": False}},
        skeleton_recall_weight=1.0,
    )
    assert loss_sk.gt_channels_expected == 3
    assert loss_sk.use_skeleton_recall

    v_base = loss_base(y_pred, y_true)
    y_true_sk = torch.cat([y_true, skel], dim=1)
    v_sk = loss_sk(y_pred, y_true_sk)
    assert torch.isfinite(v_sk), f"Sk loss not finite: {v_sk}"
    assert not torch.allclose(v_base, v_sk), (
        f"Sk should change the scalar loss (base={v_base.item():.6f}, sk={v_sk.item():.6f})"
    )
    print(f"  [PASS] Sk changes loss: base={v_base.item():.6f} sk={v_sk.item():.6f}")


def _check_gradients() -> None:
    y_pred, y_true, skel = _base_batch()
    y_pred = y_pred.clone().requires_grad_(True)
    loss_fn = instance_segmentation_loss(
        channel_weights=(1.0, 1.0),
        ndim=2,
        out_channels=["F", "Db", "Sk"],
        losses_to_use=["bce", "l1"],
        gt_channels_expected=2,
        channel_extra_opts={"Db": {"mask_values": False}},
    )
    v = loss_fn(y_pred, torch.cat([y_true, skel], dim=1))
    v.backward()
    assert y_pred.grad is not None
    assert torch.any(y_pred.grad[:, 0] != 0), "expected nonzero grad on F channel"
    print(f"  [PASS] gradients: F |grad| mean={y_pred.grad[:, 0].abs().mean().item():.6e}")


def _check_we_sk_coexistence() -> None:
    """Offset fix: We and Sk in distinct trailing slots must both be consumed correctly."""
    torch.manual_seed(1)
    h = w = 16
    y_pred = torch.randn(1, 1, h, w, requires_grad=False)
    f_gt = torch.zeros(1, 1, h, w)
    f_gt[:, :, 4:12, 4:12] = 1.0
    we = torch.ones(1, 1, h, w)
    we[:, :, 4:12, 4:12] = 5.0  # heavy border/weight inside FG
    skel_good = torch.zeros(1, 1, h, w)
    skel_good[:, :, 6:10, 6:10] = 1.0
    skel_bad = torch.zeros(1, 1, h, w)
    skel_bad[:, :, 0:2, 0:2] = 1.0  # no overlap with strong F logits region

    loss_fn = instance_segmentation_loss(
        channel_weights=(1.0,),
        ndim=2,
        out_channels=["F", "We", "Sk"],
        losses_to_use=["bce"],
        gt_channels_expected=1,
        skeleton_recall_weight=1.0,
    )
    assert loss_fn.aux_gt_channels == ["We", "Sk"]
    assert loss_fn.gt_channels_expected == 3

    # Strong positive F logits on FG so BCE is small; recall term dominates Sk sensitivity.
    y_pred_pos = torch.full((1, 1, h, w), -5.0)
    y_pred_pos[:, :, 4:12, 4:12] = 8.0

    y_true_we5 = torch.cat([f_gt, we, skel_good], dim=1)
    y_true_we1 = torch.cat([f_gt, torch.ones_like(we), skel_good], dim=1)
    v_we5 = loss_fn(y_pred_pos, y_true_we5)
    v_we1 = loss_fn(y_pred_pos, y_true_we1)
    assert not torch.allclose(v_we5, v_we1), (
        f"We map should affect BCE (we5={v_we5.item():.6f}, we1={v_we1.item():.6f})"
    )

    y_true_sk_bad = torch.cat([f_gt, we, skel_bad], dim=1)
    v_sk_bad = loss_fn(y_pred_pos, y_true_sk_bad)
    assert not torch.allclose(v_we5, v_sk_bad), (
        f"Sk overlap should affect recall (good={v_we5.item():.6f}, bad={v_sk_bad.item():.6f})"
    )
    # Worse Sk overlap -> higher (less negative recall contribution) total loss
    assert v_sk_bad.item() > v_we5.item(), (
        f"bad Sk overlap should worsen loss: good={v_we5.item():.6f} bad={v_sk_bad.item():.6f}"
    )
    print(
        f"  [PASS] We+Sk coexistence: we5={v_we5.item():.6f} we1={v_we1.item():.6f} "
        f"sk_bad={v_sk_bad.item():.6f}"
    )


def _check_missing_fm_raises() -> None:
    try:
        instance_segmentation_loss(
            channel_weights=(1.0,),
            ndim=2,
            out_channels=["Db", "Sk"],
            losses_to_use=["l1"],
            gt_channels_expected=1,
            channel_extra_opts={"Db": {"mask_values": False}},
        )
    except ValueError as e:
        assert "F" in str(e) or "M" in str(e) or "Skeleton Recall" in str(e)
        print(f"  [PASS] missing F/M raises: {e}")
        return
    raise AssertionError("expected ValueError when Sk is present without F/M")


def main() -> None:
    print("instance_segmentation_loss + Sk checks:")
    _check_sk_changes_loss()
    _check_gradients()
    _check_we_sk_coexistence()
    _check_missing_fm_raises()
    print("\nAll A2 checks passed.")


if __name__ == "__main__":
    main()
