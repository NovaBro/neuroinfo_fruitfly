"""Unit checks for SoftSkeletonRecallLoss (Phase A1).

Runs on CPU; does not load FISBe volumes or touch training. Imports the local
BiaPy-novabro checkout so the new class is found without an install.

Example
-------
    # Prefer conda's libstdc++ so scipy/skimage C++ extensions resolve inside the SIF:
    singularity exec --overlay env/BiaPy_env.ext3:ro \\
      /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \\
      /bin/bash -c 'source /ext3/env.sh; conda activate BiaPy_env; \\
      export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}; \\
      python biapy_work_folder/check_skeleton_recall_loss.py'
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "BiaPy-novabro"))

from biapy.engine.metrics import SoftSkeletonRecallLoss  # noqa: E402


def _check_perfect_overlap() -> None:
    """Large positive logits on the skeleton voxels -> recall ~ 1 -> loss ~ -1."""
    loss_fn = SoftSkeletonRecallLoss(smooth=1.0, batch_dice=True)
    skel = torch.zeros(2, 1, 16, 16)
    skel[:, :, 4:12, 4:12] = 1.0
    pred = torch.full_like(skel, -10.0)
    pred[skel == 1] = 20.0
    loss = loss_fn(pred, skel)
    assert loss.ndim == 0, f"expected scalar, got shape {tuple(loss.shape)}"
    assert loss.item() < -0.99, f"perfect overlap should be near -1, got {loss.item():.6f}"
    print(f"  [PASS] perfect overlap: loss={loss.item():.6f}")


def _check_no_overlap() -> None:
    """Negative logits on skeleton voxels -> low recall -> loss near 0 (worse than perfect)."""
    loss_fn = SoftSkeletonRecallLoss(smooth=1.0, batch_dice=True)
    skel = torch.zeros(2, 1, 16, 16)
    skel[:, :, 4:12, 4:12] = 1.0
    pred = torch.full_like(skel, -20.0)
    loss_bad = loss_fn(pred, skel)

    pred_good = torch.full_like(skel, -10.0)
    pred_good[skel == 1] = 20.0
    loss_good = loss_fn(pred_good, skel)

    assert loss_bad.item() > loss_good.item(), (
        f"no-overlap loss ({loss_bad.item():.6f}) should be worse (larger) than "
        f"perfect ({loss_good.item():.6f})"
    )
    # For a large skeleton, -smooth/(sum_skel+smooth) is near 0.
    assert loss_bad.item() > -0.1, f"no-overlap should be near 0, got {loss_bad.item():.6f}"
    print(f"  [PASS] no overlap: loss={loss_bad.item():.6f} (> perfect {loss_good.item():.6f})")


def _check_gradients() -> None:
    """Gradients must flow into y_pred on skeleton voxels."""
    loss_fn = SoftSkeletonRecallLoss(smooth=1.0, batch_dice=True)
    skel = torch.zeros(1, 1, 8, 8)
    skel[:, :, 2:6, 2:6] = 1.0
    pred = torch.zeros_like(skel, requires_grad=True)
    loss = loss_fn(pred, skel)
    loss.backward()
    assert pred.grad is not None, "y_pred.grad is None after backward"
    grad_on_skel = pred.grad[skel == 1]
    assert torch.any(grad_on_skel != 0), "expected nonzero grad where skeleton == 1"
    print(f"  [PASS] gradients: nonzero on skeleton (mean |grad|={grad_on_skel.abs().mean().item():.6e})")


def _check_shapes() -> None:
    """2D (B,1,H,W) and 3D (B,1,D,H,W) both produce a scalar."""
    loss_fn = SoftSkeletonRecallLoss()
    for shape, label in (
        ((2, 1, 12, 12), "2D"),
        ((2, 1, 4, 8, 8), "3D"),
    ):
        skel = (torch.rand(*shape) > 0.7).float()
        pred = torch.randn(*shape)
        out = loss_fn(pred, skel)
        assert out.ndim == 0 and torch.isfinite(out), f"{label} failed: {out}"
        print(f"  [PASS] shape {label} {shape}: loss={out.item():.6f}")


def main() -> None:
    print("SoftSkeletonRecallLoss checks:")
    _check_perfect_overlap()
    _check_no_overlap()
    _check_gradients()
    _check_shapes()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
