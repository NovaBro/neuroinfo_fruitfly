"""Numerical-equivalence check for training.channels_last (UnetModelWrapper.use_channels_last).

Builds the noenc model twice from identical weights -- default NCDHW and channels_last_3d --
runs the same synthetic batch through fwd + loss + backward, and compares loss, outputs and
parameter gradients. channels_last only changes memory strides, so the math is the same, but
cuDNN picks different kernels/reduction orders, so results are close, not bit-equal:
  * fp32 with TF32 off  -> must agree to ~1e-4 relative (hard check, exits non-zero)
  * AMP fp16 (as train) -> reported only (fp16 kernel-order noise ~1e-2)
Also checks that the state_dict round-trips (checkpoints stay format-agnostic).

Run inside the ppp container from PatchPerPix/experiments/ (GPU):
  python -u ../../sbatch/ppp/tools/test_channels_last.py \
    --config flylight/setups/setup01/configs/base/original_default.toml \
    --config flylight/setups/setup01/configs/presets/our-noenc-c1k_v1.toml
"""
import argparse
import copy
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flylight.setups.setup01 import torch_model, torch_loss  # noqa: E402
from profile_ppp_step import load_config, make_inputs  # noqa: E402


def build(config, device, channels_last, state=None):
    cfg = copy.deepcopy(config)
    cfg["channels_last"] = channels_last
    cfg["output_folder"] = tempfile.mkdtemp()
    torch_model.torchinfo.summary = lambda *a, **k: None
    torch.manual_seed(0)
    model = torch_model.UnetModelWrapper(cfg, device, 0)
    model.init_layers()
    model = model.to(device)
    if state is not None:
        model.load_state_dict(state)
    model = model.use_channels_last()
    model.set_padding(cfg["train_padding"])
    model.train()
    loss = torch_loss.LossWrapper(cfg, current_step=0).to(device)
    return cfg, model, loss


def step(model, loss, inputs, amp):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        out = model(**inputs)
        lo = loss(pred_logits_affs=out[0], pred_logits_fg=out[1],
                  gt_affs_samples=out[2], gt_fgbg_loss=out[3])["loss"]
    lo.backward()
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters()
             if p.grad is not None}
    return lo.detach().float(), [o.detach().float() for o in out[:2]], grads


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="max relative error (fp32, TF32 off)")
    args = ap.parse_args()
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), "torch", torch.__version__)

    config = load_config(args.config)
    config["batch_size"] = args.batch_size
    config["fast_affinities"] = True

    cfg_a, model_a, loss_a = build(config, device, channels_last=False)
    state = copy.deepcopy(model_a.state_dict())
    cfg_b, model_b, loss_b = build(config, device, channels_last=True, state=state)

    conv_w = [p for p in model_b.parameters() if p.dim() == 5]
    assert conv_w and all(p.is_contiguous(memory_format=torch.channels_last_3d)
                          for p in conv_w), "weights not channels_last_3d"
    sd_ok = all(torch.equal(state[k], v) for k, v in model_b.state_dict().items())
    print(f"state_dict values identical after conversion: {sd_ok}")

    in_shape = list(config["train_input_shape_valid"])
    in_shape, out_shape = model_a.inout_shapes(in_shape, "train_net", training=True)
    inputs = make_inputs(cfg_a, in_shape, out_shape, device, fp16=False)

    ok = sd_ok
    for amp in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = amp
        torch.backends.cudnn.allow_tf32 = amp
        torch.backends.cudnn.benchmark = False
        la, oa, ga = step(model_a, loss_a, inputs, amp)
        lb, ob, gb = step(model_b, loss_b, inputs, amp)
        r_loss = abs(la.item() - lb.item()) / max(abs(la.item()), 1e-12)
        r_out = max(rel(x, y) for x, y in zip(ob, oa))
        # global error over all grads is the hard check; per-param max is informative
        # (a near-zero grad, e.g. some bias, can have large relative noise)
        r_gall = rel(torch.cat([gb[n].flatten() for n in ga]),
                     torch.cat([ga[n].flatten() for n in ga]))
        r_grad = {n: rel(gb[n], ga[n]) for n in ga}
        worst = max(r_grad, key=r_grad.get)
        tag = "amp_fp16" if amp else "fp32"
        print(f"{tag:9s} loss {la.item():.6f} vs {lb.item():.6f} rel={r_loss:.2e}  "
              f"out rel={r_out:.2e}  grad rel all={r_gall:.2e} "
              f"per-param max={r_grad[worst]:.2e} ({worst})")
        if not amp:
            ok &= max(r_loss, r_out, r_gall) < args.tol
    print("CHANNELS_LAST OK" if ok else "CHANNELS_LAST MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
