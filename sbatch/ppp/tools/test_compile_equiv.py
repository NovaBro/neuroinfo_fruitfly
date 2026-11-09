"""Numerical-equivalence check for torch.compile on the noenc training step.

Builds the noenc model twice from identical weights -- eager, and with the UNet, both output
heads and (with --compile-loss, default on) the loss module wrapped in torch.compile (Inductor) --
runs the same synthetic batch through fwd + loss + backward and compares loss, outputs and
parameter gradients. Inductor fuses elementwise ops and may reorder reductions, so results are
close, not bit-equal:
  * fp32 with TF32 off  -> must agree to --tol relative (hard check, exits non-zero)
  * AMP fp16 (as train) -> reported only
Also runs a few AMP steps to confirm the compiled graph does not recompile every step.

Run inside the ppp container from PatchPerPix/experiments/ (GPU):
  python -u ../../sbatch/ppp/tools/test_compile_equiv.py \
    --config flylight/setups/setup01/configs/base/original_default.toml \
    --config flylight/setups/setup01/configs/presets/our-noenc-c1k_v1.toml
"""
import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_channels_last import build, rel  # noqa: E402
from profile_ppp_step import load_config, make_inputs  # noqa: E402


def step(model, loss, inputs, amp):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        out = model(**inputs)
        lo = loss(pred_logits_affs=out[0], pred_logits_fg=out[1],
                  gt_affs_samples=out[2], gt_fgbg_loss=out[3])["loss"]
    lo.backward()
    grads = {n.replace("._orig_mod", ""): p.grad.detach().float().clone()
             for n, p in model.named_parameters() if p.grad is not None}
    return lo.detach().float(), [o.detach().float() for o in out[:2]], grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--no-compile-loss", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="max relative error (fp32, TF32 off)")
    args = ap.parse_args()
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), "torch", torch.__version__)

    config = load_config(args.config)
    config["batch_size"] = args.batch_size
    config["fast_affinities"] = True

    _, model_a, loss_a = build(config, device, channels_last=False)
    state = copy.deepcopy(model_a.state_dict())
    cfg_b, model_b, loss_b = build(config, device, channels_last=False, state=state)
    model_b.unet = torch.compile(model_b.unet)
    model_b.out_code_affs = torch.compile(model_b.out_code_affs)
    model_b.out_fgbg_numinst = torch.compile(model_b.out_fgbg_numinst)
    if not args.no_compile_loss:
        loss_b = torch.compile(loss_b)

    in_shape = list(config["train_input_shape_valid"])
    in_shape, out_shape = model_a.inout_shapes(in_shape, "train_net", training=True)
    inputs = make_inputs(cfg_b, in_shape, out_shape, device, fp16=False)

    ok = True
    for amp in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = amp
        torch.backends.cudnn.allow_tf32 = amp
        torch.backends.cudnn.benchmark = False
        la, oa, ga = step(model_a, loss_a, inputs, amp)
        lb, ob, gb = step(model_b, loss_b, inputs, amp)
        missing = set(ga) ^ set(gb)
        if missing:
            print("grad name mismatch:", sorted(missing)[:5])
            ok = False
            continue
        r_loss = abs(la.item() - lb.item()) / max(abs(la.item()), 1e-12)
        r_out = max(rel(x, y) for x, y in zip(ob, oa))
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

    # recompilation check: further AMP steps must reuse the compiled graphs
    import torch._dynamo as dynamo
    before = dynamo.utils.counters["stats"]["unique_graphs"]
    for _ in range(3):
        step(model_b, loss_b, inputs, True)
    after = dynamo.utils.counters["stats"]["unique_graphs"]
    print(f"unique compiled graphs: {before} -> {after} over 3 more steps "
          f"({'no recompiles' if after == before else 'RECOMPILES'})")
    ok &= after == before
    print("COMPILE OK" if ok else "COMPILE MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
