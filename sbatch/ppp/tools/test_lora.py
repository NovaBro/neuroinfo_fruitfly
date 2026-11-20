"""Checks for setup01/lora.py (LoRA fine-tuning of the noenc UNet).

  1. injection: only LoRA + head params trainable; with B = 0 the wrapped model's
     outputs equal the plain model's;
  2. merge: after randomising the adapters, merge_lora_state_dict() + strict load into
     a plain UnetModelWrapper (what predict/decode do) reproduces the LoRA outputs
     (fp32, TF32 off, --tol relative);
  3. backward: gradients only reach trainable params; an AMP (+ compile, if the config
     sets it) optimizer step changes LoRA/head weights and no frozen weight;
  4. seed conversion: a checkpoint from sbatch/ppp/expv2/seed_finetune_ckpt.py goes
     through lora.prepare + write_seed_checkpoint, then loads gunpowder-style (strict
     model, optimizer, scheduler) with the seed's lr; the backup does not shadow the
     checkpoint; a second prepare() sees LoRA form (resume path).
Steps 1-2 run for targets [l_conv, r_conv] and [l_conv, r_conv, r_up].

Run inside the ppp container from PatchPerPix/experiments/ (GPU):
  python -u ../../sbatch/ppp/tools/test_lora.py \
    --config ../../metrics/ppp/expv2-e1.1-pre-n1000/config.toml \
    --checkpoint ../../metrics/ppp/expv2-e1.1-pre-n1000/train/train_net_checkpoint_20000
"""
import argparse
import copy
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flylight.setups.setup01 import lora, torch_model, torch_loss  # noqa: E402
from flylight.setups.setup01.train import make_warmup_cosine  # noqa: E402
from PatchPerPix.util import get_latest_checkpoint  # noqa: E402
from profile_ppp_step import load_config, make_inputs  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
SEED_SCRIPT = os.path.join(REPO, "sbatch", "ppp", "expv2", "seed_finetune_ckpt.py")
FAILS = []


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def plain_model(cfg, device, state=None):
    torch_model.torchinfo.summary = lambda *a, **k: None
    m = torch_model.UnetModelWrapper(copy.deepcopy(cfg), device, 0)
    m.init_layers()
    if state is not None:
        m.load_state_dict(state)
    return m


def forward(model, inputs, cfg, amp=False):
    model.set_padding(cfg["train_padding"])
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        out = model(**inputs)
    return out


def test_inject_merge(cfg, state, inputs, device, targets, tol):
    print(f"\n[inject/merge] targets {targets}")
    lcfg = {"enabled": True, "rank": 8, "alpha": 16.0, "targets": targets,
            "train_heads": True}
    ref = plain_model(cfg, device, state).to(device).eval()
    model = plain_model(cfg, device, state)
    wrapped = lora.inject_lora(model, lcfg)
    model = model.to(device).eval()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bad = [n for n, p in model.named_parameters() if p.requires_grad and not (
        ".lora_" in n or n.startswith(("out_code_affs.", "out_fgbg_numinst.")))]
    check(not bad, f"{len(wrapped)} convs wrapped, {n_tr} trainable params; "
                   f"only LoRA/head params trainable {bad[:3]}")
    n_tr_conv = sum(1 for w in wrapped if "r_up" in w)
    check(("r_up" in targets) == (n_tr_conv > 0),
          f"transposed convs wrapped: {n_tr_conv}")

    with torch.no_grad():
        o_ref = forward(ref, inputs, cfg)
        o_lora = forward(model, inputs, cfg)
    for i, name in ((0, "affs"), (1, "numinst")):
        d = (o_lora[i] - o_ref[i]).abs().max().item()
        check(d == 0.0, f"B = 0: {name} identical to plain model (max |diff| {d:.3g})")

    g = torch.Generator(device="cpu").manual_seed(1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".lora_B." in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
        o_lora = forward(model, inputs, cfg)
    merged = lora.merge_lora_state_dict(
        {k: v.cpu() for k, v in model.state_dict().items()})
    check(not lora.has_lora(merged) and set(merged) == set(state),
          "merged state_dict has exactly the plain keys")
    plain = plain_model(cfg, device)
    plain.load_state_dict(merged)            # strict, as in predict/decode
    plain = plain.to(device).eval()
    with torch.no_grad():
        o_merged = forward(plain, inputs, cfg)
    for i, name in ((0, "affs"), (1, "numinst")):
        changed = rel(o_lora[i], o_ref[i])
        err = rel(o_merged[i], o_lora[i])
        check(changed > 10 * tol and err < tol,
              f"merged plain model == LoRA model: {name} rel err {err:.2e} "
              f"(adapters moved output by {changed:.2e})")


def test_train_step(cfg, state, inputs, device):
    amp = bool(cfg.get("use_auto_mixed_precision", False))
    comp = bool(cfg.get("compile", False))
    print(f"\n[train step] amp={amp} compile={comp} channels_last="
          f"{bool(cfg.get('channels_last', False))}")
    model = plain_model(cfg, device, state)
    lora.inject_lora(model, {"enabled": True, "rank": 8, "alpha": 8.0})
    model = model.to(device).use_channels_last()
    model.train()
    loss = torch_loss.LossWrapper(cfg, current_step=0).to(device)
    if comp:
        import torch._dynamo
        torch._dynamo.config.automatic_dynamic_shapes = False
        model.unet.compile()
        model.out_code_affs.compile()
        model.out_fgbg_numinst.compile()
    params = lora.trainable_parameters(model)
    opt = torch.optim.Adam(params, lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for it in range(3):
        opt.zero_grad(set_to_none=True)
        out = forward(model, inputs, cfg, amp=amp)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            lo = loss(pred_logits_affs=out[0], pred_logits_fg=out[1],
                      gt_affs_samples=out[2], gt_fgbg_loss=out[3])["loss"]
        scaler.scale(lo).backward()
        if it == 0:
            frozen_grad = [n for n, p in model.named_parameters()
                           if not p.requires_grad and p.grad is not None]
            check(not frozen_grad, f"no grads on frozen params {frozen_grad[:3]}")
            with_grad = [n for n, p in model.named_parameters() if p.grad is not None]
            check(any(".lora_A." in n for n in with_grad)
                  and any(n.startswith("out_code_affs") for n in with_grad),
                  f"grads on LoRA A and heads ({len(with_grad)} tensors)")
        scaler.step(opt)
        scaler.update()
        print(f"  step {it}: loss {lo.item():.5f}", flush=True)
    after = dict(model.named_parameters())
    moved_frozen = [n for n, p in after.items()
                    if not p.requires_grad and not torch.equal(p, before[n])]
    moved_lora = sum(1 for n, p in after.items()
                     if ".lora_" in n and not torch.equal(p, before[n]))
    check(not moved_frozen, f"frozen weights unchanged {moved_frozen[:3]}")
    check(moved_lora > 0, f"{moved_lora} LoRA tensors updated")


def test_seed_conversion(cfg, src_ckpt, device):
    print("\n[seed conversion]")
    n, peak, ft = 20000, 5e-4, 100
    with tempfile.TemporaryDirectory() as tmp:
        if src_ckpt is None:
            src_ckpt = os.path.join(tmp, "src_checkpoint")
            m = plain_model(cfg, "cpu")
            torch.save({"model_state_dict": m.state_dict(),
                        "optimizer_state_dict": torch.optim.Adam(
                            m.parameters(), lr=4e-4).state_dict()}, src_ckpt)
        train_dir = os.path.join(tmp, "train")
        subprocess.run([sys.executable, SEED_SCRIPT, "--src", src_ckpt,
                        "--dst-dir", train_dir, "--iteration", str(n), "--lr", str(peak),
                        "--warmup-frac", "0.05", "--ft-iters", str(ft)], check=True)
        basename = os.path.join(train_dir, "train_net")
        path = basename + f"_checkpoint_{n}"
        seed_state = torch.load(path, map_location="cpu")
        lr0 = seed_state["optimizer_state_dict"]["param_groups"][0]["lr"]

        c = copy.deepcopy(cfg)
        c["lora"] = {"enabled": True, "rank": 8, "alpha": 8.0}
        c["optimizer"] = {"optimizer": "Adam", "lr": peak}

        def build_like_train():
            model = plain_model(c, device)       # random init, like train.py
            seed = lora.prepare(model, c, basename, n)
            model = model.to(device)
            opt = torch.optim.Adam(lora.trainable_parameters(model), lr=peak)
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=make_warmup_cosine(5, ft, 0.0, n))
            return model, opt, sched, seed

        model, opt, sched, seed = build_like_train()
        check(seed is not None, "plain seed detected for conversion")
        lora.write_seed_checkpoint(seed, model, opt, sched)
        backup = basename + f"_seed_prelora_{n}"
        check(os.path.exists(backup), "original seed kept as train_net_seed_prelora_N")
        latest = get_latest_checkpoint(basename)
        check(latest == (path, n), f"latest checkpoint is still {latest}")
        stray = [f for f in os.listdir(train_dir) if "tmp" in f]
        check(not stray, f"no temp files left {stray}")

        # resume path: fresh process -> prepare sees LoRA form; gunpowder-style load
        model, opt, sched, seed = build_like_train()
        check(seed is None, "converted checkpoint recognised as LoRA form")
        ck = torch.load(path, map_location=device)
        model.load_state_dict(ck["model_state_dict"])            # strict
        opt.load_state_dict(ck["optimizer_state_dict"])
        sched.load_state_dict(ck["lr_scheduler_state_dict"])
        g = opt.param_groups[0]
        check(abs(g["lr"] - lr0) < 1e-15 and abs(g["initial_lr"] - peak) < 1e-15,
              f"optimizer lr {g['lr']:.3e} (seed {lr0:.3e}), initial_lr {g['initial_lr']}")
        opt.step()
        sched.step()
        exp_lr = peak * make_warmup_cosine(5, ft, 0.0, n)(n + 1)
        check(abs(opt.param_groups[0]["lr"] - exp_lr) < 1e-12,
              f"next-step lr {opt.param_groups[0]['lr']:.3e} == schedule {exp_lr:.3e}")
        merged = lora.merge_lora_state_dict(
            {k: v.cpu() for k, v in model.state_dict().items()})
        src = torch.load(src_ckpt, map_location="cpu")["model_state_dict"]
        same = all(torch.equal(merged[k].float(), src[k].float()) for k in src)
        check(same, "converted checkpoint holds the pretrained weights (B = 0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--checkpoint", help="pretrained checkpoint (default: random init)")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), "torch", torch.__version__)

    cfg = load_config(args.config)
    cfg["batch_size"] = args.batch_size
    cfg["output_folder"] = tempfile.mkdtemp()
    cfg.pop("lora", None)
    torch.manual_seed(0)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")["model_state_dict"]
    else:
        state = plain_model(cfg, "cpu").state_dict()

    probe = plain_model(cfg, device, state).to(device)
    probe.set_padding(cfg["train_padding"])
    in_shape, out_shape = probe.inout_shapes(
        list(cfg["train_input_shape_valid"]), "train_net", training=True)
    inputs = make_inputs(cfg, in_shape, out_shape, device, fp16=False)
    del probe

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    test_inject_merge(cfg, state, inputs, device, ["l_conv", "r_conv"], args.tol)
    test_inject_merge(cfg, state, inputs, device, ["l_conv", "r_conv", "r_up"], args.tol)
    torch.backends.cudnn.deterministic = False
    test_train_step(cfg, state, inputs, device)
    test_seed_conversion(cfg, args.checkpoint, device)

    print("\nALL OK" if not FAILS else f"\n{len(FAILS)} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
