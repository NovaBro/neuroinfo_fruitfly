"""Profile one PPP training step on the GPU with synthetic inputs (no data pipeline).

Answers "where do the ~5 s/iter of TorchTrainExt go?" for a given TOML stack:
per-section timings (UNet fwd, code/numinst heads, affinity targets, patch gather +
decoder, loss, backward, optimizer), a torch.profiler kernel table, and variants
(cudnn.benchmark off, no AMP, channels_last, smaller batch / input).

Run inside the ppp container from PatchPerPix/experiments/ (see
profile_ppp_step_sbatch.sh). Model/loss construction and the step mirror
flylight/setups/setup01/train.py:train_until and gunpowder TorchTrainExt
(autocast + GradScaler(init_scale=8192, growth_interval=200), zero_grad, fwd, loss,
scaler.scale(loss).backward, scaler.step) -- keep in sync.
"""
import argparse
import contextlib
import copy
import os
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np
import toml
import torch

sys.path.insert(0, os.getcwd())
from flylight.setups.setup01 import torch_model, torch_loss  # noqa: E402


def merge_dicts(sink, source):
    for k, v in source.items():
        if isinstance(v, dict) and isinstance(sink.get(k), dict):
            merge_dicts(sink[k], v)
        else:
            sink[k] = v
    return sink


def load_config(paths):
    cfg = {}
    for p in paths:
        merge_dicts(cfg, toml.load(p))
    flat = {**cfg["data"], **cfg["model"], **cfg["training"],
            **cfg.get("preprocessing", {})}
    flat["optimizer"] = cfg["optimizer"]["optimizer"]  # as run_ppp passes it
    return flat


class Sections:
    """Wall-clock sections with a cuda sync on both edges (exact, slightly perturbing)."""

    def __init__(self):
        self.t = defaultdict(float)
        self.enabled = True

    @contextlib.contextmanager
    def __call__(self, name):
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        torch.cuda.synchronize()
        self.t[name] += time.perf_counter() - t0


def time_hook(sec, name):
    state = {}

    def pre(mod, args, kwargs=None):
        if sec.enabled:
            torch.cuda.synchronize()
            state["t0"] = time.perf_counter()

    def post(mod, args, out):
        if sec.enabled:
            torch.cuda.synchronize()
            sec.t[name] += time.perf_counter() - state["t0"]
    return pre, post


def build(config, device, args, sec):
    config = copy.deepcopy(config)
    config["output_folder"] = tempfile.mkdtemp()
    if args.batch_size:
        config["batch_size"] = args.batch_size
    in_shape = [args.input_shape] * 3 if args.input_shape else list(
        config["train_input_shape_valid"])

    torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # skip the constructor's throwaway torchinfo forward pass (bs x 140^3 fp32 on the
    # default stream; it crashed with an illegal memory access on gh134 and is not
    # part of the training step)
    torch_model.torchinfo.summary = lambda *a, **k: None
    model = torch_model.UnetModelWrapper(config, device, 0)
    model.init_layers()
    model = model.to(device)
    model.set_padding(config["train_padding"])
    in_shape, out_shape = model.inout_shapes(in_shape, "train_net", training=True)
    model.train()
    loss = torch_loss.LossWrapper(config, current_step=0).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last_3d)

    for name, mod in [("unet", model.unet), ("head_code_affs", model.out_code_affs),
                      ("head_numinst", model.out_fgbg_numinst)] + (
            [("decoder", model.decoder)] if getattr(model, "decoder", None) else []):
        pre, post = time_hook(sec, name)
        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)
    orig_aff = model.compute_affinities

    def timed_aff(*a, **k):
        with sec("compute_affinities"):
            return orig_aff(*a, **k)
    model.compute_affinities = timed_aff

    opt_kwargs = {k: v for k, v in config["optimizer"].items()
                  if k not in ("optimizer", "loss")}
    opt = getattr(torch.optim, config["optimizer"]["optimizer"])(
        model.parameters(), **opt_kwargs)
    return config, model, loss, opt, in_shape, out_shape


def make_inputs(config, in_shape, out_shape, device, fp16):
    B = config["batch_size"]
    ps = config["patchshape"][-1]
    nch = config.get("num_channels", 1)
    n_inst = config.get("max_num_inst", 2)
    inst_shape = [int(o) + ps - 1 for o in out_shape]
    g = torch.Generator(device=device).manual_seed(0)
    raw = torch.rand((B, nch) + tuple(in_shape), device=device, generator=g)
    if fp16:
        raw = raw.half()
    # sparse blobby foreground: ~5% voxels labelled so each sample has fg locations
    lab = (torch.rand((B, n_inst) + tuple(inst_shape), device=device, generator=g)
           < 0.05).int()
    lab = lab * torch.randint(1, 6, lab.shape, device=device, generator=g).int()
    # gt_numinst arrives as (B, z, y, x) (no channel dim), cropped to the output
    numinst = torch.clamp((lab > 0).sum(dim=1), 0, n_inst)[
        (slice(None),) + tuple(
            slice(ps // 2, ps // 2 + int(o)) for o in out_shape)].long()
    return {"raw": raw, "gt_labels": lab, "gt_fgbg_numinst": numinst}


def train_step(model, loss, opt, scaler, inputs, amp, sec):
    opt.zero_grad()
    with sec("fwd_total"):
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(**inputs)
    # outputs: 0 pred_affs, 1 pred_numinst, 2 gt_affs_samples, 3 gt_fgbg, (4 pred_code)
    with sec("loss"):
        with torch.cuda.amp.autocast(enabled=amp):
            lo = loss(pred_logits_affs=out[0], pred_logits_fg=out[1],
                      gt_affs_samples=out[2], gt_fgbg_loss=out[3])["loss"]
    with sec("backward"):
        (scaler.scale(lo) if amp else lo).backward()
    with sec("opt_step"):
        if amp:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
    return lo


def run_variant(name, config, args, device, profile=False):
    sec = Sections()
    cfg, model, loss, opt, in_shape, out_shape = build(config, device, args, sec)
    amp = bool(cfg.get("use_auto_mixed_precision", False)) and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(init_scale=8192.0, growth_interval=200) \
        if amp else None
    inputs = make_inputs(cfg, in_shape, out_shape, device,
                         cfg.get("raw_transport_fp16", False))
    if args.channels_last:
        inputs["raw"] = inputs["raw"].contiguous(memory_format=torch.channels_last_3d)
    torch.cuda.reset_peak_memory_stats()

    sec.enabled = False
    for _ in range(args.warmup):
        train_step(model, loss, opt, scaler, inputs, amp, sec)
    torch.cuda.synchronize()

    # untimed-by-section: plain end-to-end step time (no extra syncs inside)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        train_step(model, loss, opt, scaler, inputs, amp, sec)
    torch.cuda.synchronize()
    step = (time.perf_counter() - t0) / args.steps

    sec.enabled = True
    sec.t.clear()
    for _ in range(args.steps):
        train_step(model, loss, opt, scaler, inputs, amp, sec)
    sections = {k: v / args.steps for k, v in sec.t.items()}

    if profile:
        os.makedirs(args.trace_dir, exist_ok=True)
        sec.enabled = False
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(2):
                train_step(model, loss, opt, scaler, inputs, amp, sec)
            torch.cuda.synchronize()
        trace = os.path.join(args.trace_dir, "ppp_step_trace.json")
        prof.export_chrome_trace(trace)
        print(f"\n== top 15 CUDA kernels/ops ({name}); chrome trace: {trace}")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15,
                                        max_name_column_width=60))

    peak = torch.cuda.max_memory_allocated() / 2**20
    del model, loss, opt, inputs
    torch.cuda.empty_cache()
    return {"name": name, "step": step, "sections": sections, "peak_mib": peak,
            "bs": cfg["batch_size"], "in": in_shape, "out": list(out_shape)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True,
                    help="TOML stack, later files override (repeatable)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--trace-dir", default="../../metrics/ppp/profile")
    ap.add_argument("--variants", nargs="*", default=[
        "base", "no_cudnn_benchmark", "channels_last", "bs16", "in112", "no_amp"])
    ap.add_argument("--no-profiler", action="store_true")
    # per-variant overrides (set internally)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--input-shape", type=int, default=None)
    ap.add_argument("--no-cudnn-benchmark", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), "torch", torch.__version__)
    config = load_config(args.config)

    overrides = {
        "base": {},
        "no_cudnn_benchmark": {"no_cudnn_benchmark": True},
        "no_amp": {"no_amp": True},
        "channels_last": {"channels_last": True},
        "bs16": {"batch_size": 16},
        "in112": {"input_shape": 112},
    }
    base_args = vars(args).copy()
    results = []
    for v in args.variants:
        a = argparse.Namespace(**{**base_args, "batch_size": None,
                                  "input_shape": None, "no_cudnn_benchmark": False,
                                  "no_amp": False, "channels_last": False,
                                  **overrides[v]})
        print(f"\n#### variant {v}", flush=True)
        try:
            results.append(run_variant(v, config, a, device,
                                       profile=(v == "base" and not args.no_profiler)))
        except Exception as e:  # keep going: a variant may OOM / be unsupported
            print(f"variant {v} failed: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    print("\n== summary (s per step; sections measured with cuda syncs)")
    print(f"{'variant':20s} {'bs':>3s} {'in':>4s} {'step':>7s} {'sum_sec':>8s} "
          f"{'peakMiB':>8s}  sections")
    for r in results:
        secs = r["sections"]
        tot = sum(v for k, v in secs.items() if k in
                  ("fwd_total", "loss", "backward", "opt_step"))
        detail = " ".join(f"{k}={v:.3f}" for k, v in sorted(secs.items()))
        print(f"{r['name']:20s} {r['bs']:3d} {r['in'][0]:4d} {r['step']:7.3f} "
              f"{tot:8.3f} {r['peak_mib']:8.0f}  {detail}")
    if results:
        b = results[0]
        print("\nbase section share of timed step:")
        tot = sum(v for k, v in b["sections"].items() if k in
                  ("fwd_total", "loss", "backward", "opt_step"))
        for k in ("fwd_total", "unet", "head_code_affs", "head_numinst",
                  "compute_affinities", "decoder", "loss", "backward", "opt_step"):
            if k in b["sections"]:
                print(f"  {k:20s} {b['sections'][k]:.3f}s  {100*b['sections'][k]/tot:5.1f}%")


if __name__ == "__main__":
    main()
