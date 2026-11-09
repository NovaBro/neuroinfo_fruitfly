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


AMP_DTYPE = torch.float16


def build(config, device, args, sec):
    config = copy.deepcopy(config)
    if args.conv_first:
        config["autoencoder"]["upsample_conv_first"] = True
    # variant flags are authoritative (the TOML may already set these, e.g.
    # our-noenc-c1k_v1.toml has fast_affinities = true; "base" must stay the reference)
    config["fast_affinities"] = bool(args.fast_affinities)
    config["channels_last"] = bool(args.channels_last)
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
    # same code path as train.py (weights here, raw converted in forward())
    model = model.use_channels_last()

    # per-module forward timing hooks sync the GPU; they would break compiled graphs
    hooked = [] if (args.compile or args.compile_all) else [
        ("unet", model.unet), ("head_code_affs", model.out_code_affs),
        ("head_numinst", model.out_fgbg_numinst)] + (
            [("decoder", model.decoder)] if getattr(model, "decoder", None) else [])
    for name, mod in hooked:
        pre, post = time_hook(sec, name)
        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)
    if args.compile or args.compile_all:
        model.unet = torch.compile(model.unet)
        model.out_code_affs = torch.compile(model.out_code_affs)
        model.out_fgbg_numinst = torch.compile(model.out_fgbg_numinst)
    if args.compile_all:
        loss = torch.compile(loss)
    orig_aff = model.compute_affinities

    def timed_aff(*a, **k):
        with sec("compute_affinities"):
            return orig_aff(*a, **k)
    model.compute_affinities = timed_aff

    opt_kwargs = {k: v for k, v in config["optimizer"].items()
                  if k not in ("optimizer", "loss")}
    if args.fused_adam:
        opt_kwargs["fused"] = True
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
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=amp):
            out = model(**inputs)
    # outputs: 0 pred_affs, 1 pred_numinst, 2 gt_affs_samples, 3 gt_fgbg, (4 pred_code)
    with sec("loss"):
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=amp):
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


def _peak_tflops(args):
    if args.peak_tflops:
        return args.peak_tflops
    gpu = torch.cuda.get_device_name(0)
    return 362.0 if "L40S" in gpu else 989.0 if "H200" in gpu else None


def _conv_flops(mod, inp, out):
    """Multiply-add FLOPs (x2) of one conv forward pass."""
    w = mod.weight
    k = int(np.prod(w.shape[2:]))
    if isinstance(mod, (torch.nn.ConvTranspose3d, torch.nn.ConvTranspose2d)):
        return 2 * inp.numel() * w.shape[1] * k  # weight: (Cin, Cout/g, k...)
    return 2 * out.numel() * (w.shape[1]) * k   # weight: (Cout, Cin/g, k...)


def layer_bench(model, loss_mod, opt, scaler, inputs, amp, sec, step_s, args, reps=10):
    """Time every UNet block and output head alone (fwd + bwd, CUDA events) at the
    shapes they see in a real step, and count their conv FLOPs (D3/D4).

    Blocks: unet.l_conv.i, unet.l_down.i, unet.r_up.h.i, unet.r_conv.h.i,
    out_code_affs, out_fgbg_numinst. Each block's inputs are captured once from a
    full step; for the bench they are re-created with the same requires_grad, so
    backward includes the input-gradient path exactly where training has it.
    """
    import re
    unet = getattr(model.unet, "_orig_mod", model.unet)
    pat = re.compile(r"^(l_conv|l_down)\.\d+$|^(r_up|r_conv)\.\d+\.\d+$")
    blocks = [(f"unet.{n}", m) for n, m in unet.named_modules() if pat.match(n)]
    for name in ("out_code_affs", "out_fgbg_numinst"):
        m = getattr(model, name)
        blocks.append((name, getattr(m, "_orig_mod", m)))

    captured = {}

    def tmap(fn, x):
        """Apply fn to every tensor inside nested lists/tuples/dicts."""
        if torch.is_tensor(x):
            return fn(x)
        if isinstance(x, (list, tuple)):
            return type(x)(tmap(fn, v) for v in x)
        if isinstance(x, dict):
            return {k: tmap(fn, v) for k, v in x.items()}
        return x

    def cap(name):
        def hook(mod, a, kw, out):
            if name not in captured:
                # (detached value, requires_grad) per tensor, kept in the call structure
                captured[name] = (tmap(lambda t: (t.detach(), t.requires_grad), list(a)),
                                  tmap(lambda t: (t.detach(), t.requires_grad), kw))
        return hook
    hs = [m.register_forward_hook(cap(n), with_kwargs=True) for n, m in blocks]
    sec.enabled = False
    train_step(model, loss_mod, opt, scaler, inputs, amp, sec)
    torch.cuda.synchronize()
    for h in hs:
        h.remove()

    peak = _peak_tflops(args)
    rows = []
    for name, mod in blocks:
        if name not in captured:
            continue
        a_spec, kw_spec = captured[name]
        is_spec = lambda v: isinstance(v, tuple) and len(v) == 2 and torch.is_tensor(v[0]) \
            and isinstance(v[1], bool)  # noqa: E731

        def fresh(x):  # new leaf tensors with the captured requires_grad
            if is_spec(x):
                return x[0].clone().requires_grad_(x[1] and x[0].is_floating_point())
            if isinstance(x, (list, tuple)):
                return type(x)(fresh(v) for v in x)
            if isinstance(x, dict):
                return {k: fresh(v) for k, v in x.items()}
            return x

        def flat(x):
            if is_spec(x):
                return [x]
            if isinstance(x, (list, tuple)):
                return [s for v in x for s in flat(v)]
            if isinstance(x, dict):
                return [s for v in x.values() for s in flat(v)]
            return []
        specs = flat(a_spec) + flat(kw_spec)
        rg = [r for _, r in specs]
        flops = [0]

        def cf(m, i, o):
            flops[0] += _conv_flops(m, i[0], o)
        convs = [c.register_forward_hook(cf) for c in mod.modules()
                 if isinstance(c, (torch.nn.Conv3d, torch.nn.ConvTranspose3d,
                                   torch.nn.Conv2d, torch.nn.ConvTranspose2d))]
        args_ = fresh(a_spec)
        kw = fresh(kw_spec)
        has_params = any(p.requires_grad for p in mod.parameters())
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        fwd = bwd = 0.0
        for it in range(reps + 2):
            mod.zero_grad(set_to_none=True)
            args_ = fresh(a_spec)  # fresh leaves every rep: no graph reuse
            kw = fresh(kw_spec)
            ev[0].record()
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=amp):
                out = mod(*args_, **kw)
            ev[1].record()
            outs = [t for t in (out if isinstance(out, (tuple, list)) else [out])
                    if torch.is_tensor(t) and t.requires_grad]
            if outs:
                torch.autograd.backward(outs, [torch.ones_like(t) for t in outs])
            ev[2].record()
            torch.cuda.synchronize()
            if it == 0:
                fl_fwd = flops[0]
                for c in convs:
                    c.remove()
            if it >= 2:
                fwd += ev[0].elapsed_time(ev[1]) / reps
                bwd += ev[1].elapsed_time(ev[2]) / reps
        # backward of a conv = data grad (if input needs grad) + weight grad, each ~ fwd FLOPs
        n_bwd = (1 if has_params else 0) + (1 if any(rg) else 0)
        total_fl = fl_fwd * (1 + n_bwd)
        shape = tuple(specs[0][0].shape) if specs else ()
        rows.append((name, shape, fwd, bwd, fl_fwd, total_fl))

    tot_ms = sum(r[2] + r[3] for r in rows)
    tot_fl = sum(r[5] for r in rows)
    print(f"\n== layer bench ({torch.cuda.get_device_name(0)}; fwd+bwd per block in isolation, "
          f"{reps} reps; peak {peak} TFLOPS)")
    print(f"{'block':22s} {'input shape':28s} {'fwd ms':>7s} {'bwd ms':>7s} "
          f"{'tot ms':>7s} {'share':>6s} {'GFLOP':>8s} {'TFLOPS':>7s} {'%peak':>6s}")
    for name, shape, f, b, flf, flt in rows:
        t = f + b
        tf = flt / (t * 1e-3) / 1e12 if t > 0 and flt else 0.0
        pk = f"{100 * tf / peak:5.1f}%" if peak and tf else "     -"
        print(f"{name:22s} {str(shape):28s} {f:7.1f} {b:7.1f} {t:7.1f} "
              f"{100 * t / tot_ms:5.1f}% {flt / 1e9:8.1f} {tf:7.1f} {pk}")
    step_tf = tot_fl / step_s / 1e12
    print(f"{'sum of blocks':22s} {'':28s} {'':7s} {'':7s} {tot_ms:7.1f} {'':6s} "
          f"{tot_fl / 1e9:8.1f}")
    print(f"full step {step_s * 1e3:.1f} ms: blocks {100 * tot_ms / (step_s * 1e3):.1f}% of it; "
          f"model conv FLOPs / step time = {step_tf:.1f} TFLOPS"
          + (f" = {100 * step_tf / peak:.1f}% of peak" if peak else ""))


def run_variant(name, config, args, device, profile=False):
    sec = Sections()
    cfg, model, loss, opt, in_shape, out_shape = build(config, device, args, sec)
    amp = bool(cfg.get("use_auto_mixed_precision", False)) and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(init_scale=8192.0, growth_interval=200) \
        if amp else None
    inputs = make_inputs(cfg, in_shape, out_shape, device,
                         cfg.get("raw_transport_fp16", False))
    torch.cuda.reset_peak_memory_stats()

    sec.enabled = False
    t_first = time.perf_counter()
    train_step(model, loss, opt, scaler, inputs, amp, sec)
    torch.cuda.synchronize()
    t_first = time.perf_counter() - t_first
    print(f"first step (includes compile / cudnn autotune): {t_first:.1f}s", flush=True)
    for _ in range(max(args.warmup, 6 if (args.compile or args.compile_all) else 0) - 1):
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
        trace = os.path.join(args.trace_dir, f"ppp_step_trace_{name}.json")
        prof.export_chrome_trace(trace)
        print(f"\n== top 25 CUDA kernels/ops ({name}); chrome trace: {trace}")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25,
                                        max_name_column_width=60))

    peak = torch.cuda.max_memory_allocated() / 2**20
    if name in (args.layer_bench or []):
        try:
            layer_bench(model, loss, opt, scaler, inputs, amp, sec, step, args)
        except Exception:  # keep the variant's timing even if the bench breaks
            import traceback
            traceback.print_exc()
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
    ap.add_argument("--profile-variants", nargs="*", default=None,
                    help="variants that also get a torch.profiler kernel table + "
                         "chrome trace (default: the first variant and fast_aff)")
    # per-variant overrides (set internally)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--input-shape", type=int, default=None)
    ap.add_argument("--no-cudnn-benchmark", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--conv-first", action="store_true",
                    help="autoencoder.upsample_conv_first (1x1 conv before trilinear)")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the UNet and both output heads")
    ap.add_argument("--compile-all", action="store_true",
                    help="--compile plus the loss module")
    ap.add_argument("--fused-adam", action="store_true", help="optimizer fused=True")
    ap.add_argument("--layer-bench", nargs="*", default=None,
                    help="variants that also get a per-block fwd/bwd benchmark with "
                         "conv FLOPs and achieved TFLOPS (D3/D4)")
    ap.add_argument("--peak-tflops", type=float, default=None,
                    help="dense fp16 tensor peak for %%-of-peak (default: L40S 362, "
                         "H200 989, else none)")
    ap.add_argument("--fast-affinities", action="store_true",
                    help="training.fast_affinities (noenc gt affs pre-cropped, no host syncs)")
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
        "bs12": {"batch_size": 12},
        "bs24": {"batch_size": 24},
        "bf16": {"bf16": True},
        "fast_aff": {"fast_affinities": True},
        "compile": {"compile": True},
        "compile_all": {"compile_all": True},
        "fused_adam": {"fused_adam": True},
        "fast_aff_bf16": {"fast_affinities": True, "bf16": True},
        "fast_aff_bs24": {"fast_affinities": True, "batch_size": 24},
        "in112": {"input_shape": 112},
        "convfirst": {"conv_first": True},
        "convfirst_bf16": {"conv_first": True, "bf16": True},
        "convfirst_nocudnnbench": {"conv_first": True, "no_cudnn_benchmark": True},
    }
    # Composable variants: tokens joined by "+", e.g. "fast_aff+cl+bs63".
    # Each token is a name above, "cl" (= channels_last), "bsN" or "inN".
    def parse_variant(v):
        if v in overrides:
            return dict(overrides[v])
        out = {}
        for tok in v.split("+"):
            if tok == "cl":
                out["channels_last"] = True
            elif tok.startswith("bs") and tok[2:].isdigit():
                out["batch_size"] = int(tok[2:])
            elif tok.startswith("in") and tok[2:].isdigit():
                out["input_shape"] = int(tok[2:])
            elif tok in overrides:
                out.update(overrides[tok])
            else:
                raise SystemExit(f"unknown variant token {tok!r} in {v!r}")
        return out

    variant_overrides = {v: parse_variant(v) for v in args.variants}  # fail fast
    profiled = (set(args.profile_variants) if args.profile_variants is not None
                else {args.variants[0], "fast_aff"} if args.variants else set())
    base_args = vars(args).copy()
    results = []
    for v in args.variants:
        a = argparse.Namespace(**{**base_args, "batch_size": None,
                                  "input_shape": None, "no_cudnn_benchmark": False,
                                  "no_amp": False, "channels_last": False,
                                  "conv_first": False, "bf16": False,
                                  "fast_affinities": False, "compile": False,
                                  "compile_all": False, "fused_adam": False,
                                  **variant_overrides[v]})
        print(f"\n#### variant {v}", flush=True)
        global AMP_DTYPE
        AMP_DTYPE = torch.bfloat16 if a.bf16 else torch.float16
        try:
            results.append(run_variant(v, config, a, device,
                                       profile=(v in profiled
                                                and not args.no_profiler)))
        except Exception as e:  # keep going: a variant may OOM / be unsupported
            print(f"variant {v} failed: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    print("\n== summary (s per step; sections measured with cuda syncs)")
    print(f"{'variant':24s} {'bs':>3s} {'in':>4s} {'step':>7s} {'sum_sec':>8s} "
          f"{'peakMiB':>8s} {'ms/smp':>7s}  sections")
    for r in results:
        secs = r["sections"]
        tot = sum(v for k, v in secs.items() if k in
                  ("fwd_total", "loss", "backward", "opt_step"))
        detail = " ".join(f"{k}={v:.3f}" for k, v in sorted(secs.items()))
        print(f"{r['name']:24s} {r['bs']:3d} {r['in'][0]:4d} {r['step']:7.3f} "
              f"{tot:8.3f} {r['peak_mib']:8.0f} {1000*r['step']/r['bs']:7.1f}  {detail}")
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
