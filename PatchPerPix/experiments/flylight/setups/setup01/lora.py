"""LoRA (low-rank adaptation) for fine-tuning the setup01 UNet.

Enabled by a ``[training.lora]`` TOML section (run_ppp flattens [training], so it
arrives in train_until as ``config["lora"]``)::

    [training.lora]
    enabled = true
    rank = 8
    alpha = 8.0                      # update scale = alpha / rank
    targets = ["l_conv", "r_conv"]   # UNet children to adapt; "r_up" = transposed-conv upsamplers
    train_heads = true               # out_code_affs / out_fgbg_numinst stay fully trainable
    train_bias = false               # also train the biases of the adapted convs

Every targeted conv keeps its frozen ``weight``/``bias`` under the original
state_dict names and gains ``lora_A`` / ``lora_B`` sub-convs plus a
``lora_scaling`` buffer (transposed convs also ``lora_transposed``):

    y = conv(x) + scaling * B(A(x))     A: same kernel/stride/padding as conv, r channels
                                        B: 1x1x1, r -> out channels, zero-initialised

B = 0 at injection, so a freshly wrapped model computes exactly what the
pretrained one did. B is 1x1 and linear, so B(A(.)) is itself one conv with
weight B @ A: merge_lora_state_dict() folds it into ``weight`` and drops the
LoRA keys, which lets predict/decode load LoRA checkpoints into the plain model
(no inference overhead, no LoRA config needed there).

Fine-tune flow (train.py): a LoRA run starts from a checkpoint seeded by
sbatch/ppp/expv2/seed_finetune_ckpt.py (pretrain weights, fresh Adam, scheduler
at iteration N). gunpowder's TorchTrainExt loads the latest checkpoint strictly,
so prepare() loads the plain seed into the model before wrapping it, and
write_seed_checkpoint() rewrites that checkpoint in LoRA form (LoRA keys,
optimizer state over the trainable parameters only). The original seed is kept
as ``train_net_seed_prelora_<N>``. Resumes find LoRA keys and skip the rewrite.
Only the noenc model (train_code = false) is supported.
"""
import collections
import logging
import os
import shutil

import torch

logger = logging.getLogger(__name__)

DEFAULT_TARGETS = ("l_conv", "r_conv")
_LORA_KEY = ".lora_A.weight"


class LoRAConv3d(torch.nn.Conv3d):
    """Conv3d with a frozen base and a trainable low-rank update.

    Built by retyping an existing Conv3d in place (wrap_conv), so the base
    weight/bias, padding and memory format are untouched."""

    def forward(self, x):
        return super().forward(x) + self.lora_scaling * self.lora_B(self.lora_A(x))


class LoRAConvTranspose3d(torch.nn.ConvTranspose3d):
    """ConvTranspose3d (funlib Upsample "transposed_conv") with a low-rank update."""

    def forward(self, x, output_size=None):
        return (super().forward(x, output_size)
                + self.lora_scaling * self.lora_B(self.lora_A(x, output_size)))


def enabled(config):
    return bool((config.get("lora") or {}).get("enabled", False))


def has_lora(state_dict):
    return any(k.endswith(_LORA_KEY) for k in state_dict)


def wrap_conv(conv, rank, alpha):
    """Retype conv (Conv3d or ConvTranspose3d) in place into its LoRA variant."""
    if conv.groups != 1:
        raise ValueError("LoRA: grouped convs are not supported")
    dev, dt = conv.weight.device, conv.weight.dtype
    if type(conv) is torch.nn.Conv3d:
        conv.__class__ = LoRAConv3d
        conv.lora_A = torch.nn.Conv3d(
            conv.in_channels, rank, conv.kernel_size, stride=conv.stride,
            padding=conv.padding, dilation=conv.dilation, bias=False,
            padding_mode=conv.padding_mode, device=dev, dtype=dt)
    elif type(conv) is torch.nn.ConvTranspose3d:
        conv.__class__ = LoRAConvTranspose3d
        conv.lora_A = torch.nn.ConvTranspose3d(
            conv.in_channels, rank, conv.kernel_size, stride=conv.stride,
            padding=conv.padding, output_padding=conv.output_padding,
            dilation=conv.dilation, bias=False, device=dev, dtype=dt)
        conv.register_buffer("lora_transposed", torch.tensor(True, device=dev))
    else:
        raise TypeError(f"LoRA: cannot wrap {type(conv).__name__}")
    # A keeps nn.Conv's default (kaiming-uniform) init; B = 0 -> no change at start
    conv.lora_B = torch.nn.Conv3d(rank, conv.out_channels, 1, bias=False,
                                  device=dev, dtype=dt)
    torch.nn.init.zeros_(conv.lora_B.weight)
    conv.register_buffer("lora_scaling",
                         torch.tensor(float(alpha) / rank, dtype=torch.float32, device=dev))
    return conv


def inject_lora(model, cfg):
    """Wrap the targeted UNet convs of a UnetModelWrapper and set requires_grad.

    Call after the base weights are in place and before .to(device),
    use_channels_last() and compile (so the adapters follow them).
    Returns the list of wrapped module names (relative to model.unet)."""
    rank = int(cfg.get("rank", 8))
    alpha = float(cfg.get("alpha", rank))
    targets = tuple(cfg.get("targets", DEFAULT_TARGETS))
    if rank < 1:
        raise ValueError(f"LoRA: rank must be >= 1, got {rank}")
    children = dict(model.unet.named_children())
    unknown = [t for t in targets if t not in children]
    if unknown:
        raise ValueError(f"LoRA: unknown targets {unknown}; UNet children are "
                         f"{sorted(children)}")

    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for name, m in list(model.unet.named_modules()):
        if name.split(".")[0] not in targets:
            continue
        if type(m) in (torch.nn.Conv3d, torch.nn.ConvTranspose3d):
            wrap_conv(m, rank, alpha)
            wrapped.append(name)
            if cfg.get("train_bias", False) and m.bias is not None:
                m.bias.requires_grad_(True)
    if not wrapped:
        raise ValueError(f"LoRA: no Conv3d/ConvTranspose3d found under {targets}")

    for name, p in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            p.requires_grad_(True)
    if cfg.get("train_heads", True):
        for head in (model.out_code_affs, model.out_fgbg_numinst):
            for p in head.parameters():
                p.requires_grad_(True)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_lora = sum(p.numel() for n, p in model.named_parameters()
                 if ".lora_A." in n or ".lora_B." in n)
    n_base = sum(p.numel() for n, p in model.named_parameters()
                 if ".lora_A." not in n and ".lora_B." not in n)
    logger.info("LoRA: rank %d, alpha %g (scale %g), %d convs under %s; trainable %d "
                "params (%d LoRA) = %.2f%% of the %d base params", rank, alpha,
                alpha / rank, len(wrapped), list(targets), n_train, n_lora,
                100.0 * n_train / n_base, n_base)
    return wrapped


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def merge_lora_state_dict(state_dict):
    """Fold LoRA updates into the base weights; returns a plain state_dict.

    Works on any key prefix (e.g. SWA's ``module.``); state dicts without
    LoRA keys are returned unchanged."""
    if not has_lora(state_dict):
        return state_dict
    sd = collections.OrderedDict(state_dict)
    meta = getattr(state_dict, "_metadata", None)
    if meta is not None:
        sd._metadata = meta
    for key in [k for k in sd if k.endswith(_LORA_KEY)]:
        p = key[:-len("lora_A.weight")]
        A = sd.pop(p + "lora_A.weight").float()
        B = sd.pop(p + "lora_B.weight").float().flatten(1)      # (out, r)
        scale = float(sd.pop(p + "lora_scaling"))
        transposed = bool(sd.pop(p + "lora_transposed", False))
        W = sd[p + "weight"]
        if transposed:   # weight (in, out, k...), A (in, r, k...)
            delta = torch.einsum("ir...,or->io...", A, B.to(A.device))
        else:            # weight (out, in, k...), A (r, in, k...)
            delta = torch.einsum("or,ri...->oi...", B.to(A.device), A)
        sd[p + "weight"] = (W.float() + scale * delta.to(W.device)).to(W.dtype)
    return sd


def prepare(model, config, checkpoint_basename, trained_until):
    """train.py hook, right after init_layers(). No-op unless [training.lora] enabled.

    Returns (seed_path, seed_checkpoint) when the latest checkpoint is a plain
    seed that write_seed_checkpoint() must convert, else None."""
    if not enabled(config):
        return None
    if config.get("train_code"):
        raise NotImplementedError(
            "training.lora is only implemented for noenc (train_code = false)")
    if config.get("use_swa"):
        raise NotImplementedError("training.lora does not support use_swa")
    if trained_until == 0:
        raise RuntimeError(
            "training.lora fine-tunes a pretrained model, but %s has no checkpoint; "
            "seed one with sbatch/ppp/expv2/seed_finetune_ckpt.py first"
            % os.path.dirname(checkpoint_basename))
    from PatchPerPix.util import get_latest_checkpoint
    path, _ = get_latest_checkpoint(checkpoint_basename)
    ckpt = torch.load(path, map_location="cpu")
    seed = None
    if has_lora(ckpt["model_state_dict"]):
        logger.info("LoRA: resuming %s (already in LoRA form)", path)
        del ckpt
    else:
        logger.info("LoRA: loading plain seed weights from %s", path)
        model.load_state_dict(ckpt["model_state_dict"])
        seed = (path, ckpt)
    inject_lora(model, config["lora"])
    return seed


def write_seed_checkpoint(seed, model, opt, lr_scheduler):
    """Rewrite the plain seed checkpoint in LoRA form, before gunpowder loads it.

    model_state_dict -> the wrapped model (seed weights + fresh adapters);
    optimizer        -> fresh state over the trainable params, lr / initial_lr
                        taken from the seed's param groups (seed_finetune_ckpt.py);
    lr scheduler     -> the seed's state (base_lrs resized to the group count);
    no scaler state  -> fresh GradScaler, as for any seeded fine-tune."""
    path, ckpt = seed
    seed_groups = ckpt["optimizer_state_dict"]["param_groups"]
    opt_sd = opt.state_dict()
    for i, g in enumerate(opt_sd["param_groups"]):
        src = seed_groups[min(i, len(seed_groups) - 1)]
        g["lr"] = src["lr"]
        if "initial_lr" in src:
            g["initial_lr"] = src["initial_lr"]
    out = {"model_state_dict": model.state_dict(), "optimizer_state_dict": opt_sd}
    n_groups = len(opt_sd["param_groups"])
    if lr_scheduler is not None:
        if "lr_scheduler_state_dict" in ckpt:
            sd = dict(ckpt["lr_scheduler_state_dict"])
            for k in ("base_lrs", "_last_lr", "lr_lambdas"):
                if isinstance(sd.get(k), list) and sd[k]:
                    sd[k] = [sd[k][min(i, len(sd[k]) - 1)] for i in range(n_groups)]
            out["lr_scheduler_state_dict"] = sd
        else:
            logger.warning("LoRA: seed %s has no lr_scheduler_state_dict; using a "
                           "fresh scheduler state", path)
            out["lr_scheduler_state_dict"] = lr_scheduler.state_dict()

    peak = seed_groups[0].get("initial_lr", seed_groups[0]["lr"])
    cfg_lr = opt.defaults.get("lr")
    if cfg_lr is not None and abs(peak - cfg_lr) > 1e-12 * max(abs(peak), 1.0):
        logger.warning("LoRA: seed peak lr %g (seed_finetune_ckpt.py --lr) differs from "
                       "[optimizer.optimizer] lr %g; the seed's value is used", peak, cfg_lr)

    # neither name may match "<basename>_checkpoint_*" (get_latest_checkpoint's glob)
    d, base = os.path.split(path)
    backup = os.path.join(d, base.replace("_checkpoint_", "_seed_prelora_"))
    tmp = os.path.join(d, base.replace("_checkpoint_", "_lora_tmp_"))
    torch.save(out, tmp)
    if not os.path.exists(backup):
        try:
            os.link(path, backup)
        except OSError:
            shutil.copy2(path, backup)
    os.replace(tmp, path)
    logger.info("LoRA: rewrote seed %s in LoRA form (original kept as %s); peak lr %g, "
                "first-step lr %g", path, backup, peak, opt_sd["param_groups"][0]["lr"])
