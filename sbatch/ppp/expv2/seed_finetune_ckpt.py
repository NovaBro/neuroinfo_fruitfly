#!/usr/bin/env python3
"""Seed a fine-tune experiment from a pretrained PPP checkpoint (experiment_v2 EXP 1.3/1.4).

Writes ``<dst_dir>/train_net_checkpoint_<N>`` so gunpowder's TorchTrainExt resumes at
iteration N with:
  * model_state_dict   copied from the source checkpoint (pretrained weights);
  * optimizer_state_dict  fresh Adam (empty moments), param_groups lr set to the
    first warmup step (peak * factor(N)), initial_lr = peak;
  * lr_scheduler_state_dict  LambdaLR positioned at last_epoch = N with
    base_lrs = [peak] and lr_lambdas = [None], so the WarmupCosine lambda built
    from the fine-tune config (start_iter = N) is kept, not the pretrain's;
  * no scaler_state_dict  -> fresh GradScaler (gunpowder only warns).

The fine-tune config must set [optimizer.optimizer] lr = peak and
[optimizer.lr_scheduler] lr_scheduler = "WarmupCosine", start_iter = N.

Run inside the ppp container (needs torch):
  python sbatch/ppp/expv2/seed_finetune_ckpt.py --src <ckpt> --dst-dir <exp>/train \
      --iteration N --lr 1e-4 --warmup-frac 0.05 --ft-iters 6000
"""
import argparse
import math
import os
import sys

import torch


def factor(step, warmup, total, min_ratio, start):
    """Same as setup01/train.py make_warmup_cosine.__call__."""
    step = max(0, step - start)
    if step < warmup:
        return (step + 1) / warmup
    prog = min(1.0, (step - warmup) / (total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst-dir", required=True)
    ap.add_argument("--iteration", type=int, required=True, help="N: source checkpoint iteration")
    ap.add_argument("--lr", type=float, required=True, help="fine-tune peak lr")
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--ft-iters", type=int, required=True)
    ap.add_argument("--min-lr-ratio", type=float, default=0.0)
    args = ap.parse_args()

    n = args.iteration
    dst = os.path.join(args.dst_dir, f"train_net_checkpoint_{n}")
    if os.path.exists(dst):
        print(f"exists, not overwriting: {dst}")
        return 0
    src = torch.load(args.src, map_location="cpu")
    if "model_state_dict" not in src or "optimizer_state_dict" not in src:
        print(f"error: {args.src} lacks model/optimizer state", file=sys.stderr)
        return 1

    warmup = int(round(args.warmup_frac * args.ft_iters))
    total = max(warmup + 1, args.ft_iters)
    warmup = max(0, warmup)
    lr0 = args.lr * factor(n, warmup, total, args.min_lr_ratio, n)

    groups = []
    for g in src["optimizer_state_dict"]["param_groups"]:
        g = dict(g)
        g["lr"] = lr0
        g["initial_lr"] = args.lr
        groups.append(g)
    opt_state = {"state": {}, "param_groups": groups}

    # Build a real LambdaLR state dict (keys match this torch version), then reposition it.
    dummy = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(dummy, lr_lambda=lambda s: 1.0)
    sd = sched.state_dict()
    sd["base_lrs"] = [args.lr] * len(groups)
    sd["last_epoch"] = n
    sd["_step_count"] = n + 1
    sd["_last_lr"] = [lr0] * len(groups)
    sd["lr_lambdas"] = [None] * len(groups)

    out = {"model_state_dict": src["model_state_dict"],
           "optimizer_state_dict": opt_state,
           "lr_scheduler_state_dict": sd}
    os.makedirs(args.dst_dir, exist_ok=True)
    tmp = dst + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, dst)
    print(f"seeded {dst} from {args.src}: peak lr {args.lr}, warmup {warmup}/{total} its, "
          f"first-step lr {lr0:.3e}, fresh Adam + GradScaler")
    return 0


if __name__ == "__main__":
    sys.exit(main())
