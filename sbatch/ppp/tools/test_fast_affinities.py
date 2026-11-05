"""Exact-equality + timing check: seg_to_affgraph_3d_multi_torch (+ the noenc crop)
vs torch_model.seg_to_affgraph_3d_multi_torch_cropped (training.fast_affinities).

Run inside the ppp container from PatchPerPix/experiments/ (GPU):
  python -u ../../sbatch/ppp/tools/test_fast_affinities.py
Exits non-zero on any mismatch.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
from PatchPerPix.util import seg_to_affgraph_3d_multi_torch  # noqa: E402
from flylight.setups.setup01.torch_model import (  # noqa: E402
    seg_to_affgraph_3d_multi_torch_cropped)


def nhood_7():
    return [[k, i, j] for k in range(-3, 4) for i in range(-3, 4) for j in range(-3, 4)]


def reference(seg, nh_list, device):
    nh = torch.as_tensor(nh_list, device=device)  # CUDA tensor, as in torch_model
    p = 3
    return seg_to_affgraph_3d_multi_torch(seg, nh, device)[:, :, p:-p, p:-p, p:-p]


def make_seg(B, C, S, device, kind, g):
    if kind == "sparse":  # ~5% fg, ids 1..5 (as profile_ppp_step.make_inputs)
        lab = (torch.rand((B, C, S, S, S), device=device, generator=g) < 0.05).int()
        return lab * torch.randint(1, 6, lab.shape, device=device, generator=g).int()
    if kind == "blocks":  # large constant regions -> many true affinities
        small = torch.randint(0, 4, (B, C, S // 6 + 1, S // 6 + 1, S // 6 + 1),
                              device=device, generator=g).int()
        up = small.repeat_interleave(6, 2).repeat_interleave(6, 3).repeat_interleave(6, 4)
        return up[:, :, :S, :S, :S].contiguous()
    if kind == "overlap":  # channel 1 copies channel 0 ids in half the volume
        a = make_seg(B, 1, S, device, "blocks", g)
        b = a.clone()
        b[..., S // 2:] = 0
        return torch.cat([a, b], dim=1)
    raise ValueError(kind)


def main():
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(0)
    nh = nhood_7()
    ok = True
    for kind in ("sparse", "blocks", "overlap"):
        for (B, S) in ((2, 20), (3, 33)):
            seg = make_seg(B, 2, S, device, kind, g)
            ref = reference(seg, nh, device)
            new = seg_to_affgraph_3d_multi_torch_cropped(seg, nh, device)
            same = (ref.shape == new.shape and ref.dtype == new.dtype
                    and torch.equal(ref, new))
            print(f"{kind:8s} B={B} S={S} shape={tuple(new.shape)} "
                  f"pos_frac={new.mean().item():.4f} equal={same}")
            ok &= same

    # timing at training size (bs18, gt_labels side = out + 6); out=52 is close to the
    # 140^3 valid-input output; exact value only matters for scale
    B, S = 18, 58
    seg = make_seg(B, 2, S, device, "sparse", g)
    for name, fn in (("reference", lambda: reference(seg, nh, device)),
                     ("cropped", lambda: seg_to_affgraph_3d_multi_torch_cropped(
                         seg, nh, device))):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        print(f"time {name:10s} bs{B} side{S}: {(time.perf_counter() - t0) / 5:.3f} s")

    print("ALL EQUAL" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
