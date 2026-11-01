"""Equivalence test: vectorized seg_to_affgraph_*_multi_torch_code vs the original loop version.

Run (CPU is enough):
  singularity exec --overlay env/ppp.ext3:ro <sif> /bin/bash -c \
    'source /ext3/env.sh; conda activate ppp; python sbatch/ppp/tools/test_affgraph_vectorized.py'
"""
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "PatchPerPix"))
from PatchPerPix.util.train_util import (
    seg_to_affgraph_2d_multi_torch_code, seg_to_affgraph_3d_multi_torch_code)


def reference(seg, nhood, device, ps, psH, samples_loc):
    nd = nhood.shape[1]
    patches = torch.zeros((len(samples_loc), seg.shape[1]) + (ps,) * nd, dtype=seg.dtype)
    for i, loc in enumerate(samples_loc):
        sl = tuple(slice(int(loc[d + 1]), int(loc[d + 1]) + ps) for d in range(nd))
        patches[i] = seg[(int(loc[0]), slice(None)) + sl]
    fg = torch.any(patches, dim=1).to(torch.int32)
    aff = torch.zeros((len(samples_loc), nhood.shape[0]))
    c = (slice(None), slice(None)) + (psH,) * nd
    for e in range(nhood.shape[0]):
        o = (slice(None), slice(None)) + tuple(int(x) for x in nhood[e])
        t1 = (patches[c] == patches[o]).to(torch.int32)
        t1[patches[c] == 0] = 0
        aff[:, e] = torch.any(t1, dim=1) * fg[(slice(None),) + (psH,) * nd] \
            * fg[(slice(None),) + tuple(int(x) for x in nhood[e])]
    return aff


def check(nd, fn):
    g = torch.Generator().manual_seed(0)
    ps, psH, B, C, S = 7, 3, 3, 2, 24
    seg = torch.randint(0, 4, (B, C) + (S,) * nd, generator=g)
    seg[torch.rand(seg.shape, generator=g) < 0.5] = 0
    grid = torch.stack(torch.meshgrid(*[torch.arange(ps)] * nd, indexing="ij"), -1).reshape(-1, nd)
    nhood = grid
    n = 200
    loc = torch.cat([torch.randint(0, B, (n, 1), generator=g)] +
                    [torch.randint(0, S - ps + 1, (n, 1), generator=g) for _ in range(nd)], 1)
    loc[0, 1:] = 0
    loc[1, 1:] = S - ps
    a = reference(seg, nhood, "cpu", ps, psH, loc)
    b = fn(seg, nhood, "cpu", ps, psH, loc.int())
    assert torch.equal(a, b), f"{nd}D mismatch"
    print(f"{nd}D ok", tuple(b.shape))


check(3, seg_to_affgraph_3d_multi_torch_code)
check(2, seg_to_affgraph_2d_multi_torch_code)
