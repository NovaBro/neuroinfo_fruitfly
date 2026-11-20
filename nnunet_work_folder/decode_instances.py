#!/usr/bin/env python
"""nnU-Net region probabilities -> FISBe instance label maps.

Input : <npz-dir>/<case>.npz with 'probabilities' = [p_neuron, p_contact, p_overlap] (float, ZYX), as written
        by `nnUNetv2_train ... --npz` (fold_0/validation*/) or `nnUNetv2_predict ... --save_probabilities`.
        Regions are sigmoid outputs: neuron = labels {1,2,3}, contact = {2,3}, overlap = {3}
        (see fisbe_to_nnunet.py).
Output: <out>/<case>.zarr  (zarr v2)  volumes/pred_instance  uint32 [Z,Y,X]  (0 = background), the layout
        evalinstseg / score_fisbe.py read. zarr attrs record the decode parameters.

Modes (all connectivity is 26 = 3x3x3; 6-connectivity shatters 1-3 voxel neurites):
  cc      : instances = CC26(p_neuron > t_fg).                         (plain baseline)
  contact : fg = p_neuron > t_fg; contact = fg & (p_contact > t_c);
            cores = CC26(fg & ~contact), cores smaller than min_core voxels dropped;
            inside each fg component with >= 2 cores, every fg voxel is flooded from the cores
            (skimage watershed on a flat image, mask = component, 26-conn -> geodesic nearest core);
            fg components with <= 1 core stay one instance.

Runs in env/evaluate.ext3 (numpy, scipy, scikit-image, zarr 2) or env/nnunet.ext3.
  python nnunet_work_folder/decode_instances.py --npz-dir <dir> --out <dir> --mode contact \
      --t-fg 0.5 --t-c 0.5 --min-core 20 [--gt-dir fisbe/completely/val] [--workers 4]
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import zarr
from scipy import ndimage as ndi
from skimage.segmentation import watershed

S26 = np.ones((3, 3, 3), bool)


def decode(p, mode, t_fg=0.5, t_c=0.5, min_core=20):
    fg = p[0] > t_fg
    lab_fg, _ = ndi.label(fg, S26)
    if mode == 'cc':
        return lab_fg.astype(np.uint32)
    contact = fg & (p[1] > t_c)
    cores, n = ndi.label(fg & ~contact, S26)
    if n:
        sz = np.bincount(cores.ravel())
        small = sz < min_core
        small[0] = False
        cores[small[cores]] = 0
    out = np.zeros(fg.shape, np.uint32)
    nxt = 1
    for i, sl in enumerate(ndi.find_objects(lab_fg), 1):  # per fg component (its bbox) to bound cost
        if sl is None:
            continue
        m = lab_fg[sl] == i
        c = np.where(m, cores[sl], 0)
        ids = np.unique(c[c > 0])
        o = out[sl]  # view into out
        if len(ids) <= 1:
            o[m] = nxt
            nxt += 1
            continue
        ws = watershed(np.zeros(m.shape, np.uint8), markers=c, mask=m, connectivity=3)
        _, inv = np.unique(ws[m], return_inverse=True)
        o[m] = (inv.ravel() + nxt).astype(np.uint32)
        nxt += int(inv.max()) + 1
    return out


def gt_shape(gt_dir, case):
    with open(os.path.join(gt_dir, case + '.zarr', 'volumes', 'gt_instances', '.zarray')) as fh:
        return tuple(json.load(fh)['shape'][1:])


def run_case(job):
    f, a = job
    case = os.path.basename(f)[:-len('.npz')]
    out_z = os.path.join(a.out, case + '.zarr')
    if not a.overwrite and os.path.isfile(os.path.join(out_z, 'volumes', 'pred_instance', '.zarray')):
        return case, 'skipped', None, None
    p = np.load(f)['probabilities']
    assert p.ndim == 4 and p.shape[0] >= 2, (case, p.shape)
    if a.gt_dir:
        gs = gt_shape(a.gt_dir, case)
        assert tuple(p.shape[1:]) == gs, f'{case}: prediction {p.shape[1:]} != GT {gs}'
    lab = decode(p, a.mode, a.t_fg, a.t_c, a.min_core)
    g = zarr.open_group(out_z, mode='w')
    ds = g.require_group('volumes').create_dataset('pred_instance', data=lab, chunks=(64, 64, 64),
                                                   overwrite=True)
    ds.attrs.update({'mode': a.mode, 't_fg': a.t_fg, 't_c': a.t_c, 'min_core': a.min_core,
                     'source_npz': os.path.abspath(f)})
    sizes = np.bincount(lab.ravel())[1:]
    return case, 'written', int(lab.max()), int((sizes > 1500).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=['cc', 'contact'], default='contact')
    ap.add_argument('--t-fg', type=float, default=0.5)
    ap.add_argument('--t-c', type=float, default=0.5)
    ap.add_argument('--min-core', type=int, default=20)
    ap.add_argument('--gt-dir', default=None, help='assert prediction ZYX == GT ZYX (evalinstseg crops silently)')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.npz_dir, '*.npz')))
    assert files, f'no .npz in {a.npz_dir}'
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.workers) as pool:
        for case, status, n, n_big in pool.imap_unordered(run_case, [(f, a) for f in files]):
            print(f'{case}: {status} n_inst={n} n_inst_gt1500vox={n_big}', flush=True)
    with open(os.path.join(a.out, 'decode_params.json'), 'w') as fh:
        json.dump({k: v for k, v in vars(a).items()}, fh, indent=1)


if __name__ == '__main__':
    main()
