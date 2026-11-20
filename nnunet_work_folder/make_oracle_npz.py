#!/usr/bin/env python
"""Sanity/ceiling tool: turn converted region labels (labelsTr/<case>.tif) into fake nnU-Net probability
files (<out>/<case>.npz, key 'probabilities' = [neuron, contact, overlap] as 0/1 float16, ZYX), so the
decode -> score path can be tested before a model exists and the decode modes get an oracle ceiling.

  python nnunet_work_folder/make_oracle_npz.py --labels $nnUNet_raw/Dataset501_FISBeCompletely/labelsTr \
      --cases-from fisbe/completely/val --out metrics/nnunet/oracle/val_npz
"""
import argparse
import glob
import os

import numpy as np
import tifffile

ap = argparse.ArgumentParser()
ap.add_argument('--labels', required=True)
ap.add_argument('--cases-from', required=True, help='folder of <case>.zarr selecting which labels to use')
ap.add_argument('--out', required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
cases = sorted(os.path.basename(z)[:-5] for z in glob.glob(os.path.join(a.cases_from, '*.zarr')))
for c in cases:
    lab = tifffile.imread(os.path.join(a.labels, c + '.tif'))
    p = np.stack([lab >= 1, lab >= 2, lab == 3]).astype(np.float16)
    np.savez_compressed(os.path.join(a.out, c + '.npz'), probabilities=p)
    print(c, lab.shape, 'voxels neuron/contact/overlap', [int(x.sum()) for x in p.astype(bool)], flush=True)
