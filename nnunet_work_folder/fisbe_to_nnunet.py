#!/usr/bin/env python
"""FISBe zarr -> nnU-Net v2 raw dataset (Tiff3DIO), with region labels neuron / contact / overlap.

Images  : imagesTr/<sample>_000{0,1,2}.tif  = volumes/raw channels (uint16, zlib), + <sample>.json spacing
Labels  : labelsTr/<sample>.tif (uint8 ZYX), + <sample>.json spacing
            0 background
            1 neuron   : voxel in >= 1 GT instance
            2 contact  : neuron voxel whose 26-neighbourhood (or itself) touches a *different* instance
            3 overlap  : voxel in >= 2 GT instances
          Nested regions (overlap c contact c neuron), declared in dataset.json as
            neuron=[1,2,3], contact=[2,3], overlap=3, regions_class_order=[1,2,3]
          -> nnU-Net trains 3 sigmoid outputs; decode_instances.py turns them into instances.
          Labels are built from volumes/gt_instances (NOT gt_fg_rm_5, whose opening erases thin neurites).
Test    : imagesTs/<sample>_000{0,1,2}.tif + <sample>.json (prediction input only; nnU-Net never trains on it)
Split   : splits_final.json = [{"train": <train split>, "val": <val split>}]  -> fold 0 = the FISBe split.
          nnU-Net copies it into nnUNet_preprocessed/<dataset>/ during planning.

Spacing : 0.44 um isotropic (FISBe paper sec. 2.1; the zarrs carry no spacing). Tiff3DIO reads the
          <case>.json sidecar next to every image; a missing sidecar silently means spacing (1,1,1),
          which would make nnU-Net resample by 2.27x per axis -> every image folder gets sidecars.

Usage (inside env/nnunet.ext3, see sbatch/nnunet/nnunet_prep_sbatch.sh):
  python nnunet_work_folder/fisbe_to_nnunet.py --fisbe fisbe/completely \
      --out "$nnUNet_raw/Dataset501_FISBeCompletely" --workers 8
  --crop Z Y X  : smoke test only, centre a crop on an overlap/multi-instance voxel.
  --cases a,b   : restrict to these sample names.
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import tifffile
import zarr
from scipy import ndimage as ndi

SPACING = [0.44, 0.44, 0.44]  # um, ZYX
S26 = np.ones((3, 3, 3), bool)
LABELS = {'background': 0, 'neuron': [1, 2, 3], 'contact': [2, 3], 'overlap': 3}
REGIONS_CLASS_ORDER = [1, 2, 3]


def find_center(numinst, shape):
    """Centre of a crop of `shape`, on a voxel covered by >= 2 instances if one exists (smoke tests)."""
    for z0 in range(0, numinst.shape[0], 32):
        idx = np.argwhere(numinst[z0:z0 + 32] >= 2)
        if len(idx):
            c = idx[len(idx) // 2]
            c[0] += z0
            return [int(np.clip(c[i], shape[i] // 2, numinst.shape[i] - shape[i] // 2)) for i in range(3)]
    return [s // 2 for s in numinst.shape]


def build_label(gt_arr, sl):
    """gt_arr: zarr [N,Z,Y,X] (one binary-ish channel per neuron); sl: spatial slice. Returns uint8 ZYX."""
    shape = tuple(len(range(*s.indices(n))) for s, n in zip(sl, gt_arr.shape[1:]))
    fg_cnt = np.zeros(shape, np.uint8)
    near_cnt = np.zeros(shape, np.uint8)
    for i in range(gt_arr.shape[0]):
        m = gt_arr[(i,) + sl] > 0          # read one instance at a time to bound memory
        if not m.any():
            continue
        fg_cnt += m
        bb = ndi.find_objects(m.astype(np.uint8))[0]
        bb = tuple(slice(max(b.start - 1, 0), min(b.stop + 1, n)) for b, n in zip(bb, m.shape))
        near_cnt[bb] += ndi.binary_dilation(m[bb], S26)
    lab = (fg_cnt > 0).astype(np.uint8)
    lab[(fg_cnt > 0) & (near_cnt >= 2)] = 2
    lab[fg_cnt >= 2] = 3
    return lab


def write_sidecar(folder, name):
    with open(os.path.join(folder, f'{name}.json'), 'w') as fh:
        json.dump({'spacing': SPACING}, fh)


def convert_case(job):
    zpath, out_img, out_lab, crop, overwrite = job
    name = os.path.basename(zpath)[:-len('.zarr')]
    done_marker = os.path.join(out_lab or out_img, f'{name}.json')
    if not overwrite and os.path.isfile(done_marker) and os.path.isfile(os.path.join(out_img, f'{name}.json')):
        return name, 'skipped', None
    g = zarr.open(zpath, mode='r')
    raw = g['volumes/raw']
    if crop:
        c = find_center(g['volumes/gt_numinst'], crop)
        sl = tuple(slice(ci - s // 2, ci - s // 2 + s) for ci, s in zip(c, crop))
    else:
        sl = (slice(None),) * 3
    r = raw[(slice(None),) + sl]  # [3,Z,Y,X] uint16
    assert r.ndim == 4 and r.shape[0] == 3, (name, r.shape)
    for ch in range(r.shape[0]):
        tifffile.imwrite(os.path.join(out_img, f'{name}_{ch:04d}.tif'), r[ch], compression='zlib')
    write_sidecar(out_img, name)  # written last: its presence marks the case as complete
    counts = None
    if out_lab:
        gt = g['volumes/gt_instances']
        assert tuple(gt.shape[1:]) == tuple(raw.shape[1:]), (name, gt.shape, raw.shape)
        lab = build_label(gt, sl)
        assert lab.shape == r.shape[1:], (name, lab.shape, r.shape)
        tifffile.imwrite(os.path.join(out_lab, f'{name}.tif'), lab, compression='zlib')
        write_sidecar(out_lab, name)
        counts = np.bincount(lab.ravel(), minlength=4).tolist()
    return name, 'written', counts


def check_folder(folder, names, n_channels):
    """Pre-flight: every case has all channel tifs and a spacing sidecar; nothing extra."""
    for n in names:
        for ch in range(n_channels):
            f = os.path.join(folder, f'{n}_{ch:04d}.tif' if n_channels else f'{n}.tif')
            assert os.path.isfile(f), f'missing {f}'
        if n_channels == 0:
            assert os.path.isfile(os.path.join(folder, f'{n}.tif')), f'missing label for {n}'
        assert os.path.isfile(os.path.join(folder, f'{n}.json')), f'missing spacing sidecar for {n} in {folder}'
    sidecars = glob.glob(os.path.join(folder, '*.json'))
    assert len(sidecars) == len(names), f'{folder}: {len(sidecars)} sidecars for {len(names)} cases'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fisbe', required=True, help='e.g. fisbe/completely (contains train/ val/ test/)')
    p.add_argument('--out', required=True, help='$nnUNet_raw/DatasetXXX_Name')
    p.add_argument('--train', default='train')
    p.add_argument('--val', default='val')
    p.add_argument('--test', default='test', help="'' to skip test images")
    p.add_argument('--crop', type=int, nargs=3, default=None)
    p.add_argument('--cases', default=None)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()

    keep = set(a.cases.split(',')) if a.cases else None

    def sel(split):
        if not split:
            return []
        fs = sorted(glob.glob(os.path.join(a.fisbe, split, '*.zarr')))
        return [f for f in fs if keep is None or os.path.basename(f)[:-5] in keep]

    img_tr, lab_tr, img_ts = (os.path.join(a.out, d) for d in ('imagesTr', 'labelsTr', 'imagesTs'))
    for d in (img_tr, lab_tr, img_ts):
        os.makedirs(d, exist_ok=True)

    tr, va, ts = sel(a.train), sel(a.val), sel(a.test)
    assert tr and va, f'no train/val zarrs under {a.fisbe}'
    jobs = [(f, img_tr, lab_tr, a.crop, a.overwrite) for f in tr + va] + \
           [(f, img_ts, None, a.crop, a.overwrite) for f in ts]
    stats = {}
    with Pool(a.workers) as pool:
        for name, status, counts in pool.imap_unordered(convert_case, jobs):
            print(f'{name}: {status} label_counts(bg,neuron,contact,overlap)={counts}', flush=True)
            if counts is not None:
                stats[name] = counts

    names = lambda fs: [os.path.basename(f)[:-5] for f in fs]
    tr_n, va_n, ts_n = names(tr), names(va), names(ts)

    dataset = {
        'channel_names': {'0': 'mcfo_c0', '1': 'mcfo_c1', '2': 'mcfo_c2'},  # non-reserved -> per-image z-score
        'labels': LABELS,
        'regions_class_order': REGIONS_CLASS_ORDER,
        'numTraining': len(tr_n) + len(va_n),
        'file_ending': '.tif',
        'overwrite_image_reader_writer': 'Tiff3DIO',
        'name': os.path.basename(os.path.normpath(a.out)),
        'description': 'FISBe completely; region labels neuron/contact/overlap from volumes/gt_instances; '
                       '0.44 um isotropic; fold 0 = FISBe train/val split',
        'reference': 'https://kainmueller-lab.github.io/fisbe/',
    }
    with open(os.path.join(a.out, 'dataset.json'), 'w') as fh:
        json.dump(dataset, fh, indent=2, sort_keys=False)
    with open(os.path.join(a.out, 'splits_final.json'), 'w') as fh:
        json.dump([{'train': tr_n, 'val': va_n}], fh, indent=2)

    # ---- pre-flight checks ----
    check_folder(img_tr, tr_n + va_n, 3)
    check_folder(lab_tr, tr_n + va_n, 0)
    if ts_n:
        check_folder(img_ts, ts_n, 3)
    for n in tr_n + va_n:
        lab = tifffile.imread(os.path.join(lab_tr, f'{n}.tif'))
        img_shape = tifffile.TiffFile(os.path.join(img_tr, f'{n}_0000.tif')).series[0].shape
        assert lab.shape == tuple(img_shape), (n, lab.shape, img_shape)
        vals = np.unique(lab)
        assert set(vals.tolist()) <= {0, 1, 2, 3}, (n, vals)
        assert 1 in vals or 2 in vals, f'{n}: empty label'
    if stats:
        with open(os.path.join(a.out, 'conversion_label_counts.json'), 'w') as fh:
            json.dump(stats, fh, indent=1, sort_keys=True)
    print(f'OK: {len(tr_n)} train + {len(va_n)} val (imagesTr/labelsTr), {len(ts_n)} test (imagesTs) -> {a.out}')


if __name__ == '__main__':
    main()
