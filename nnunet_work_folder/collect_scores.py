#!/usr/bin/env python
"""Collect score_fisbe.py summaries under a folder into one ranked table (stdlib only).

  python nnunet_work_folder/collect_scores.py metrics/nnunet/fisbe501_resencL_f0/val_tune [--rm 1500]

Finds <root>/<name>/eval_rm<rm>/fisbe_summary_rm<rm>.csv, prints MEAN and POOLED S/avF1/C/FS/FM/nPred per
setting sorted by MEAN S, and writes <root>/ranking_rm<rm>.csv.
"""
import argparse
import csv
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument('root')
ap.add_argument('--rm', type=int, default=1500)
a = ap.parse_args()

rows = []
for f in glob.glob(os.path.join(a.root, '*', f'eval_rm{a.rm}', f'fisbe_summary_rm{a.rm}.csv')):
    name = f.split(os.sep)[-3]
    agg = {r['sample']: r for r in csv.DictReader(open(f))}
    if 'MEAN' not in agg or 'POOLED' not in agg:
        continue
    m, p = agg['MEAN'], agg['POOLED']
    rows.append([name, float(m['S']), float(p['S']), float(m['avF1']), float(m['C']),
                 float(p['FS']), float(p['FM']), float(p['nPred']), float(p['nGT'])])
rows.sort(key=lambda r: -r[1])
hdr = ['name', 'S_mean', 'S_pooled', 'avF1_mean', 'C_mean', 'FS', 'FM', 'nPred', 'nGT']
print(('%-26s' + ' %9s' * 8) % tuple(hdr))
for r in rows:
    print(('%-26s' + ' %9.4f' * 4 + ' %9d' * 4) % tuple(r[:5] + [int(x) for x in r[5:]]))
out = os.path.join(a.root, f'ranking_rm{a.rm}.csv')
with open(out, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(hdr)
    w.writerows(rows)
print('wrote', out, f'({len(rows)} settings)')
