import pandas as pd, numpy as np, os
from emlabel.nb import NB, read_h5j
from emlabel.propose import propose, Params
nb=NB('cache')
d=pd.read_csv('splitset_demo/splitset.csv'); lines=pd.read_csv('split/split_lines_hemilineage_avail.csv').set_index('line')
em=d[d.mask_method=='em_guided'].reset_index(drop=True)
def swcs_of(line):
    ids=[int(i) for i in str(lines.loc[line,'em_ids']).split()][:2]
    out=[]
    for b in ids:
        e=[x for x in nb.em_images(b) if x['libraryName']=='FlyEM_MANC_v1.2.1']
        if e and e[0]['files'].get('AlignedBodySWC'): out.append(nb.swc(e[0]['files']['AlignedBodySWC']))
    return out
rows=[]
for i,r in em.iterrows():
    other=em.iloc[(i+5)%len(em)]
    if other.hemilineage==r.hemilineage: other=em.iloc[(i+9)%len(em)]
    fp=os.path.join('cache','stacks',os.path.basename(r.stack.replace('+',' ')))
    stack,kinds,vox,_=read_h5j(fp)
    pres=[]
    for swc in swcs_of(other.line):
        best=max((propose(stack,kinds,vox,swc,p=Params(z_thresh=6.0),orientations=(m,)) for m in (False,True)), key=lambda p: -1 if p is None else p.presence)
        if best is not None: pres.append(best.presence)
    if pres: rows.append(dict(image=r.sample_id, em_from=other.line, true_presence=r.em_presence, shuffled_presence=float(np.mean(pres))))
    del stack
c=pd.DataFrame(rows); c.to_csv('splitset_demo/negative_control.csv',index=False)
print(c.describe().round(2)); print('shuffled >=0.7:', (c.shuffled_presence>=0.7).sum(), 'of', len(c)); print('true >=0.7:', (em.em_presence>=0.7).sum(),'of',len(em))
