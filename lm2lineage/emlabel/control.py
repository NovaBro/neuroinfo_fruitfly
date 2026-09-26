"""Negative control: pair each MCFO stack with the EM skeleton of an unrelated neuron."""
import pandas as pd, numpy as np, io, os
from emlabel.nb import NB, read_h5j
from emlabel.propose import propose
nb=NB('cache')
d=pd.read_csv('proposals_demo/proposals.csv')
rows=[]
n=len(d)
for i,r in d.iterrows():
    stack_fp=os.path.join('cache','stacks',os.path.basename(r['stack'].replace('+',' ')))
    stack,kinds,vox,_=read_h5j(stack_fp)
    for shift in (7, 13):
        o=d.iloc[(i+shift)%n]
        if o.hemilineage==r.hemilineage: continue
        p=propose(stack,kinds,vox,nb.swc(o['em_swc']),mirrored_hint=bool(o['mirrored_meta']))
        if p is None: continue
        rows.append(dict(stack_of=r.proposal_id, em_of=o.proposal_id, coverage=p.coverage, precision=p.precision, presence=p.presence))
    del stack
c=pd.DataFrame(rows); c.to_csv('proposals_demo/negative_control.csv',index=False)
def flag(s): return 'likely_good' if (s.coverage>=0.6 and s.precision>=0.6 and s.presence>=0.7) else ('likely_bad' if (s.coverage<0.3 or s.presence<0.4) else 'check')
c['flag']=c.apply(flag,axis=1)
print('TRUE matches (n=%d):'%len(d)); print(d[['coverage','precision','presence']].describe().loc[['mean','25%','50%','75%']].round(2))
print('SHUFFLED (n=%d):'%len(c)); print(c[['coverage','precision','presence']].describe().loc[['mean','25%','50%','75%']].round(2))
print(c.flag.value_counts())
