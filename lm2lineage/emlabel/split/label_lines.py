"""Split-GAL4 line -> hemilineage table (run from anywhere; inputs are fetched if missing)."""
import os, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "elife-98405-fig1-data1-v2.xlsx")
ANN = os.path.join(HERE, "..", "cache", "male_cns_annotations.feather")
for url, fp in (("https://cdn.elifesciences.org/articles/98405/elife-98405-fig1-data1-v2.xlsx", XLSX),
                ("https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
                 "body-annotations-male-cns-v1.0-minconf-0.5.feather", ANN)):
    if not os.path.exists(fp):
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        urllib.request.urlretrieve(url, fp)
import pandas as pd, re, numpy as np
x=pd.read_excel(XLSX); x.columns=[c.replace('\n',' ') for c in x.columns]
ad=x[x['Adult/ Larval']=='Adult'].copy()
ann=pd.read_feather(ANN)
m=ann[ann.mancBodyid.notna()].copy(); m['mancBodyid']=m.mancBodyid.astype(np.int64)
m=m.drop_duplicates('mancBodyid')
b2=m.set_index('mancBodyid')
type2hl=m.groupby('mancType').trumanHl.agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None)
type2sc=m.groupby('mancType').superclass.agg(lambda s: s.mode().iloc[0])
def norm_hl(h):
    mm=re.fullmatch(r'(\d{1,2})([AB])',h); return f"{int(mm.group(1)):02d}{mm.group(2)}" if mm else h
def ok(v): return isinstance(v,str) and v not in ('nan','')
rows=[]
for _,r in ad.iterrows():
    ct=str(r['Cell types']) if pd.notna(r['Cell types']) else ''
    ids=[int(t) for t in re.findall(r'\d{4,7}', str(r['EM Body IDs']))] if pd.notna(r['EM Body IDs']) else []
    found=[i for i in ids if i in b2.index]
    hl_ids=[b2.loc[i,'trumanHl'] for i in found]
    types_ids=[b2.loc[i,'mancType'] for i in found]
    hl_tok=[norm_hl(t) for t in re.findall(r'(?<![A-Za-z0-9])(\d{1,2}[AB])(?![A-Za-z0-9])', ct)]
    names=[c.strip() for c in re.split(r'[,;]| and/or ', ct) if c.strip()]
    hl_names=[type2hl[n] for n in names if n in type2hl.index and ok(type2hl[n])]
    sc=set(b2.loc[i,'superclass'] for i in found) | set(type2sc[n] for n in names if n in type2sc.index)
    rows.append(dict(line=r['Line'], cell_types=ct, em_ids=' '.join(map(str,ids)), em_ids_in_manc=len(found),
        manc_types_from_ids=', '.join(sorted(set(t for t in types_ids if ok(t)))),
        hl_from_ids=', '.join(sorted(set(h for h in hl_ids if ok(h)))),
        hl_from_label=', '.join(sorted(set(hl_tok))), hl_from_typename=', '.join(sorted(set(hl_names))),
        superclass=', '.join(sorted(x for x in sc if ok(x))), vnc_only=r['VNC only'], quality=r['Quality'], ref=r['First author(s) & year'], doi=r['DOI to cite']))
t=pd.DataFrame(rows)
def _clean(field):
    out=set()
    for h in field.split(', '):
        h=re.sub(r'_put\d*','',h)
        if h and h not in ('putative_primary','TBD','nan'): out.add(h)
    return out
def combine(r):
    # priority: MANC body ids > MANC type names > hemilineage written in the paper's label
    for k,src in (('hl_from_ids','manc_body_ids'),('hl_from_typename','manc_type_name'),('hl_from_label','paper_label')):
        s=_clean(r[k])
        if s: return pd.Series({'hemilineage':', '.join(sorted(s)),'label_source':src})
    return pd.Series({'hemilineage':'','label_source':''})
t[['hemilineage','label_source']]=t.apply(combine,axis=1)
has=t[(t.hemilineage!='') & ~t.superclass.str.contains('descending|sensory',regex=True)].copy()
has['single_hemilineage']=~has.hemilineage.str.contains(',')
def conflict(r):
    a=set(re.sub(r'_put\d*','',h) for h in r.hl_from_ids.split(', ') if h) - {'putative_primary','TBD'}
    b=set(r.hl_from_label.split(', '))-{''}
    return bool(a and b and not (a & b))
has['ids_vs_label_conflict']=has.apply(conflict,axis=1)
print('adult lines:',len(t),' with EM ids:',(t.em_ids!='').sum(),' ids found in MANC:',(t.em_ids_in_manc>0).sum())
print('lines with usable hemilineage:',len(has),' single-hemilineage:',has.single_hemilineage.sum(), ' id/label conflicts:', has.ids_vs_label_conflict.sum())
print(has[has.single_hemilineage].hemilineage.value_counts().to_string())
both=has[(has.hl_from_ids!='')&(has.hl_from_label!='')]
print('both evidence:',len(both)); print(both[['line','cell_types','hl_from_ids','hl_from_label']].head(12).to_string())
has.to_csv(os.path.join(HERE,'split_lines_hemilineage.csv'),index=False)
