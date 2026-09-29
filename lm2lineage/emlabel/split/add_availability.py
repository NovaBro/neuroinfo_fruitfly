"""Add NeuronBridge aligned-VNC-stack availability to the split-line table.
Run from the emlabel/ directory:  python split/add_availability.py"""
import os
import pandas as pd
from emlabel.nb import NB

HERE = os.path.dirname(os.path.abspath(__file__))
nb = NB("cache")
t = pd.read_csv(os.path.join(HERE, "split_lines_hemilineage.csv"))
t = t[t.single_hemilineage]
rows = []
for line in t.line:
    d = nb.meta(f"by_line/{line}.json")
    recs = [] if not d else [r for r in d["results"]
                             if r["anatomicalArea"] == "VNC" and r["files"].get("VisuallyLosslessStack")]
    slides = sorted({(r["slideCode"], r["objective"]) for r in recs})
    rows.append(dict(line=line, n_vnc_stacks=len(slides), objectives=",".join(sorted({s[1] for s in slides}))))
a = pd.DataFrame(rows)
t.merge(a, on="line").to_csv(os.path.join(HERE, "split_lines_hemilineage_avail.csv"), index=False)
print((a.n_vnc_stacks > 0).sum(), "of", len(a), "lines have aligned VNC stacks")
