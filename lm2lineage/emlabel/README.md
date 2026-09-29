# emlabel: VNC MCFO labels without hand segmentation

Two ways to get labelled VNC light-microscopy data from public resources. Both
put the hemilineage on every mask.

## Option 1: EM-guided mask proposals (`emlabel.build`)

NeuronBridge publishes matches between FlyLight Gen1 MCFO VNC samples and MANC
neurons (PatchPerPixMatch, EM → LM; colour depth search, both directions). It
also publishes the EM skeletons already aligned to the same template as the
MCFO stacks (JRC2018_VNC_Unisex_40x_DS). For each (EM neuron, MCFO sample) pair:

1. The EM skeleton is placed on the aligned stack. Both the stated orientation
   and the mirrored one are tried, and the better fit is kept.
2. The neuron's MCFO colour is read from bright voxels within 2.5 µm of the skeleton.
3. The mask is the voxels within 6 µm of the skeleton that are bright
   (z > 5 robust noise units) and close to that colour (cosine ≥ 0.85), kept as
   connected components that touch the skeleton.
4. Three scores are computed:
   - **coverage**: fraction of skeleton length with mask nearby;
   - **precision**: fraction of mask near the skeleton;
   - **presence**: fraction of skeleton length with any bright signal nearby.
   Each proposal is auto-flagged `likely_good`, `check` or `likely_bad` and gets a review PNG
   (raw | raw + EM skeleton | proposal).

```bash
# balanced over hemilineages: EM neurons -> their top PPPM MCFO samples
python -m emlabel.build by-hemilineage --hemilineages 01A,03A,06B --per-hemilineage 5 --top 2
# one of your own Gen1 MCFO VNC samples -> top colour-depth EM candidates
python -m emlabel.build by-sample --line R74B04 --slide 20190927_61_H3 --top 10
```

Review: open `review/*.png`, then fill `human_verdict` (accept / reject) in
`proposals.csv`. The hemilineage label comes from the male CNS
cross-matching (`mancBodyid → trumanHl`). Even a match to the wrong copy of the
right cell type (e.g. a serial homologue) usually carries the right hemilineage.

## Option 2: split-GAL4 lines of known identity (`emlabel.splitset`)

`split/label_lines.py` turns the split-GAL4 line resource table (Meissner et al.
2025, eLife 98405, Figure 1-source data 1; 3,060 adult lines) into a
hemilineage per line. Sources, in priority order:

1. the listed MANC body ids, via the male CNS table;
2. MANC type names;
3. the hemilineage written in the paper's own label.

Descending and sensory lines are dropped. That leaves 273 VNC lines with a
hemilineage, 231 of them with a single hemilineage, and every one has aligned
20× VNC stacks on FlyLight (2,267 samples). For each line, `emlabel.splitset`
takes one stack. Lines with MANC body ids get EM-guided masks, and the EM skeleton's
presence score checks the published identity against the image
(`identity_check`). Other lines get a thresholded foreground mask:
residual above a heavily blurred local background, > 12 robust noise units
and > 2.5× the local background, with small components removed.

```bash
python split/label_lines.py                      # rebuild the line table
python -m emlabel.splitset --per-hemilineage 0   # all 231 lines (~2 h)
```

## Demo results (in this zip)

- **Option 1** (`proposals_demo/`): 14 hemilineages × 1 MANC neuron × top 2
  PPPM samples gave 28 proposals, 17 flagged `likely_good` and 11 `check`.
  Median coverage was 0.86, precision 0.65 and presence 0.93. Shuffled
  pairs (the same stacks with an unrelated EM neuron, n = 56) had median
  coverage 0.44 and presence 0.70, and none was flagged `likely_good`.
- **Option 2** (`splitset_demo/`): up to 2 lines per hemilineage gave 53 samples
  covering 28 hemilineages. Of the 30 EM-guided samples, 26 had their identity
  confirmed (presence ≥ 0.7) and 4 were doubtful; 23 samples used threshold masks.
  With unrelated EM neurons, median presence was 0.13 (max 0.60) and none of
  30 reached 0.7.

## Caveats

- **Option 1 scores are not independent evidence.** PPPM chose these samples
  because they fit the EM neuron, so a high score shows the mask extraction
  worked, not that the match is independently confirmed. A shuffled-pair
  control helps calibrate the flags, but it is easier than real errors,
  which are usually a similar-looking neighbour.
- **Old body ids.** PPPM results exist only for the early VNC v0.5 bodies. Ids are
  mapped by number to MANC v1.2.1, and the presence score catches ids that
  changed meaning.
- **Label naming differs between papers.** Ehrhardt et al. write "16A" where
  MANC says 16B, and "8B" where MANC says 07B. MANC-derived labels win, and
  conflicts are recorded in `split_lines_hemilineage.csv`.
- **Split-line masks are incomplete or impure.** EM-guided masks cover only the
  listed bodies, not every copy labelled in the image. Thresholded masks include any
  off-target expression. Both are fine for image-level evaluation ("this
  image contains hemilineage X"), not for voxel-perfect instance ground truth.
- The 20× split images are full-expression patterns rather than MCFO; they test
  hemilineage recognition, not MCFO colour separation.
- A further known-identity resource: Soffers et al. 2025 (eLife 106042) made
  hemilineage-specific split lines for 32 of 34 VNC hemilineages. The images
  don't appear to be deposited, so ask the authors.
