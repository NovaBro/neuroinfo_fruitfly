# CLAUDE.md — lm2lineage

Start-up instructions for Claude working locally on this project. Read the whole
file before acting. It records the goal, what exists, what has been verified,
known pitfalls, and the next steps.

---

## 1. Project goal

Build a pipeline that takes **FlyLight MCFO confocal images of the adult
*Drosophila* ventral nerve cord (VNC)**, extracts the individual neurons, and assigns
each neuron its **hemilineage** (e.g. 03A, 16B, 19A).

The owner is Vince (NYU, neuroinfolab):

- He has a local collection of FlyLight MCFO VNC images.
- He is **not a domain expert in hand segmentation**. Do not plan on him hand-labelling neurons. Proofreading accept/reject decisions is fine.
- He wants reproducible methods. Heavy rendering and training run on his own compute, not in a chat sandbox.

---

## 2. Scientific basis (why hemilineage is predictable from morphology)

- **Hemilineage fixes the broad morphology.** Each VNC neuroblast divides into Notch-ON (A) and Notch-OFF (B) hemilineages. Neurons of a hemilineage share soma position, the tract where the primary neurite enters the neuropil, and the gross neuropil targeting (the "ground plan"). Birth order (temporal identity) then refines morphology within a hemilineage.
- **Diagnostic features in light microscopy:**
  - soma location;
  - primary-neurite entry point and tract;
  - gross arbor.

  The MANC annotators assigned hemilineages mainly from soma-tract entry and gross morphology.
- **Neurons with no VNC hemilineage.** Descending neurons (soma in the brain), sensory afferents, and many embryonic "primary" neurons have none. Any classifier needs an `unassigned` class.
- **Key references** (all verified):
  - Harris, Pfeiffer, Rubin & Truman 2015 eLife 4:e04493. 33 thoracic hemilineages; thermogenetic *activation* with dTrpA1 in decapitated flies. It is **not** an ablation study, and the authors are **not** "Zwart et al." Both errors appeared in an early answer.
  - Shepherd et al. 2016 J Comp Neurol (doi 10.1002/cne.23988). Neuroglian-labelled adult hemilineage tracts.
  - Marin et al. 2024, MANC systematic annotation (eLife reviewed preprint 97766). 97 % of thoracic neurons carry a hemilineage; 35 secondary hemilineages; 38 % of abdominal neurons annotated.
  - Clowney lab 2026 Nature (doi 10.1038/s41586-026-10526-3). Knocking out hemilineage transcription factors (fd59a, dsf, TfAP-2) disrupts projections: causal evidence in the cerebrum.
  - Soffers et al. 2025 eLife 106042. Split-GAL4 lines targeting 32 of 34 VNC hemilineages. The images do not appear to be deposited; ask the authors.

---

## 3. Data resources (all public and anonymous unless noted)

| Resource | Where | Notes |
|---|---|---|
| Male CNS connectome v1.0 (JRCFIB2022M, brain + VNC, 166k neurons) | `https://storage.googleapis.com/flyem-male-cns/v1.0/` | Annotation table: `connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather` (14 MB).<br>Useful columns: `bodyId`, `type`, `superclass`, `trumanHl` (VNC hemilineage), `itoleeHl` (brain), `mancBodyid`, `mancType`, `somaLocation` (8 nm voxels), `somaNeuromere`. |
| Male CNS skeletons | `.../segmentation/skeletons-malecns/skeletons-swc/<id>.swc` | Raw EM space; xyz and radius in 8 nm units. |
| Male CNS skeletons in template space | `.../segmentation/skeletons-unisex-template/skeletons-swc/<id>.swc` | µm, JRC2018 **unisex brain**. Valid for brain neurons. **VNC coordinates fall outside the JRCVNC2018U bounding box**; don't trust them for the VNC (see §7). |
| MANC → hemilineage | the male CNS table: `mancBodyid → trumanHl` | This is how MANC body ids get hemilineage labels in all code here. |
| NeuronBridge metadata | `https://janelia-neuronbridge-data-prod.s3.amazonaws.com/<ver>/metadata/` | `current.txt` gives the version (v3_10_0 when this was written).<br>`by_line/<line>.json`: LM images and stack paths.<br>`by_body/<bodyId>.json`: EM images and aligned SWC.<br>`cdsresults/<id>.json`: colour-depth search, LM↔EM.<br>`pppmresults/<id>.json`: PatchPerPixMatch, EM → LM; VNC only for `FlyEM_VNC_v0.5` bodies. |
| Aligned EM skeletons (VNC) | `https://janelia-flylight-color-depth.s3.amazonaws.com/JRC2018_VNC_Unisex_40x_DS/FlyEM_MANC_v1.2.1/SWC/<id>.swc` | µm, same space as the aligned MCFO stacks. |
| FlyLight imagery | `https://janelia-flylight-imagery.s3.amazonaws.com/Gen1 MCFO/<line>/...` | Also split collections: `Dorsal VNC 2023`, `Ascending Neurons 2023`, `Descending Neurons 2018/2025`, `Cheong, Eichler, Stuerner 2023`.<br>Aligned stacks: `...-JRC2018_VNC_Unisex-aligned_stack.h5j`. Raw tiles: `.lsm.bz2`. |
| PPPM overlay PNGs | `https://janelia-ppp-match-prod.s3.amazonaws.com/...` | Paths come from the `pppmresults` JSON. |
| FISBe (brain MCFO instance ground truth) | Zenodo doi:10.5281/zenodo.10875063 | 30 completely and 71 partly labelled images, ~600 neurons, **brain only**. See the format notes below. Evaluation code: `github.com/Kainmueller-Lab/evaluate-instance-segmentation`. |
| Split-GAL4 line → cell type table | eLife 98405 Figure 1-source data 1 (`https://cdn.elifesciences.org/articles/98405/elife-98405-fig1-data1-v2.xlsx`) | 3,060 adult lines, with `Cell types` and `EM Body IDs` columns. |
| MN line matching | eLife 96084 supplementary file 3 (csv) | MANC motor neuron types → split lines. |

**FISBe format:**

- **Layout:** `<sample>.zarr/volumes/raw` holds uint16 (3, Z, Y, X), 0.44 µm isotropic, chunks (3, 64, 64, 64), blosc-zstd clevel 3 with bitshuffle. `volumes/gt_instances` holds uint8 (N, Z, Y, X), one channel per neuron, with `.zattrs {"dim_neurons": [...]}`.
- **Images:** native stitched 40× tiles, not template-registered, averaging ~385×685×686 voxels (~231 MB compressed).
- **Labels:** the completely labelled set has about 4.6 labelled neurons per image; the partly labelled set leaves unlabelled "overlapping neurons with the same or similar colour".

**Reading NeuronBridge stacks (`.h5j`):**

- **Container:** an HDF5 file holding `Channels/Channel_k`, one HEVC (H.265) stream per channel, pix_fmt `gray12le`.
- **Attributes:** the attrs give `width`, `height` and `frames`; frames are padded, so crop them. Each channel has a `content_type` of `signal` or `reference`, and the stack attrs give `voxel_size`.
- **Voxels:** JRC2018_VNC_Unisex_40x_DS stacks are 573×1119×219 at **0.461122 × 0.461122 × 0.7 µm**.
- **Decoding:** use `h5py` + `av` (PyAV); see `emlabel/emlabel/nb.py:read_h5j`.
- **Mirroring an EM skeleton** onto the other side of the template: `x' = 573·0.461122 − x`.

**Real Gen1 40× VNC detector statistics** (measured on the R10A07 prothoracic tile):

- 12-bit data;
- dark level ~2–3 ADU, dark noise s.d. ~1.3 ADU;
- signal-channel foreground median ~46 ADU, foreground p99.9 ~1,000 ADU;
- neuropil background barely above the dark level.

---

## 4. Code in this repository

Unzip `lm2lineage_code.zip` next to this file. The layout:

```
lm2lineage/
  CLAUDE.md                     <- this file
  mcfo_sim/                     synthetic MCFO simulator (VNC + brain)
    requirements.txt
    README.md                   VNC simulator notes
    README_brain.md             brain / FISBe method (full protocol)
    mcfo_sim/
      data.py geometry.py labeling.py render.py imaging.py
      scene.py                  shared renderer -> FISBe zarr layout
      simulate.py               VNC front end (male CNS VNC, EM space)
      verify.py
      brain/                    brain front end (male CNS brain, JRC2018U)
        config.py pool.py manifest.py render.py stats.py validate.py cli.py
  emlabel/                      labelled VNC data without hand segmentation
    README.md
    emlabel/ nb.py propose.py qc.py build.py splitset.py
    split/label_lines.py add_availability.py split_lines_hemilineage*.csv
    control.py control_split.py  (negative controls)
    proposals_demo/ splitset_demo/ bysample_demo/   (result tables from demos)
```

**Environment.** Python 3.11. Install `pip install -r mcfo_sim/requirements.txt h5py av openpyxl`,
then set `export PYTHONPATH=$PWD/mcfo_sim:$PWD/emlabel`.

### 4.1 `mcfo_sim`: synthetic MCFO images

Physics and labelling model, shared by both front ends:

- **MCFO-1 labelling.** Three tags (HA, V5, FLAG), each excised independently with probability q, conditioned on at least one tag being on.
- **Expression levels.** Lognormal per cell type, per cell and per tag.
- **Membrane label.** Myristoylated smFP, so signal ∝ membrane area. Thin neurites are splatted at 2πr per µm; thick processes and somata are hollow shells, so somata look rim-bright.
- **Boutons.** Random membrane beads on thin neurites.
- **Brightness along the cable.** Smooth variation plus dim gaps.
- **Optics.** Gaussian PSF that widens with depth, depth attenuation, illumination field and channel crosstalk.
- **Tissue.** Haze with fine texture, speckle, and soma-sized non-specific blobs.
- **Detector.** Poisson, gamma (PMT excess) and read noise, 12-bit, with a per-channel auto-gain. Calibrated to the real tile above.
- **Ground truth.** Filled tubes gated to voxels where the neuron's own SNR ≥ 2.
- **Randomisation.** Every imaging parameter is resampled per sample.

The two front ends:

- **VNC** (`python -m mcfo_sim.simulate --n 10 --neuromeres T2`):
  - uses male CNS VNC neurons in raw EM space, reoriented to image axes (X = left–right, Y = anterior–posterior, Z = depth);
  - scales by 0.70, the EM → DPX-cleared-LM shrinkage fitted from the release's own template transform;
  - builds a "virtual GAL4 line" from random cell types, then applies MCFO.
- **Brain / FISBe** (`python -m mcfo_sim.brain.cli pool|manifest|stats|render|validate`): see `mcfo_sim/README_brain.md`.
  - Uses the male CNS brain (~50k neurons, 8.3k types) in JRC2018U µm.
  - `manifest` designs N images without making pixels: 6 count bins from 1 to 60 neurons; overlap modes isolated / natural / crowded; forced same-colour overlaps; splits by cell type.
  - It logs body ids and count and overlap statistics.
  - Output is in FISBe layout (`volumes/raw`, `volumes/gt_instances` + `dim_neurons`, plus `gt_labels`).
  - The manifest and rendering are deterministic and were tested bit-identical.
  - Estimated cost at 0.44 µm: ~3–5 min, ~10–12 GB RAM and ~250 MB per image, so 5,000 images take ~250–400 CPU-hours and ~1.3 TB. Use `--shard k/n`.

### 4.2 `emlabel`: labelled VNC data

- **Option 1, `python -m emlabel.build by-hemilineage|by-sample`.** EM-guided mask proposals.
  - It pulls NeuronBridge matches (PPPM EM → LM, colour-depth search LM → EM), overlays the aligned MANC skeleton on the aligned MCFO stack, infers the neuron's colour, and grows a 3D mask within 6 µm of the skeleton.
  - Scores: coverage, precision, presence. Auto-flag: `likely_good` needs coverage ≥ 0.6, precision ≥ 0.6 and presence ≥ 0.7.
  - Each proposal gets a review PNG, and `proposals.csv` has a `human_verdict` column.
  - `by-sample --line <L> --slide <S>` runs the same thing on one of Vince's own Gen1 MCFO samples.
- **Option 2, `python -m emlabel.splitset`.** Known-identity split-GAL4 lines.
  - `split/label_lines.py` maps lines to a hemilineage, preferring MANC body ids, then MANC type names, then the paper's own label. That gives 231 single-hemilineage VNC lines, all with aligned 20× stacks (2,267 samples).
  - `split/add_availability.py` (run from `emlabel/`) adds the NeuronBridge stack availability that `splitset` reads. Both CSVs are already included.
  - Lines with MANC ids get EM-guided masks plus an identity check (presence ≥ 0.7); the rest get threshold masks.

---

## 5. Verified results so far (small demos; treat as evidence, not final numbers)

- **Option 1** (28 proposals across 14 hemilineages): 17 `likely_good`, 11 `check`, none `likely_bad`. Median coverage 0.86, precision 0.65, presence 0.93.
  - **Shuffled control** (the same stacks with unrelated EM neurons, n = 56): median coverage 0.44, and none flagged `likely_good`.
- **LM-first mode** on one test sample (R74B04, slide 20190927_61_H3): four of the six candidates tried were IN16B042/IN16B118 copies (16B), all flagged `likely_good`. The other two (03A, 01B) got `check`.
- **Option 2** (53 samples, 28 hemilineages): 26 of 30 EM-guided samples had their identity confirmed.
  - **Shuffled control:** median presence 0.13, max 0.60; none of 30 reached 0.7.
  - Trying up to 3 slides per line fixed cases where the first slide lacked the cell type.
  - Threshold masks needed a strict setting (z = 12, ≥ 2.5× local background). Looser settings captured tissue haze.
- **VNC simulator:** noise statistics match the real tile.
- **Brain method:** 5,000-image manifest designed on a 3k-neuron test pool (neurons per image 1–60, median 10; 0 empty images; 0 split leakage). Four images rendered, and all passed integrity checks.
- **Not yet done:**
  - any comparison against real FISBe (the download was too large for the sandbox);
  - any segmentation network training;
  - hemilineage classification on real data.

---

## 6. Recommended plan (next steps, in order)

1. **Set up locally.** Unzip, install, and rerun the smoke tests (§8) to confirm the environment reproduces.
2. **Get FISBe.**
   - Download `fisbe_v1.0_completely.zip` and `fisbe_v1.0_partly.zip`.
   - Run `validate calibrate` on **train only**, then `compare` and `c2st` on a few synthetic renders.
   - Tune until the image-level KS < 0.3 and the c2st AUC < 0.75.
   - **Check the MCFO colour multiplicity (q).** In the test plan, about 45 % of neurons carry more than one colour tag. Real MCFO may be more often single-colour, so calibrate q from FISBe.
3. **Brain synthetic set.** Build the full pool, then a calibrated 5,000-image manifest, then render sharded on local or cluster compute.
4. **Segmentation.**
   - Train PatchPerPix (the best FISBe baseline) under four regimes: FISBe-only (A), synthetic-only (B), synthetic pre-train then FISBe fine-tune (C), mixed (D).
   - Evaluate on FISBe test with the official code and report S, avF1, C and FS/FM, overall and for dim neurons. The protocol is in `README_brain.md` §4.5.
   - Then add VNC synthetic data (the `simulate` front end) and fine-tune for the VNC.
5. **VNC labelled data.**
   - Scale Option 1 across all hemilineages (`--per-hemilineage 5 --top 3`).
   - Have Vince accept or reject proposals. A phone-friendly review page is a possible add-on.
   - Scale Option 2 to all 231 lines.
6. **Hemilineage classifier.** Run two routes and compare them:
   - **Route A, match to connectome.** Transform the segmented neuron and NBLAST it against MANC or male CNS neurons in the same space, then take a k-NN vote on `trumanHl`.
   - **Route B, direct features.** Train a classifier on soma position, primary-neurite entry point and the soma-tract segment, using MANC neurons as training data (augmented to look LM-like) plus the Option 1 and 2 labels.

   Include an `unassigned` class. Evaluate on the Option 2 identity-confirmed images and on held-out Option 1 accepts.
7. **End-to-end on Vince's images.** Segment, then classify, then review a sample by eye.

---

## 7. Pitfalls and gotchas (learned the hard way)

- **Template-space VNC skeletons are unreliable.** In the male CNS bucket's `skeletons-unisex-template`, **VNC neurons are not in JRCVNC2018U** (coordinates such as z ≈ 294–536 µm fall outside that template's bounds). For the VNC in LM space, use NeuronBridge's `JRC2018_VNC_Unisex_40x_DS/FlyEM_MANC_v1.2.1/SWC/`. Alternatively, chain the flybrains transforms maleCNS → BANC → JRCVNC2018F.
- **Most imaging here is DPX-cleared** (JRC2018 templates, FlyLight), so light-microscopy tissue is ~0.63–0.79× the size of the EM tissue.
- **PPPM for the VNC uses old ids.** Results exist only for `FlyEM_VNC_v0.5` bodies. Ids are reused by number for MANC v1.2.1; the presence score catches ids whose meaning changed.
- **PPPM self-selection.** PPPM chose those MCFO samples *because* they fit the EM neuron, so high proposal scores validate mask extraction, not identity. Real errors are usually a similar-looking neighbour, which is harder than the shuffled control.
- **Hemilineage names differ between papers.** Ehrhardt et al. write "16A" where MANC says 16B, and "8B" where MANC says 07B. Prefer MANC-derived labels.
- **Male CNS table has duplicate `mancBodyid` rows.** De-duplicate before indexing (`drop_duplicates("mancBodyid")`).
- **Memory.** A full aligned VNC stack is ~1.1 GB as uint16, so load only the signal channels. `emlabel.propose` bins x/y for crops larger than 30 M voxels and computes distances on a coarse grid; an earlier version was OOM-killed at ~6 GB.
- **Split-line masks.** 20× split images are full-expression patterns, not MCFO. EM-guided masks cover only the listed bodies; threshold masks include off-target expression. Use them for image-level labels, not voxel-perfect instance ground truth.
- **Shell safety.** Don't `pkill -f <pattern>` when the pattern also appears in your own shell command; that kills your own shell. Kill by PID.
- **FISBe metrics.** clDice-based avF1 over thresholds 0.1–0.9, coverage C, S = ½(avF1 + C). Matching is greedy with pixel tracking.

---

## 8. Smoke tests (run first; each takes minutes)

```bash
export PYTHONPATH=$PWD/mcfo_sim:$PWD/emlabel

# VNC simulator (coarse voxel for speed) + consistency check
python -m mcfo_sim.simulate --out /tmp/vnc --cache data --n 1 --seed 11 --voxel 0.88
python -m mcfo_sim.verify /tmp/vnc/synth_00011.zarr          # expect colour_consistent n/n

# Brain method on a small pool
python -m mcfo_sim.brain.cli pool --cache cache --out pool_small --max-neurons 3000
python -m mcfo_sim.brain.cli manifest --pool pool_small --out man_small --n-images 12
python -m mcfo_sim.brain.cli render --manifest man_small --pool pool_small --cache cache --out r_small --first 3
python -m mcfo_sim.brain.cli validate integrity --rendered r_small --manifest man_small   # expect passed 3/3

# EM-guided proposal on the known-good pair (MANC 16539 / R74B04 20190927_61_H3)
cd emlabel && python -m emlabel.build by-sample --line R74B04 --slide 20190927_61_H3 --top 3 --out /tmp/bs
# expect IN16B042 / IN16B118 (16B) rows flagged likely_good
```

---

## 9. Working conventions for Claude on this project

- **Verify before citing.** Fetch or read a source before summarising it, and never guess authors, methods or numbers. An earlier answer misattributed Harris et al. 2015; don't repeat that.
- **Run a negative control** for every automatic labelling or matching score (shuffled pairs, unrelated EM neurons) before trusting its thresholds.
- **Prefer deterministic, logged pipelines.** Seed everything, write the config next to the outputs, and log body ids, types and hemilineages for every sample.
- **Look at the images.** Inspect review PNGs and MIPs after each change; the statistics alone have hidden failures before (e.g. threshold masks that captured tissue haze).
- **Axes and units.** Arrays are stored Z, Y, X (and C, Z, Y, X); coordinates are µm unless the file says 8 nm EM voxels. JRC2018 VNC stacks: 0.461122 × 0.461122 × 0.7 µm. FISBe and 40× Gen1: 0.44 µm isotropic.
- **Keep the non-expert workflow.** Vince decides accept/reject. Don't design steps that need expert tracing.
- **Say what's unverified.** In summaries, state what has and hasn't been validated.
