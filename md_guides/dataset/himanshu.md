# Himanshu dataset (split-line MCFO, no GT)

Last updated 2026-11-17.

A test-only set for PatchPerPix inference: MCFO images of 20 Janelia split-GAL4 lines,
taken from each line's **own** MCFO samples on the public FlyLight S3 bucket
(`s3://janelia-flylight-imagery`, anonymous). There is no ground truth, so the set has no
`gt_instances` and no evaluate step. How to run inference is in the generic runbook
[`md_guides/ppp/ppp_new_dataset_inference.md`](../ppp/ppp_new_dataset_inference.md); this page
records what the dataset is and how it was built.

---

## 1. Datasets on disk

| Name | Path | Source images | Voxel size (x × y × z µm) | Lines | Used by |
|---|---|---|---|---|---|
| **`himanshu_20x`** (current) | `fisbe/himanshu_20x/test` (20 zarrs, 7.0 GB) | 20x unaligned h5j, native grid | 0.52 × 0.52 × 1.0 (SS47398: 0.59 × 0.59 × 1.0) | current list (§3) | inference not run yet |
| `himanshu` | `fisbe/himanshu/test` (46 GB) | 63x unaligned h5j, native grid | 0.19 × 0.19 × 0.38 | **old** list: f_brain = SS00043 SS00078 SS00090 SS00098 SS00191 | source of `himanshu_r044` |
| `himanshu_r044` | `fisbe/himanshu_r044/test` (7.2 GB) | `himanshu` resampled (`resample_himanshu_zarr.py`) | 0.44 isotropic (FISBe) | old list | `metrics/ppp/himanshu-e1.3-rm1500` (EXP 1.3 @ 23000, rm1500-filtered) |

Each zarr has a sidecar `manifest.csv` in its `test/` dir. The download manifests are in
`downloads/`: `mcfo_manifest_20x.tsv` (current) and `mcfo_manifest.tsv` (the 63x set, old list).
All three names are in `KNOWN_DATASETS` (`PatchPerPix/experiments/run_ppp.py`) and have a data
overlay `configs/data/datasets_test_<name>.toml`. The web viewer knows `himanshu` and
`himanshu_r044` only.

The 15 lines outside the f_brain cell are the same flies (same slide code) in all three datasets.
Rebuilding the 63x set now (`OBJECTIVE=63x NAME=himanshu`) would use the current list, so its
f_brain cell would change too.

---

## 2. Zarr contract

| Item | Value |
|---|---|
| Path | `fisbe/<name>/test/<stem>.zarr`, `<stem>` = h5j name minus `_stack.h5j` (e.g. `SS02195-20160603_20_E3-f-20x-brain-Split_GAL4-unaligned`) |
| `volumes/raw` | uint16, CZYX, exactly 3 channels (the 3 MCFO signal channels; the reference channel is dropped), chunks (1, 64, 64, 64), Blosc zstd |
| `volumes/raw_normalized` | float16, `clip(raw, 0, 1500) / 1500` (lossless for the 1500 levels) |
| Root attrs | `voxel_size_um` [vx, vy, vz] (from the h5j `voxel_size` attr), `publishing_name`, `image_id`, `release`, `slide_code`, `objective`, `area`, `stack`, `source_s3_key`, `voxel_size_source`, `alignment_space` |

There is no resampling: each volume keeps the shape of its unaligned stack
(20x: 1024 × 1024 × 112–153 slices).

---

## 3. Line selection

**Source list:** `data_download_methods/elife-98405-fig1-data1-v2.xlsx - Cell type lines.csv`
(eLife 98405, Figure 1 data 1: 4433 rows, 3060 Adult). "CSV row" below is the 0-based index among
the Adult rows.

**Design:** a 2 × 2 of fly sex × imaged area (f/m × brain/vnc), 5 lines per cell, one image per line.
A cell is about the **image** (sex of the fly, brain or VNC stack), not the line's expression
pattern.

**Rules:**
1. Adult lines in CSV order. The 2026-11-16 picks were recorded as "CSV order, rarest cell first".
   That date's 63x picks for f_vnc / m_brain / m_vnc are kept unchanged.
2. MCFO only: effector `hsPESTOPT_attP3_3stop1_X_0036`, merged sample records (not 63x LSM tiles).
3. One fly per line, pinned by `slide_code`. The fly must have a **3-colour (`sssr`)** MCFO image
   at **both 20x and 63x** for the cell's sex and area. So 20x and 63x are the same specimen.
4. If a line has several such flies, take the first by (release, slide_code, image id).

**f_brain swap (2026-11-17):** the original f_brain lines SS00043, SS00078, SS00090, SS00098 and
SS00191 (rows 0–4) were dropped. Their 20x images are 2013 screen scans with **1 signal channel +
reference** (h5j `channel_spec = sr`; blank `channel_spec` in the S3 metadata). Four of them have no
3-colour 20x image at all; SS00090 has some, but only from other flies. The replacements are the
next 5 Adult lines in CSV order that satisfy rule 3:
- **Skipped:** rows 5, 7, 8, 9 (only 63x is 3-colour), rows 12–14 and 16–20 (no merged MCFO
  sample), row 21 (no same-fly 63x).
- **Not chosen:** the next qualifying line, SS47405 (row 23, slide `20200918_20_D2`, 0.52 µm). It
  is the swap to make if a uniform 0.52 µm XY is wanted instead of SS47398's 0.59 µm.

**Note on m_brain:** all five m_brain lines are annotated "VNC only" in the CSV. Their brain stacks
may hold few labelled neurons.

| Cell | Line | CSV row | Expression (CSV) | Cell types (CSV) | slide_code | 20x image id | Release | Voxel (µm) | Zarr shape (C,Z,Y,X) |
|---|---|---|---|---|---|---|---|---|---|
| f_brain | SS02195 | 6 | both | PEcG | 20160603_20_E3 | 8684845 | Wolff et al 2024 | 0.52×0.52×1 | 3,115,1024,1024 |
| f_brain | SS02718 | 10 | brain only | vDeltaA_a, vDeltaA_b | 20151223_23_D3 | 8686697 | Wolff et al 2024 | 0.52×0.52×1 | 3,137,1023,1023 |
| f_brain | SS04778 | 11 | both | IbSpsP | 20160120_21_G1 | 8687640 | Split-GAL4 Omnibus Broad | 0.52×0.52×1 | 3,121,1024,1024 |
| f_brain | SS27853 | 15 | both | PEG | 20160715_21_C2 | 8684731 | Split-GAL4 Omnibus Broad | 0.52×0.52×1 | 3,128,1024,1024 |
| f_brain | SS47398 | 22 | both | LNO1 | 20200918_20_A5 | 8699011 | Split-GAL4 Omnibus Broad | **0.59×0.59×1** | 3,123,1024,1024 |
| f_vnc | SS31997 | 282 | VNC only | Power muscle motor neuron population | 20170217_20_A1 | 8547283 | Dorsal VNC 2023 | 0.52×0.52×1 | 3,112,1024,1024 |
| f_vnc | SS34781 | 289 | VNC only | i2 MN | 20170331_23_A1 | 8545263 | Dorsal VNC 2023 | 0.52×0.52×1 | 3,123,1024,1024 |
| f_vnc | SS42475 | 328 | VNC only | XBI002 0A, C2BI005 7B | 20190426_27_B2 | 8544976 | Dorsal VNC 2023 | 0.52×0.52×1 | 3,135,1024,1024 |
| f_vnc | SS49779 | 391 | VNC only | WBL022 3B, WUI005 3B, WBI015 3B, WBI016 03B | 20191018_20_H4 | 8544617 | Dorsal VNC 2023 | 0.52×0.52×1 | 3,133,1023,1024 |
| f_vnc | SS53435 | 414 | VNC only | bilateral local interneuron, leg bilateral interneuron | 20190329_20_E1 | 8543354 | Dorsal VNC 2023 | 0.52×0.52×1 | 3,132,1024,1024 |
| m_brain | SS51895 | 188 | VNC only | LUI22, LUL174 | 20190517_22_E2 | 8603196 | Split-GAL4 Omnibus Broad | 0.52×0.52×1 | 3,145,1024,1024 |
| m_brain | SS57939 | 194 | VNC only | LUI22 | 20190524_24_A4 | 8607749 | Split-GAL4 Omnibus Broad | 0.52×0.52×1 | 3,131,1024,1024 |
| m_brain | SS31259 | 273 | VNC only | WBL030 3B, WBI018 11B, WBI019 11B | 20170616_24_C2 | 8546128 | Dorsal VNC 2023 | 0.52×0.52×1 | 3,112,1024,1024 |
| m_brain | SS37231 | 293 | VNC only | hi1 MN, hi2 MN | 20170420_20_A3 | 8546027 | Dorsal VNC 2023 | 0.52×0.52×1 | 3,125,1024,1024 |
| m_brain | SS37233 | 294 | VNC only | WBL034 19B, WBL035 19B, WBL038 19B | 20170420_19_A6 | 8545967 | Dorsal VNC 2023 | 0.52×0.52×1 | 3,135,1024,1024 |
| m_vnc | SS33343 | 155 | VNC only | LUL92, LUL93, LUL97, LUL100 | 20190215_24_D1 | 8597465 | Split-GAL4 Omnibus Broad | 0.52×0.52×1 | 3,143,1024,1024 |
| m_vnc | SS44255 | 164 | VNC only | LUL50, LUL145 | 20190419_22_D1 | 8605012 | Split-GAL4 Omnibus Broad | 0.52×0.52×1 | 3,152,1024,1024 |
| m_vnc | SS46225 | 165 | VNC only | LBI64 | 20190404_25_A1 | 8606652 | Split-GAL4 Omnibus Broad | 0.52×0.52×1 | 3,125,1024,1024 |
| m_vnc | SS46730 | 173 | VNC only | LUI12, LUL24 | 20180914_19_B3 | 8591606 | Split-GAL4 Omnibus Broad | 0.52×0.52×1 | 3,134,1024,1024 |
| m_vnc | SS47847 | 174 | VNC only | LUL89 | 20190405_23_E1 | 8611784 | Split-GAL4 Omnibus Broad | 0.52×0.52×1 | 3,153,1024,1024 |

The 20 h5j stacks total 486.6 MB.

---

## 4. Build

The list lives in [`data_download_methods/himanshu_lines.tsv`](../../data_download_methods/himanshu_lines.tsv).
Its columns are `ss sex area slide_code`; a blank column means any.
One CPU job ([`sbatch/ppp/prep/himanshu/himanshu_sbatch.sh`](../../sbatch/ppp/prep/himanshu/himanshu_sbatch.sh),
`cs`, 4 CPU, 64 GB) builds the set in two steps:

1. [`download_mcfo.py`](../../data_download_methods/download_mcfo.py) runs on the host
   (python3 + aws CLI). It lists the 54 releases, syncs `*metadata.json` of every release folder
   holding a line, and selects per rules 2–4. It writes `downloads/mcfo_manifest_<objective>.tsv`
   and downloads the stacks to `downloads/<release>/<SS>/`, keeping files already on disk with the
   same size.
2. [`prepare_himanshu_for_ppp.py`](../../data_download_methods/prepare_himanshu_for_ppp.py) runs in
   the `ppp` container and needs PyAV (`sbatch/ppp/env/ppp_env_install_av_sbatch.sh`, once). It
   converts every `selected` row to a zarr (§2) and writes `manifest.csv`.

```bash
DRY_RUN=1 sbatch sbatch/ppp/prep/himanshu/himanshu_sbatch.sh   # manifest only: picks + sizes
LIMIT=1   sbatch sbatch/ppp/prep/himanshu/himanshu_sbatch.sh   # smoke: 1 volume
CLEAN=1   sbatch sbatch/ppp/prep/himanshu/himanshu_sbatch.sh   # wipe fisbe/himanshu_20x/test, rebuild all
OBJECTIVE=63x NAME=himanshu CLEAN=1 sbatch ...                 # 63x set at native 0.19 µm
```

Other variables: `STACK` (`unaligned` | `gendered` | `unisex`), `STAGE` (`all` | `download` | `prep`),
`OVERWRITE`, `RAW_DTYPE`, `CLIPMAX`. `download_mcfo.py --min-signal-channels 3` (default) rejects
1-colour images. Variable table: [`md_guides/ppp/ppp_prep.md`](../ppp/ppp_prep.md).

**Runtime (2026-11-17, job 23536277, 23 min):** about 7 min for the S3 metadata scan + download,
then about 45–60 s per volume to convert, one at a time.

**Checks done at the 2026-11-17 rewrite** (replaced `s3_download_fast.sh` and its sbatch jobs):
- `download_mcfo.py --objective 63x` reproduced the old 63x manifest exactly (20/20 rows, all
  columns), both with and without slide codes pinned.
- The new converter gave bit-identical `raw` / `raw_normalized` to the old one on a 63x volume
  (SS00191).

---

## 5. Inference settings

- **Command:** [`ppp_new_dataset_inference.md`](../ppp/ppp_new_dataset_inference.md) ("Preferred: split
  pipeline without evaluate"): `--datasets himanshu_20x --config datasets_test_himanshu_20x.toml
  --stage all --no-eval`, in a new exp dir (e.g. `himanshu20x-e1.3` with EXP 1.3 @ 23000 copied in).
- **No rm1500 (decision 2026-11-17):** Himanshu inference keeps the raw `vote_instances`. Do not
  run `filter_small_instances.py` and do not use `eval_rm*.toml`. The only size filter is PPP's
  label-stage `[vote_instances] ignore_small_comps = 200` (`original_default.toml`, all datasets).
  It drops foreground components under 200 voxels from the mask before patch voting.
- **Scale:** checkpoints are trained on FISBe at 0.44 µm isotropic. Native 20x is about 1.2×
  coarser in XY and about 2.3× coarser in Z, and SS47398 is 0.59 µm in XY. Volumes are fed at the
  native grid; resampling (`resample_himanshu_sbatch.sh` with `SRC`/`DST`) is optional.
- **Size:** about 2.7 Gvox per channel over 20 volumes (2614 slices of 1024²), similar to `himanshu_r044`.
- **Cleanup:** strip `pred_affs` by hand afterwards (`strip_pred_affs.py --arrays volumes/pred_affs`).
  Auto-strip only runs after evaluate.

---

## 6. FlyLight S3 metadata notes

- A line sits in 3–4 releases (e.g. Split-GAL4 Omnibus Broad, Wolff et al 2024, Dorsal VNC 2023);
  scan all of them.
- Some Omnibus Broad folders hold metadata only, with no stacks (e.g. SS58626).
- Releases with `<SS>.metadata.json` list all images. Others have only per-image
  `*-metadata.json`, which lacks `imaging_project`, `id` and `voxel_size`. There, MCFO is detected
  from the effector and the id is parsed from the file name.
- A 63x MCFO sample is one merged record (has the h5j stack keys) plus 2 tile records (only
  `-imageN.lsm.bz2`).
- `channel_spec`: `sssr` = 3 MCFO colours + reference. A blank value on the 2013 20x screen images
  means `sr` in the h5j (1 signal channel).
- 20x voxel sizes: 0.46 × 0.46 × 1.0 (2013), 0.52 × 0.52 × 1.0 (most), 0.59 × 0.59 × 1.0 (some 2020).
  63x: 0.19 × 0.19 × 0.38.
- `unisex` aligned stacks are `JRC2018_Unisex_20x_HR` (a template space, not a fly sex).

---

## 7. History

| Date | Change |
|---|---|
| 2026-11-16 | Switched from parent Gen1 MCFO lines to the split lines' own MCFO. 2 × 2 design, 63x, 20 lines (`filelist.txt`, `s3_download_fast.sh --mode mcfo`). |
| 2026-11-17 | 63x resampled to 0.44 µm (`himanshu_r044`). Inference with EXP 1.3 @ 23000 + rm1500 filter → `metrics/ppp/himanshu-e1.3-rm1500`. |
| 2026-11-17 | Pipeline simplified: `himanshu_lines.tsv`, `download_mcfo.py`, `himanshu_sbatch.sh`. Default switched to 20x at the native grid (`himanshu_20x`). |
| 2026-11-17 | f_brain cell swapped to 3-colour lines (§3). `--min-signal-channels 3` added. rm1500 dropped for Himanshu inference. |
