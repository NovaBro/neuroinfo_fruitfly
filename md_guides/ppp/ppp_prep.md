# PPP data prep (`sbatch/ppp/prep`)

Jobs that build or refresh datasets under `fisbe/` before
[`ppp_submit_train.sh`](../../sbatch/ppp/ppp_submit_train.sh) /
[`ppp_submit_test.sh`](../../sbatch/ppp/ppp_submit_test.sh).
Submit every command from the repo root. Logs land in `sbatch/ppp/logs/%x-%j.{out,err}`.

Training order, when a dataset changes:

1. Dataset-specific build (synth or himanshu), if the zarrs do not exist yet.
2. `prepare_fisbe_for_ppp_sbatch.sh` — PPP arrays inside each zarr. Sqf-backed
   datasets with more than 50 zarrs (`1k_v1`, `1k_v1_u8`) use
   `pack_ppp_sqf_sbatch.sh` instead, so the chunk files stay inside SquashFS.
3. `prepare_fg_points_sbatch.sh` — only if the preset sets `sample_fg_points = true`.
4. `prepare_ppp_stage_tars_sbatch.sh` — tars that train jobs extract into `$SLURM_TMPDIR`.
   Skip this for sqf-packed datasets. Train jobs mount the images and read
   the chunks through FUSE. Extracting the tar writes the files out again.

Himanshu is inference-only, so it stops after its convert job. Full chain:
[`ppp_new_dataset_inference.md`](ppp_new_dataset_inference.md).

---

## 1. Structure

```text
sbatch/ppp/prep/
  prepare_fisbe_for_ppp_sbatch.sh    # shared: any dataset under fisbe/
  pack_ppp_sqf_sbatch.sh             # large sqf-backed sets → fisbe/sqf/<dataset>/
  prepare_fg_points_sbatch.sh
  prepare_ppp_stage_tars_sbatch.sh
  himanshu/                          # Gen1 MCFO → inference zarrs
  synth/                             # vh2308 synth_MCFO → fisbe/synth_vh_u8
  fisbe/                             # completely/partly one-offs
```

The scripts at the top take `DATASETS` or `SPLITS`. `prepare_fisbe_for_ppp_sbatch.sh`,
`prepare_fg_points_sbatch.sh`, and `prepare_ppp_stage_tars_sbatch.sh` are the
ones preset TOML headers paste. `pack_ppp_sqf_sbatch.sh` replaces loose prepare
for a large sqf-backed dataset. `fisbe` in those filenames is the root directory
(`--fisbe-root fisbe`), not the `completely` dataset.

Subfolders are jobs that only make sense for one source. After them, synth
still goes through prepare (or `pack_ppp_sqf_sbatch.sh` when the set is large),
then foreground points and stage tars when those apply. Himanshu does not: it writes
`raw` and `raw_normalized` itself and has no train split.

---

## 2. What each file does

### Shared

| Script | Calls | Writes |
|--------|--------|--------|
| `prepare_fisbe_for_ppp_sbatch.sh` | `PatchPerPix/experiments/flylight/prepare_fisbe_for_ppp.py` | In each `fisbe/<dataset>/{train,val,test}/*.zarr`: `volumes/raw_normalized` (clip to `CLIPMAX`, divide), `gt_numinst`, `gt_fg_rm_5` (opening, radius 1), `gt_instances_rm_5` (labels copied; opening is only on the mask). Chunks are 64³. Empty chunks are not stored. Refuses a source `.sqf` dataset with more than 50 zarrs unless `ALLOW_LOOSE=1`. |
| `pack_ppp_sqf_sbatch.sh` | `prepare_fisbe_for_ppp.py`, then `mksquashfs` | One SquashFS image per batch under `fisbe/sqf/<dataset>/batch_NNNN.sqf`. Scratch zarrs keep `raw` and `gt_instances` symlinks into the source image. The four PPP arrays become symlinks to `/synth_MCFO/<dataset>_ppp/...`. Train and eval jobs mount those images. |
| `prepare_fg_points_sbatch.sh` | `prepare_fg_points_csv.py` | `fisbe/fg_points/<dataset>/<sample>.csv`, space-separated `z y x` on a stride-4 grid of `gt_fg_rm_5`. Training uses these instead of building a mask integral at startup. Requires step 1. |
| `prepare_ppp_stage_tars_sbatch.sh` | `prepare_ppp_stage_tars.py` | `fisbe/<dataset>/train_ppp.tar` and `val_ppp.tar`. Each tar holds the PPP arrays plus `gt_instances`, not `volumes/raw`. Train jobs with `STAGE_DATA=1` extract them to `$SLURM_TMPDIR/ppp_data/<dataset>/{train,val}/`. |

### `himanshu/`

| Script | Calls | Writes |
|--------|--------|--------|
| `prepare_himanshu_for_ppp_sbatch.sh` | `data_download_methods/prepare_himanshu_for_ppp.py` | `fisbe/himanshu/test/*.zarr` from aligned Gen1 MCFO `.h5j` stacks (brain and VNC). Three channels, `volumes/raw` and `volumes/raw_normalized`. Needs PyAV in the `ppp` overlay. |
| `delete_himanshu_zarrs_sbatch.sh` | `find` + `rm` | Deletes everything inside `fisbe/himanshu/test/` (zarrs and sidecars such as `manifest.csv`) and leaves the directory. Run this on a compute node; the tree is many small chunk files. |

### `synth/`

| Script | Calls | Writes |
|--------|--------|--------|
| `build_synth_from_sqf_sbatch.sh` | `build_synth_vh.py` | `fisbe/<sqf basename>/{train,val,test}` from a SquashFS image under `/scratch/vh2308/data/synth_MCFO`. `volumes/raw` and `volumes/gt_instances` are symlinks to `/synth_MCFO/<name>/`. They resolve only while that `.sqf` is mounted. |
| `build_synth_vh_sbatch.sh` | `build_synth_vh.py` | `fisbe/synth_vh_u8/{train,val,test}/<sample>.zarr` from vh2308 `synth_MCFO` samples. `volumes/raw` and `volumes/gt_instances` are symlinks into that tree unless `COPY=1`. |
| `materialize_synth_vh_u8_from_sqf_sbatch.sh` | `materialize_synth_links.py` | Replaces those symlinks with real copies read from `v1_u8.sqf` (mounted at `/synth_MCFO/v1_u8`). Each copy is byte-compared to the symlink target before the swap. Default splits are `val,test`: train already reads `train_ppp.tar`, and copying train adds on the order of 80k files. |

### `fisbe/`

| Script | Calls | Writes |
|--------|--------|--------|
| `check_gt_instance_connectivity_sbatch.sh` | `imaging_helpers_hpc/check_gt_instance_connectivity.py` | `imaging_helpers_hpc/output/gt_instance_connectivity.csv`. One row per non-empty `gt_instances` channel in `fisbe/completely/{train,val,test}`: voxel count and number of 26-connected components. Uses the compute-node login Python (`zarr`/`scipy`), not the `ppp` container. |
| `prepare_fisbe_rechunk_sbatch.sh` | `prepare_fisbe_for_ppp.py` with `--overwrite` and no `--datasets` | Rewrites the PPP arrays at 64³ chunks. The Python default datasets are `completely,partly` only. `clipmax` stays 1500. |

---

## 3. How to use each file

All `sbatch` lines are from the repo root. Override a variable by putting it
on the same command line.

### `prepare_fisbe_for_ppp_sbatch.sh`

CPU, 4 cores, 64 GB, 8 hours, partition `cs`. Container `env/ppp.ext3`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATASETS` | `partly` | Comma-separated names under `fisbe/`. Each existing `train`/`val`/`test` is prepared. |
| `OVERWRITE` | `1` | `0` skips a zarr that already has all four PPP arrays. |
| `CLIPMAX` | `1500` | Raw is clipped to this, then divided by it. Use `255` for uint8 synth (`synth_vh_u8`). `1k_v1.sqf` is the non-uint8 image, so leave this at 1500; `1k_v1_u8.sqf` is a separate dataset. |
| `SQF` | auto | Comma-separated `.sqf` paths under `/scratch/vh2308/data/synth_MCFO`. Unset: for each name in `DATASETS`, mount `<name>.sqf` when that file exists, read-only at `/synth_MCFO/<name>`. `build_synth_from_sqf_sbatch.sh` leaves `volumes/raw` and `volumes/gt_instances` as symlinks into that mount. The job exits if a link names an image that is not mounted. |
| `ALLOW_LOOSE` | `0` | `1` allows loose prepare of a source `.sqf` dataset with more than 50 zarrs. Leave this off for `1k_v1` and `1k_v1_u8`. |

```bash
DATASETS=completely,partly,synth_vh sbatch sbatch/ppp/prep/prepare_fisbe_for_ppp_sbatch.sh

OVERWRITE=0 DATASETS=partly sbatch sbatch/ppp/prep/prepare_fisbe_for_ppp_sbatch.sh

CLIPMAX=255 DATASETS=synth_vh_u8 sbatch sbatch/ppp/prep/prepare_fisbe_for_ppp_sbatch.sh
```

`ALLOW_LOOSE=1` turns the large-dataset refusal off. Do not use it for `1k_v1` or `1k_v1_u8`.

### `pack_ppp_sqf_sbatch.sh`

CPU, 4 cores, 96 GB, 8 hours, partition `cs`. The first submission counts
`fisbe/<dataset>/{train,val,test}/*.zarr` and submits one array task per batch.
Each task prepares that batch on `$SLURM_TMPDIR` (source `.sqf` mounted), packs
`synth_MCFO/<dataset>_ppp/` with `mksquashfs -keep-as-directory -mem 16G`, and
leaves one file at `fisbe/sqf/<dataset>/batch_NNNN.sqf`.

Skeletons must already exist (`build_synth_from_sqf_sbatch.sh`). `ppp_job.sh`
mounts the source image and every `fisbe/sqf/<dataset>/*.sqf` when a
`volumes/raw` or prepared-array symlink points at `/synth_MCFO/`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATASETS` | required | One folder name under `fisbe/`. Not a comma-separated list. |
| `BATCH` | `40` | Brains per array task. Changing it changes which brains land in each image; rerun with `OVERWRITE=1`. |
| `OVERWRITE` | `0` | `1` rebuilds a batch image that already exists. `0` only refreshes scratch symlinks. |
| `CLIPMAX` | `1500` | Same normalization as loose prepare. Use `255` for uint8 synth (`1k_v1_u8`, `synth_vh_u8`). `93` matches FISBe's 1500 ADU after `rint(v*255/4095)`. |

```bash
SQF=/scratch/vh2308/data/synth_MCFO/1k_v1.sqf \
  sbatch sbatch/ppp/prep/synth/build_synth_from_sqf_sbatch.sh
DATASETS=1k_v1 sbatch sbatch/ppp/prep/pack_ppp_sqf_sbatch.sh

SQF=/scratch/vh2308/data/synth_MCFO/1k_v1_u8.sqf \
  sbatch sbatch/ppp/prep/synth/build_synth_from_sqf_sbatch.sh
CLIPMAX=255 DATASETS=1k_v1_u8 sbatch sbatch/ppp/prep/pack_ppp_sqf_sbatch.sh
```

One batch of one brain (`BATCH=1` and a single-task array, or a skeleton that contains only that brain) is the check before the full 1,000: scratch should gain a skeleton plus one `.sqf`, and inside the container `raw_normalized` and `gt_instances_rm_5` should open through the scratch symlinks.

### `prepare_fg_points_sbatch.sh`

CPU, 4 cores, 64 GB, 2 hours, partition `cs` or `cpu_short`. Always overwrites
existing CSVs. Stride is fixed at 4. Fails if `volumes/gt_fg_rm_5` is missing.
Sqf-packed datasets (`1k_v1`, `1k_v1_u8`) need `fisbe/sqf/<dataset>/*.sqf`
already built by `pack_ppp_sqf_sbatch.sh`. This job mounts those shards itself
so the `gt_fg_rm_5` symlinks into `/synth_MCFO/<dataset>_ppp/` resolve. It
exits if a prepared-array symlink points at `/synth_MCFO/` and no shard matches.

| Variable | Default | Meaning |
|----------|---------|---------|
| `SPLITS` | `partly/train,partly/val,partly/test` | Comma-separated `<dataset>/<split>`. The dataset name is the first path component and becomes the output folder. |

```bash
SPLITS=completely/train,completely/val,completely/test \
  sbatch sbatch/ppp/prep/prepare_fg_points_sbatch.sh

SPLITS=synth_vh_u8/train,synth_vh_u8/val,synth_vh_u8/test \
  sbatch sbatch/ppp/prep/prepare_fg_points_sbatch.sh

SPLITS=1k_v1/train,1k_v1/val,1k_v1/test \
  sbatch sbatch/ppp/prep/prepare_fg_points_sbatch.sh
```

### `prepare_ppp_stage_tars_sbatch.sh`

CPU, 4 cores, 16 GB, 4 hours, partition `cs`. Always overwrites the tars.
Run after the PPP arrays exist. `ppp_job.sh` prints this command when a
train job cannot find `train_ppp.tar` / `val_ppp.tar`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATASETS` | `completely` | Comma-separated names under `fisbe/`. |
| `SQF` | auto | Same as `prepare_fisbe_for_ppp_sbatch.sh`. The tar copies `gt_instances` through the SquashFS symlinks, so this job mounts the image too. `STAGE_DATA=1` train jobs then read the tar. |

```bash
DATASETS=completely,partly,synth_vh sbatch sbatch/ppp/prep/prepare_ppp_stage_tars_sbatch.sh
```

Do not build these tars for `1k_v1` or `1k_v1_u8`. Those datasets are read from
`fisbe/sqf/<dataset>/*.sqf`.

### `himanshu/prepare_himanshu_for_ppp_sbatch.sh`

CPU, 4 cores, 64 GB, 24 hours, partition `cs`.

Prerequisites: `downloads/hemidrivers.tsv` and the Gen1 stacks
(`sbatch/data/s3_download_meta_sbatch.sh`, `s3_download_gen1_sbatch.sh`),
and PyAV in the overlay (`sbatch sbatch/ppp/env/ppp_env_install_av_sbatch.sh`,
exclusive `:rw` — no other PPP job on that overlay).

| Variable | Default | Meaning |
|----------|---------|---------|
| `TSV` | `downloads/hemidrivers.tsv` | Sample table. |
| `DOWNLOADS` | `downloads/Gen1 MCFO` | Stack root. |
| `OUT` | `fisbe/himanshu/test` | Output directory. |
| `CLIPMAX` | `1500` | Same normalization as FISBe prep. |
| `OVERWRITE` | `0` | `1` rewrites zarrs that already exist. |
| `LIMIT` | all | Convert only the first N stacks (smoke). |

```bash
LIMIT=1 sbatch sbatch/ppp/prep/himanshu/prepare_himanshu_for_ppp_sbatch.sh

sbatch --dependency=afterok:<gen1_jid>:<av_jid> \
  sbatch/ppp/prep/himanshu/prepare_himanshu_for_ppp_sbatch.sh
```

### `himanshu/delete_himanshu_zarrs_sbatch.sh`

CPU, 2 cores, 8 GB, 4 hours, partition `cs`. Exits 0 if `OUT` does not exist.

| Variable | Default | Meaning |
|----------|---------|---------|
| `OUT` | `fisbe/himanshu/test` | Directory whose children are removed. |

```bash
DEL_JID=$(sbatch --parsable sbatch/ppp/prep/himanshu/delete_himanshu_zarrs_sbatch.sh)
sbatch --dependency=afterok:${DEL_JID} \
  sbatch/ppp/prep/himanshu/prepare_himanshu_for_ppp_sbatch.sh
```

### `synth/build_synth_vh_sbatch.sh`

CPU, 2 cores, 8 GB, 1 hour, partition `cs`. Default is symlinks, so the
source tree must stay readable.

| Variable | Default | Meaning |
|----------|---------|---------|
| `SRC` | `/scratch/vh2308/data/synth_MCFO/v1_u8/samples` | Source samples. |
| `OUT` | `fisbe/synth_vh_u8` | Destination dataset root. |
| `SAMPLE_LIST` | `.../v1_u8/sample_list_per_split.txt` | Split manifest. Empty string skips the flag. |
| `COPY` | `0` | `1` copies `raw` and `gt_instances` instead of linking them. |

```bash
sbatch sbatch/ppp/prep/synth/build_synth_vh_sbatch.sh

COPY=1 sbatch sbatch/ppp/prep/synth/build_synth_vh_sbatch.sh
```

Then run the shared PPP-array, FG-point, and stage-tar jobs with
`DATASETS=synth_vh_u8` / the matching `SPLITS`, and `CLIPMAX=255`.

### `synth/materialize_synth_vh_u8_from_sqf_sbatch.sh`

CPU, 2 cores, 8 GB, 1 hour, partition `cs`. Mounts `v1_u8.sqf` read-only
beside `env/ppp.ext3`. Run after `build_synth_vh` if the dataset should not
depend on the loose sample tree.

| Variable | Default | Meaning |
|----------|---------|---------|
| `SQF` | `/scratch/vh2308/data/synth_MCFO/v1_u8.sqf` | SquashFS image. |
| `DATASET_ROOT` | `fisbe/synth_vh_u8` | Dataset whose symlinks are replaced. |
| `SPLITS` | `val,test` | Comma-separated splits. Add `train` only if you accept the extra files. |
| `DRY_RUN` | `0` | `1` passes `--dry-run`. |

```bash
DRY_RUN=1 sbatch sbatch/ppp/prep/synth/materialize_synth_vh_u8_from_sqf_sbatch.sh

sbatch sbatch/ppp/prep/synth/materialize_synth_vh_u8_from_sqf_sbatch.sh
```

### `fisbe/check_gt_instance_connectivity_sbatch.sh`

CPU, 32 cores, 128 GB, 2 hours, partition `cs`. Account is
`torch_pr_61_tandon_priority`. No environment variables; the root, splits,
and CSV path are fixed in the script.

```bash
sbatch sbatch/ppp/prep/fisbe/check_gt_instance_connectivity_sbatch.sh
```

### `fisbe/prepare_fisbe_rechunk_sbatch.sh`

CPU, 4 cores, 64 GB, 4 hours, partition `cs` or `cpu_short`. One-off: same
work as `prepare_fisbe_for_ppp_sbatch.sh` with `OVERWRITE=1` and
`DATASETS=completely,partly`. No environment variables. Do not use it to
refresh synth; pass `DATASETS` to the shared script instead.

```bash
sbatch sbatch/ppp/prep/fisbe/prepare_fisbe_rechunk_sbatch.sh
```
