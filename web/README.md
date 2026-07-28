# FISBe 3D Image Viewer

Web scaffold for browsing and visualizing FISBe 3D microscopy volumes (Zarr, CZYX layout).

## Architecture

- **`client/`** — Vite + React + TypeScript frontend
- **`server/`** — FastAPI backend that reads Zarr volumes and serves 2D slices, MIPs, and downsampled 3D volumes

The API does not send full volumes to the browser; it extracts individual slices, maximum-intensity projections, or downsampled 3D volumes for interactive viewing.

For how a voxel becomes a pixel end-to-end — the server downsample/contrast pipeline, the vtk.js render path, and the client component/module breakdown (`VolumeViewer3D` + `vtkVolumeScene` + `VolumeControls`/`RangeSlider`) — see [`vizualize.md`](vizualize.md).

## Prerequisites

- Node.js 18+ (on Greene this comes from the user's `nvm` install, loaded by `~/.bashrc`)
- Python 3.10+ (on Greene this is the base conda env inside the `webdev.ext3` overlay)
- FISBe data extracted locally under `fisbe/completely/` (see root README for download instructions)

## Running on NYU Greene (compute node — recommended)

**Do not run the viewer on a login node.** Reading and downsampling the ~1.4 GB FISBe
volumes is memory- and CPU-heavy; on a shared login node a large MIP/volume request
can OOM and take your SSH session down with it. Run both processes on a compute node
instead.

The server runs **inside the Singularity `webdev` container** (its base conda env
carries `fastapi`/`uvicorn`/`zarr` — there is no login-node `.venv` to maintain). The
client runs on the host using `node` from `nvm`. A single SLURM job co-locates both on
one node so Vite's `/api` proxy works:

```bash
# from the repo root
sbatch sbatch/web/web.sh
```

Then find the compute node name and open a tunnel from your **laptop** (the job also
prints this line to `sbatch/web/web.out`):

```bash
squeue --me --name=web --states=R -o '%N'   # e.g. cs123
ssh -S none -N -L 9000:cs123:5173 $USER@login.torch.hpc.nyu.edu
```

Browse `http://localhost:9000`. Vite (port 5173 on the compute node) is forwarded to
local port `9000`; only that client port is tunneled — Vite proxies `/api` to the
server on the same node.

**Tunnel gotchas (learned the hard way):**

- **Run this in a plain terminal** (macOS Terminal.app / iTerm), **not** the integrated
  terminal of Cursor/VS Code. Those editors' *auto port-forwarding* scans terminal
  output and open files for port numbers and binds them locally first, so your `ssh -L`
  loses the race with `bind: Address already in use` — even on a fresh, unused port.
  (Alternatively, disable `remote.autoForwardPorts` / `remote.restoreForwardedPorts`
  and clear the editor's Ports panel.)
- **Use a distinct local port** (here `9000`) rather than `5173`. If the editor is
  connected to the same login node it will already be forwarding `5173`.
- **`-S none`** forces a standalone connection, bypassing any `ControlMaster`
  multiplexing in your `~/.ssh/config` (which otherwise fails with
  `mux_client_forward: ... master forward request failed`).
- `-N` means the terminal just hangs with no prompt — that's correct; it's holding the
  tunnel open. Leave it running.

One-time setup on the node (client deps): `cd web/client && npm install`. The server
container needs no install step. Vite is launched with `--host` so the login node can
reach it through the tunnel.

### Splitting server and client across nodes (advanced)

`sbatch/web/server.sh` and `sbatch/web/client.sh` run the two halves as separate jobs.
Because they land on different nodes, tell the client where the server is:

```bash
sbatch sbatch/web/server.sh                       # note its node, e.g. cs200
VITE_API_TARGET=http://cs200:8000 sbatch sbatch/web/client.sh
```

## Quick start (local machine)

### Terminal 1 — API server

```bash
cd web/server
python -m venv .venv # just need to be done once on setup
source .venv/bin/activate
pip install -r requirements.txt # just need to be done once on setup
FISBE_ROOT=../../fisbe/completely uvicorn main:app --reload --port 8000
```

### Terminal 2 — Frontend

```bash
cd web/client
npm install # just need to be done once on setup
npm run dev
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FISBE_ROOT` | `../../fisbe/completely` | Root directory containing `train/`, `val/`, `test/` Zarr folders |
| `SAMPLE_LIST_PATH` | `../../evaluate-instance-segmentation/assets/sample_list_per_split.txt` | Train/val/test sample list |
| `BIAPY_RESULTS_BASE` | `../../BiaPy/results` | Primary BiaPy results base (also used for legacy bare prediction-set ids) |
| `BIAPY_RESULTS_BASES` | `BiaPy/results:metrics/biapy` | Colon-separated list of bases scanned for BiaPy prediction sets (any run dir containing a `per_image_instances` folder). Default includes train-eval runs under `metrics/biapy` alongside `BiaPy/results`. |
| `BIAPY_RESULT_ROOT` | `../../BiaPy/results/train_3d_instance_segmentation/results/train_3d_instance_segmentation_1` | Default prediction set (used when the client doesn't pick one) |
| `PPP_EXPERIMENTS_BASE` | `../../PatchPerPix/experiments/ppp_experiments` | Base dir scanned for PatchPerPix prediction sets (per experiment: `numinst` count maps under `test/processed/`, `instances` vote-labels under `test/instanced/`) |

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness check |
| `GET` | `/api/prediction-sets` | List available prediction sets across models. Each carries a `source` (`biapy` \| `ppp`) and `kind`; BiaPy run dirs plus PatchPerPix `numinst`/`instances` overlays. |
| `GET` | `/api/samples` | List samples with split and path status (`has_predicted` is true if *any* set has output) |
| `GET` | `/api/samples/{name}/meta` | Volume shapes and dtypes (`prediction_set` query param selects which set's predicted shape to report) |
| `GET` | `/api/samples/{name}/metrics` | Scoring metrics for the sample (`prediction_set` selects the run): the `tests/metrics/<stem>.zarr.toml` scores and the matching `test_results_metrics.csv` row, each grouped by detection threshold. Shown in the right-hand **Scoring Metrics** column. |
| `GET` | `/api/aggregate-metrics` | Samples × metrics matrix across *all* samples for one prediction set (`prediction_set`, `threshold` query params). Backs the **Aggregate** tab's heatmap. |
| `GET` | `/api/samples/{name}/slice.png` | 2D slice (`volume`, `channel`, `axis`, `index` query params) |
| `GET` | `/api/samples/{name}/mip.png` | Maximum-intensity projection (`volume`, `channel` query params). `volume=raw&channel=all` → RGB raw MIP; `volume=gt&channel=all` → merged colored MIP of all `gt_instances` channels (matches the 3D GT overlay). The 3D tab shows both as live reference panels beneath the render. |
| `GET` | `/api/samples/{name}/volume.bin` | Downsampled 3D volume for MIP rendering (`volume=raw|gt|predicted`, `channel=0|1|2|all`, `max_size` 64–512, `prediction_set` selects which set for `volume=predicted`) |

The **3D Viewer** tab loads downsampled raw data from FISBe Zarr and, when available, overlays predicted instances (`volume=predicted`) from the selected prediction set. BiaPy sets are loaded via [`ipynb/scripts/biapy.py`](../ipynb/scripts/biapy.py); PatchPerPix sets via [`server/services/ppp_loader.py`](server/services/ppp_loader.py). A dispatcher ([`server/services/predictions.py`](server/services/predictions.py)) routes each request to the right loader by the set's `source`. Drag or use arrow keys to rotate the view.

### Aggregate tab

The **Aggregate** tab renders a samples × metrics heatmap for the selected prediction
set, so per-sample scores can be compared at a glance instead of one sample at a time.
It spans every sample, so it ignores the sidebar selection (and hides the per-sample
**Scoring Metrics** column, which would be redundant).

Columns are the curated "core quality" set defined by `METRIC_SPECS` in
[`server/services/aggregate_metrics.py`](server/services/aggregate_metrics.py) —
`iou (f/c channel)`, `avAP`, `avFscore`, `avg_TP_05_cldice`, `TP_05_rel` (threshold-free),
plus `precision`/`recall`/`f1`/`panoptic_quality` at a selectable detection
threshold (0.3 / 0.5 / 0.75, default 0.5). Add or drop a column by editing
`METRIC_SPECS`; the client renders whatever the endpoint lists.

**Each column is colour-scaled to its own min/max across samples**, since the metrics
span wildly different ranges here (IoU ≈ 0.39–0.68 while avAP ≈ 0–0.0004) — a shared
0–1 scale would render every AP-family column uniformly dark. Cell colour is therefore
only comparable *within* a column, never across columns; the tooltip carries the exact
value. A column whose samples are all equal renders mid-ramp. Like the per-sample
panel, this is **BiaPy-only** — PatchPerPix sets produce an empty matrix.

Two **mean / median** rows close the table, summarising each column across samples
(endpoint field `summary`, with `summary.n` giving how many samples each statistic was
taken over — a column with missing cells is averaged only over the ones present).
They are *derived* rows rather than samples, so they are returned separately from
`samples`/`values`, are excluded from the colour normalisation, and render as plain
unfilled numbers below a rule.

### Selecting a prediction set

`/api/prediction-sets` merges sets from every model source, so you can compare predictions across setups. The **Predictions** dropdown above the viewer tabs (grouped by **BiaPy** / **PatchPerPix**) switches which set is overlaid; the predicted overlay and reported shapes update accordingly. The set marked `default` (a BiaPy set matching `BIAPY_RESULT_ROOT`) is selected on load.

- **BiaPy** — every run directory under each path in `BIAPY_RESULTS_BASES` (default: `BiaPy/results` and `metrics/biapy`) with a `per_image_instances` folder. Set ids are repo-relative so the two trees do not collide. Rendered as per-neuron coloured instance labels.
- **PatchPerPix** — every experiment under `PPP_EXPERIMENTS_BASE`, each exposing up to two overlays:
  - **numinst** — the per-voxel overlap-count map (`test/processed/<ckpt>/<stem>.zarr` → `volumes/pred_numinst`, `(3, Z, Y, X)` = P(0/1/2+ instances)). This is a foreground/count probability map, **not** per-neuron labels, so it renders as a two-colour foreground (argmax of the count channels: 1-instance vs 2+-overlap regions). Streamed in Z-blocks so the ~0.5 GB float16 array is never fully loaded.
  - **instances** — the final vote-instances labels (`test/instanced/.../<stem>.hdf` → dataset `vote_instances`), coloured per neuron like the BiaPy/GT overlays. **Requires `h5py`** in the server env to read the HDF5 output.

Scoring metrics in the right-hand column are currently BiaPy-only; PatchPerPix sets report no metrics.

## Follow-ups (not in scaffold)

- h5j / MCFO raw stack support via PyImageJ pipeline
- Full-resolution in-browser 3D volume rendering (tile streaming)
- PatchPerPix scoring metrics in the **Scoring Metrics** column
- Authentication and production deployment
