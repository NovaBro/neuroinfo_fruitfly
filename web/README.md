# FISBe 3D Image Viewer

Web scaffold for browsing and visualizing FISBe 3D microscopy volumes (Zarr, CZYX layout).

## Architecture

- **`client/`** — Vite + React + TypeScript frontend
- **`server/`** — FastAPI backend that reads Zarr volumes and serves 2D slices, MIPs, and downsampled 3D volumes

The API does not send full volumes to the browser; it extracts individual slices, maximum-intensity projections, or downsampled 3D volumes for interactive viewing.

## Prerequisites

- Node.js 18+
- Python 3.10+
- FISBe data extracted locally under `fisbe/completely/` (see root README for download instructions)

## Quick start

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
<!-- \. "$HOME/.nvm/nvm.sh" -->

### (Local) Terminal 3 - Connect Remotely
HPC Web Dev
ssh -L 8000:localhost:5173 -J wmz2007@login.torch.hpc.nyu.edu wmz2007@torch-login-b-1
Open [http://localhost:5173](http://localhost:5173).

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FISBE_ROOT` | `../../fisbe/completely` | Root directory containing `train/`, `val/`, `test/` Zarr folders |
| `SAMPLE_LIST_PATH` | `../../evaluate-instance-segmentation/assets/sample_list_per_split.txt` | Train/val/test sample list |
| `BIAPY_RESULT_ROOT` | `../../BiaPy/results/3d_instance_segmentation/results/3d_instance_segmentation_1` | BiaPy per-image / instance TIFF outputs |

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness check |
| `GET` | `/api/samples` | List samples with split and path status |
| `GET` | `/api/samples/{name}/meta` | Volume shapes and dtypes |
| `GET` | `/api/samples/{name}/slice.png` | 2D slice (`volume`, `channel`, `axis`, `index` query params) |
| `GET` | `/api/samples/{name}/mip.png` | Maximum-intensity projection (`volume`, `channel` query params) |
| `GET` | `/api/samples/{name}/volume.bin` | Downsampled 3D volume for MIP rendering (`volume=raw|gt|predicted`, `channel=0|1|2|all`, `max_size` 64–512) |

The **3D Viewer** tab loads downsampled raw data from FISBe Zarr and, when available, overlays BiaPy predicted instances (`volume=predicted`) using helpers from [`ipynb/scripts/biapy.py`](../ipynb/scripts/biapy.py). Drag or use arrow keys to rotate the view.

## Follow-ups (not in scaffold)

- h5j / MCFO raw stack support via PyImageJ pipeline
- Full-resolution in-browser 3D volume rendering (tile streaming)
- Segmentation model output comparison (BiaPy / PatchPerPix)
- Authentication and production deployment
