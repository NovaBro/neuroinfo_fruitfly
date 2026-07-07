# How the FISBe Viewer Visualizes Data

This document explains, end-to-end, how the `web/` app turns large 3D microscopy
volumes on disk into interactive pictures in the browser. It complements
[`README.md`](README.md) (which covers *running* the app) by focusing on the
*visualization mechanics*: what happens to a voxel between the Zarr store and a
pixel on screen.

---

## 1. The core problem

FISBe volumes are **Zarr arrays in `CZYX` layout** (channels, Z, Y, X), and a
single channel is on the order of **~1 GB**. You cannot ship that to a browser,
and you cannot even load a whole channel into host RAM carelessly without risking
an OOM. So the guiding rule of the whole design is:

> **The browser never receives a full-resolution volume.** The server always
> extracts a *reduced* representation — a 2D slice, a maximum-intensity
> projection (MIP), or a heavily *downsampled* 3D volume — before anything
> crosses the network.

Everything below is a consequence of that rule.

---

## 2. The three visualization modalities

The app offers three ways to look at a sample, exposed as two UI tabs:

| Modality | Tab | What the server sends | What renders it |
|----------|-----|-----------------------|-----------------|
| **2D orthogonal slice** | Slice / MIP Viewer | a `.png` of one Z/Y/X plane | `<img>` (`SliceImage`) |
| **Maximum-intensity projection (MIP)** | Slice / MIP Viewer + 3D tab | a `.png` (on-the-fly, or pre-shipped FISBe MIP) | `<img>` (`SliceImage`) |
| **Interactive 3D volume** | 3D Viewer | raw `uint8` bytes of a downsampled volume | **vtk.js** GPU volume renderer |

Each modality corresponds to a server endpoint and a client component:

```
                             ┌─────────────────────────────────────────┐
                             │            FastAPI (server/)             │
  Browser (client/)          │                                          │
  ─────────────────          │  GET /slice.png   → zarr_reader          │
  OrthoSliceViewer  ───────► │  GET /mip.png     → zarr_reader          │
    └─ SliceImage (<img>)    │  GET /fisbe_mip.png → fisbe_mip          │
                             │  GET /volume.bin  → zarr_reader /        │
  VolumeViewer3D    ───────► │                     biapy_loader         │
    └─ vtk.js volume mapper  │  GET /meta,/metrics,/samples,...         │
                             └─────────────────────────────────────────┘
                                              │  reads
                                              ▼
                             Zarr volumes (CZYX)  +  BiaPy TIFF predictions
```

---

## 3. Server side: from Zarr voxels to a transmittable picture

All server extraction lives in `server/services/`. The two workhorses are
`zarr_reader.py` (raw + ground-truth from Zarr) and `volume_pipeline.py` (the
shared downsample/contrast/encode math). `biapy_loader.py` supplies predicted
instances, and `fisbe_mip.py` serves pre-rendered PNGs.

### 3.1 The shared display pipeline (`volume_pipeline.py`)

Every image the server produces passes through the same **contrast enhancement**
so raw, GT, and predicted layers look consistent:

```
enhance_display_values(data):
    1. cast to float32
    2. sample nonzero voxels
    3. percentile clip:  p_low = 1.0 %   p_high = 99.5 %
    4. linear stretch to [0, 1]
    5. gamma brighten:   out = scaled ** 0.72     (γ<1 lifts mid-tones)
```

This is what makes dim fluorescence readable. `to_display_uint8()` then scales to
`0–255`. For multi-channel raw data, `normalize_rgb_stack()` / `to_display_rgb_*`
apply the *same* stretch across channels and pack them into an RGB image.

### 3.2 2D slices (`slice_to_png`)

```
        CZYX zarr array
              │  arr[channel, index, :, :]      (axis = z)
              │  arr[channel, :, index, :]       (axis = y)
              │  arr[channel, :, :, index]       (axis = x)
              ▼
        2D numpy plane
              │
    raw ──────┤ contrast-stretch → grayscale "L"
    gt  ──────┤ label id → RGB via _GT_COLORS palette
   raw+all ───┤ 3 channels → RGB "RGB"
              ▼
        PIL.Image → PNG bytes  (@lru_cache 512)
```

Only the requested plane is pulled out of Zarr, so a slice is cheap regardless of
volume size. Ground-truth label IDs are mapped to a small fixed color table so
each neuron reads as a distinct color.

### 3.3 MIPs (`mip_to_png`)

A maximum-intensity projection flattens the whole Z stack into one image by taking
the brightest voxel along each viewing ray. The catch: a full channel is too big
to load at once, so the projection is done **in bounded Z-blocks**:

```
_max_project_channel(arr, channel):
    out = None
    for zi in 0, 32, 64, ... :                 # _MIP_Z_BLOCK = 32
        block = arr[channel, zi:zi+32].max(axis=0)
        out   = max(out, block)                # running per-pixel max
    return out                                 # never holds > 32 planes
```

This bounds memory to ~32 Z-planes at a time — the key trick that keeps MIPs from
OOM-ing the host on gigabyte channels.

### 3.4 Downsampled 3D volume (`volume_to_bytes`, `volume.bin`)

For the 3D viewer the server must send an actual volume — just a small one. It
computes an integer downsample factor from a client-supplied `max_size` (the
longest edge, 64–512) and **block max-pools** the Zarr array down to that size:

```
compute_downsample_factor(shape, max_size) = ceil(max(z,y,x) / max_size)

downsample_max_pool_zarr(channel, factor):     # streams Z, never loads full vol
    for each output-Z block:
        z_max   = max over `factor` input Z-planes
        pooled  = z_max.reshape(...).max over Y,X blocks
```

Max-pooling (not averaging) is deliberate: thin bright neurites survive
downsampling instead of being averaged into the dark background.

The result is packed into a `VolumeBytesResult` — raw `uint8` bytes plus metadata
returned as **HTTP headers** (not JSON), so the body stays a tight binary blob:

| Header | Meaning |
|--------|---------|
| `X-Volume-Shape` | downsampled `z,y,x` |
| `X-Original-Shape` | full-res `z,y,x` (for aspect ratio) |
| `X-Downsample-Factor` | integer pooling factor |
| `X-Volume-Components` | `1` = grayscale, `3` = RGB |

Three encodings are produced depending on `volume` / `channel`:

- **`raw`, single channel** → grayscale (`components=1`), contrast-stretched.
- **`raw`, `channel=all`** → RGB (`components=3`), channels stacked.
- **`gt`** → all GT channels merged into one colored instance volume. Each
  `(channel, label)` pair gets a globally unique id (`label + channel*1000`) so
  distinct neurons keep distinct colors, then `encode_label_volume_rgb` assigns
  each id a deterministic vivid RGB (seeded RNG → reproducible palette).
- **`predicted`** → same `labels_rgb` encoding, but the source volume comes from
  BiaPy (below).

### 3.5 Predicted instances (`biapy_loader.py`)

Predicted overlays don't come from Zarr; they come from **BiaPy run directories**
(`BiaPy/results/**/per_image_instances/*.tif`). `biapy_loader` bridges to the
shared helpers in `ipynb/scripts/biapy.py`:

```
load_biapy_per_image_instances(stem)  →  (Z,Y,X) label volume
        │
        └─► volume_array_to_bytes(..., encoding="labels_rgb")   # same pipeline as GT
```

A "prediction set" is one run directory; `discover_prediction_sets()` scans for
any dir containing `per_image_instances`, which is how the **Predictions**
dropdown lets you compare different training/testing setups. The set overlaid is
selected per-request via the `prediction_set` query param.

### 3.6 Pre-generated FISBe MIPs (`fisbe_mip.py`)

Separately from the on-the-fly `mip.png`, FISBe *ships* a MIP PNG per sample
(`fisbe_v1.0_mips.zip`). Each is a side-by-side composite: **left = original
colored raw MIP, right = GT-segmentation MIP**. `fisbe_mip_png` just locates the
file and crops it to the requested half (`full` / `raw` / `gt`) — no projection
computed. The 3D tab shows this beneath the live render as a reference.

---

## 4. Client side: from bytes to pixels

### 4.1 The API layer (`api/client.ts`)

Pure URL builders + fetch helpers. `sliceUrl` / `mipUrl` / `fisbeMipUrl` return
`.png` URLs; `fetchVolumeData` fetches `volume.bin`, reads the `X-*` headers, and
returns a typed `VolumeData { data: Uint8Array, shape, originalShape,
downsampleFactor, components }`.

### 4.2 PNG-based views: `SliceImage`

Slices and MIPs are just images. `SliceImage`:

1. `fetch`es the PNG as a **blob** (so an in-flight request can be aborted when
   you scrub the slice slider), creates an object URL, sets it as `<img src>`.
2. Applies **brightness/contrast live via a CSS `filter`** (`cssBrightnessContrast`)
   — no re-fetch needed to adjust display.
3. Revokes the object URL on cleanup to avoid leaks.

`OrthoSliceViewer` composes these: it picks axis + index (slider) or MIP mode,
builds the raw URL, and optionally stacks a **GT overlay** as a second
absolutely-positioned `SliceImage` with adjustable opacity. Slider scrubbing is
debounced (`useDebouncedValue`, 200 ms) so dragging doesn't spam the server.

```
OrthoSliceViewer
  ├─ controls: view(slice|mip) · channel · brightness · contrast · axis · index · GT
  └─ canvas
       ├─ SliceImage(raw  url)        ← base layer
       └─ SliceImage(gt   url, opacity)  ← overlay (absolute-positioned)
```

### 4.3 The 3D viewer: `VolumeViewer3D` + vtk.js

This is the most involved piece. It uses **[vtk.js](https://kitware.github.io/vtk-js/)**
GPU volume rendering. The flow:

```
fetchVolumeData ──► Uint8Array + shape/components
        │
        ▼
vtkImageData                      (the 3D grid)
  ├─ setDimensions([x, y, z])
  ├─ setSpacing([1, oy/ox, oz/ox]) ← restores true aspect ratio from originalShape
  └─ pointData.setScalars(vtkDataArray, numberOfComponents = 1 or 3)
        │
        ▼
vtkVolumeMapper
  └─ setBlendModeToMaximumIntensity()   ← 3D render IS a GPU-side MIP
        │
        ▼
vtkVolume + property (color + opacity transfer functions)
        │
        ▼
vtkGenericRenderWindow → WebGL canvas
   + TrackballCamera interactor (drag / arrow keys to rotate)
```

Key details:

- **Three independent layers coexist** in one render window — `raw`, `predicted`,
  and `gt` — each its own `vtkVolume`/mapper/imageData. Checkboxes toggle
  visibility and per-layer opacity sliders scale their transfer functions, so you
  can overlay predicted instances on raw data (or compare against GT) in 3D.
- **Blend mode = Maximum Intensity.** The GPU casts rays and keeps the brightest
  sample — the 3D analogue of the server-side MIP, and what makes sparse neurons
  pop against a dark background.
- **Transfer functions** (`configureVolumeProperty`) map scalar value → color and
  → opacity. Raw uses a grayscale ramp with an opacity curve that suppresses dim
  background and reveals bright structure; instance layers (`instance_rgb`) render
  the pre-colored RGB directly.
- **The RGB brightness fix** (documented at length in the code): for dependent
  3-component data vtk.js scales stored values by the transfer function's *range*.
  A null function defaults to range `[0,1024]`, which would render `uint8` RGB at
  ~25% brightness ("dark filter"). Setting the range to `[0,255]` makes the scale
  ≈1 so colors render at full brightness.
- **Brightness/contrast are applied in JS**, not just CSS: `remapVolumeUint8`
  (`displayAdjust.ts`) re-maps the `uint8` scalars through a
  brightness/contrast/gamma curve before handing them to vtk. For RGB it scales
  each voxel *uniformly* (by luminance) to lift brightness without shifting hue.
- **`max_size` (resolution slider)** drives the server downsample factor; it's
  debounced (400 ms) so dragging it doesn't refetch on every tick. Lower = faster,
  higher = sharper.
- **Lifecycle safety.** A `vtkGenerationRef` counter guards against stale async
  results: if the sample/effect changes mid-fetch, results from the old generation
  are dropped and the vtk context is torn down cleanly (`removeVolume` + `delete`).

### 4.4 Top-level composition (`App.tsx`)

```
App
 ├─ SampleBrowser (sidebar)     ── pick a sample (split-grouped, "has predicted" badge)
 ├─ Predictions <select>        ── choose which BiaPy run to overlay
 ├─ tabs: Slice/MIP | 3D
 │    ├─ OrthoSliceViewer
 │    └─ VolumeViewer3D
 └─ MetricsPanel (right)        ── scoring metrics for the selected sample/set
```

`useSampleMeta` fetches shapes/dtypes (which drive channel/axis ranges and the
"has GT / has predicted" flags); `useSampleMetrics` fetches the scoring numbers.

---

## 5. End-to-end example: rendering the 3D raw + predicted overlay

```
User picks sample "R38F04-..." and opens the 3D tab
        │
        ▼
VolumeViewer3D effect fires
        │  fetchVolumeData(volume="raw",  channel=0, maxSize=128)
        │  fetchVolumeData(volume="predicted", predictionSet=<sel>, maxSize=128)
        ▼
FastAPI  /volume.bin
        │  raw:       zarr_reader.volume_to_bytes
        │               → downsample_max_pool_zarr (streamed)
        │               → contrast stretch → uint8 bytes (components=1)
        │  predicted: biapy_loader.predicted_instances_to_bytes
        │               → load BiaPy TIFF (Z,Y,X labels)
        │               → labels_rgb encode (unique color/label) (components=3)
        ▼
bytes + X-Volume-Shape / X-Original-Shape / X-Downsample-Factor / X-Volume-Components
        │
        ▼
Client: build vtkImageData per layer
        │  spacing from originalShape → correct anisotropic aspect ratio
        │  brightness/contrast remap (remapVolumeUint8)
        ▼
vtkVolumeMapper (MaximumIntensity blend) → WebGL
        │
        ▼
Interactive GPU render:  raw (grayscale) + predicted instances (colored)
   drag / arrow keys → TrackballCamera rotates
```

---

## 6. Design decisions worth remembering

- **Reduce on the server, never in the browser.** Slices, MIPs, and downsampled
  volumes are all computed server-side so the wire payload is small and the
  browser only does display work.
- **Stream, don't slurp.** Both the MIP (`_MIP_Z_BLOCK`) and the downsampler
  (`downsample_max_pool_zarr`) process Zarr in Z-blocks to bound host memory on
  ~GB channels.
- **Max, not mean.** MIP and max-pool downsampling both use *maximum* so thin,
  bright neuron structures survive against dark background.
- **One shared contrast pipeline** (`volume_pipeline.py`) keeps raw / GT /
  predicted looking consistent across 2D and 3D.
- **Deterministic instance colors.** Label IDs → seeded-RNG vivid RGB, so the
  same neuron gets the same color every time and across channels.
- **Metadata in headers, pixels in the body.** `volume.bin` returns a bare binary
  blob plus `X-*` headers, avoiding base64/JSON overhead for the large payload.
- **Caching everywhere.** `@lru_cache` on `slice_to_png`, `mip_to_png`,
  `volume_to_bytes`, and `predicted_instances_to_bytes` makes re-viewing a
  sample/plane instant.

## 7. File map

| File | Role |
|------|------|
| `server/main.py` | FastAPI routes; wires params → services, sets `X-*` headers |
| `server/services/volume_pipeline.py` | contrast stretch, max-pool, label→RGB encode |
| `server/services/zarr_reader.py` | Zarr slice / MIP / downsampled-volume extraction |
| `server/services/biapy_loader.py` | predicted-instance volumes from BiaPy runs |
| `server/services/fisbe_mip.py` | serve/crop pre-shipped FISBe MIP PNGs |
| `client/src/api/client.ts` | URL builders + typed fetch (`fetchVolumeData`) |
| `client/src/components/SliceImage.tsx` | blob-fetch a PNG into `<img>` + CSS filter |
| `client/src/components/OrthoSliceViewer.tsx` | 2D slice/MIP tab with GT overlay |
| `client/src/components/VolumeViewer3D.tsx` | vtk.js GPU volume renderer |
| `client/src/utils/displayAdjust.ts` | brightness/contrast/gamma remap (JS + CSS) |
| `client/src/App.tsx` | tabs, sample/prediction-set selection, layout |
</content>
</invoke>
