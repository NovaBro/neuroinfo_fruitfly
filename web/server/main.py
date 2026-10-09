"""FastAPI server for FISBe 3D volume visualization."""

from __future__ import annotations

import asyncio
import logging
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from config import CORS_ORIGINS, FISBE_ROOT
from services import biapy_loader, ppp_loader
from services.local_stage import ensure_local
from services.predictions import (
    get_predicted_instances_meta,
    list_prediction_sets,
    predicted_instances_to_bytes,
    stems_with_predictions,
)
from services.aggregate_metrics import get_aggregate_metrics
from services.fisbe_mip import fisbe_mip_png
from services.metrics import get_sample_metrics
from services.sample_list import SampleEntry, find_sample, parse_sample_list, sample_zarr_path
from services.zarr_reader import (
    AxisKind,
    VolumeKind,
    get_volume_meta,
    mip_to_png,
    slice_to_png,
    volume_to_bytes,
)

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="FISBe Volume Viewer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Volume-Shape",
        "X-Original-Shape",
        "X-Downsample-Factor",
        "X-Volume-Components",
        "X-Volume-Compute-Ms",
        "X-Volume-Cache",
    ],
)


@lru_cache(maxsize=1)
def _cached_samples() -> tuple[SampleEntry, ...]:
    return tuple(parse_sample_list())


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "fisbe_root": str(FISBE_ROOT),
        "fisbe_root_exists": FISBE_ROOT.is_dir(),
    }


@app.get("/api/prediction-sets")
def prediction_sets():
    """List prediction sets (BiaPy runs + PatchPerPix overlays) to overlay."""
    return list_prediction_sets()


@app.get("/api/samples")
def list_samples(prediction_set: str | None = Query(None)):
    """List samples; ``has_predicted`` is scoped to ``prediction_set`` when set."""
    entries = _cached_samples()
    predicted = stems_with_predictions(prediction_set)
    return [
        {
            "split": e.split,
            "name": e.name,
            "dataset": e.dataset,
            "path_exists": e.path_exists,
            "has_predicted": e.name in predicted,
        }
        for e in entries
    ]


@app.get("/api/samples/{name}/meta")
def sample_meta(name: str, prediction_set: str | None = Query(None)):
    entry = find_sample(name, list(_cached_samples()))
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Sample not found: {name}")

    zarr_path = sample_zarr_path(entry)
    if not zarr_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Zarr data not found on disk: {zarr_path}",
        )

    meta = get_volume_meta(zarr_path)
    predicted = get_predicted_instances_meta(name, prediction_set)
    return {
        "name": name,
        "split": entry.split,
        "dataset": entry.dataset,
        "zarr_path": str(zarr_path),
        "prediction_set": prediction_set,
        "predicted_instances": predicted,
        **meta,
    }


@app.get("/api/samples/{name}/metrics")
def sample_metrics(name: str, prediction_set: str | None = Query(None)):
    """Scoring metrics for ``name`` from the given prediction set.

    Returns raw ``toml`` / ``csv`` plus a curated ``leaderboard`` list using
    FISBe website names (S · avF1 · C · clDiceTP · tp · FS · FM). Either raw
    source may be ``null`` when absent for the sample/set.
    """
    entry = find_sample(name, list(_cached_samples()))
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Sample not found: {name}")
    return {
        "name": name,
        "prediction_set": prediction_set,
        **get_sample_metrics(name, prediction_set),
    }


@app.get("/api/aggregate-metrics")
def aggregate_metrics(
    prediction_set: str | None = Query(None),
    threshold: str | None = Query(None),
):
    """Samples x metrics matrix for the given prediction set.

    Columns are the FISBe website leaderboard metrics (S · avF1 · C ·
    clDiceTP · tp · FS · FM; see ``services.fisbe_metrics``). ``threshold``
    is accepted for API compatibility; current columns are threshold-free.
    """
    return get_aggregate_metrics(prediction_set, threshold)


def _resolve_zarr_or_404(name: str) -> tuple[SampleEntry, Path]:
    entry = find_sample(name, list(_cached_samples()))
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Sample not found: {name}")
    zarr_path = sample_zarr_path(entry)
    if not zarr_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Zarr data not found on disk: {zarr_path}",
        )
    # Copy-on-first-open onto $SLURM_TMPDIR when WEB_STAGE_DIR is configured.
    return entry, ensure_local(zarr_path)


def _validate_channel_param(channel: str) -> str:
    if channel == "all":
        return channel
    if not channel.isdigit():
        raise ValueError(f"channel must be a non-negative integer or 'all', got {channel!r}")
    return str(int(channel))


@app.get("/api/samples/{name}/slice.png")
def sample_slice(
    name: str,
    volume: VolumeKind = Query("raw"),
    channel: str = Query("0"),
    axis: AxisKind = Query("z"),
    index: int = Query(0, ge=0),
):
    _, zarr_path = _resolve_zarr_or_404(name)
    try:
        png = slice_to_png(
            str(zarr_path),
            volume=volume,
            channel=_validate_channel_param(channel),
            axis=axis,
            index=index,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(content=png, media_type="image/png")


@app.get("/api/samples/{name}/mip.png")
def sample_mip(
    name: str,
    volume: VolumeKind = Query("raw"),
    channel: str = Query("0"),
):
    _, zarr_path = _resolve_zarr_or_404(name)
    try:
        png = mip_to_png(
            str(zarr_path),
            volume=volume,
            channel=_validate_channel_param(channel),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(content=png, media_type="image/png")


@app.get("/api/samples/{name}/fisbe_mip.png")
def sample_fisbe_mip(
    name: str,
    half: str = Query("full", pattern="^(full|raw|gt)$"),
):
    """Serve the pre-generated FISBe MIP PNG (raw half, GT half, or both)."""
    entry = find_sample(name, list(_cached_samples()))
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Sample not found: {name}")
    try:
        png = fisbe_mip_png(entry, half=half)  # type: ignore[arg-type]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return Response(content=png, media_type="image/png")


@app.get("/api/samples/{name}/volume.bin")
async def sample_volume(
    name: str,
    volume: VolumeKind = Query("raw"),
    channel: str = Query("0"),
    max_size: int = Query(256, ge=64, le=512),
    prediction_set: str | None = Query(None),
):
    channel_param = channel
    try:
        if volume == "predicted":
            cached_fn = (
                ppp_loader.predicted_instances_to_bytes
                if ppp_loader.is_ppp_set(prediction_set)
                else biapy_loader.predicted_instances_to_bytes
            )
            hits_before = cached_fn.cache_info().hits
            t0 = time.perf_counter()
            result = await asyncio.to_thread(
                predicted_instances_to_bytes,
                name,
                max_size,
                prediction_set,
            )
        else:
            channel_param = _validate_channel_param(channel)
            _, zarr_path = _resolve_zarr_or_404(name)
            cached_fn = volume_to_bytes
            hits_before = cached_fn.cache_info().hits
            t0 = time.perf_counter()
            result = await asyncio.to_thread(
                volume_to_bytes,
                str(zarr_path),
                volume,
                channel_param,
                max_size,
            )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    compute_ms = round((time.perf_counter() - t0) * 1000)
    cache = "hit" if cached_fn.cache_info().hits > hits_before else "miss"

    z, y, x = result.shape
    oz, oy, ox = result.original_shape
    nbytes = len(result.data)
    logger.info(
        "volume.bin name=%s volume=%s channel=%s max_size=%s "
        "cache=%s compute_ms=%s shape=%s,%s,%s bytes=%s",
        name,
        volume,
        channel_param if volume != "predicted" else "-",
        max_size,
        cache,
        compute_ms,
        z,
        y,
        x,
        nbytes,
    )
    headers = {
        "X-Volume-Shape": f"{z},{y},{x}",
        "X-Original-Shape": f"{oz},{oy},{ox}",
        "X-Downsample-Factor": str(result.downsample_factor),
        "X-Volume-Components": str(result.components),
        "X-Volume-Compute-Ms": str(compute_ms),
        "X-Volume-Cache": cache,
    }
    return Response(
        content=result.data,
        media_type="application/octet-stream",
        headers=headers,
    )
