"""FastAPI server for FISBe 3D volume visualization."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from config import CORS_ORIGINS, FISBE_ROOT
from services.biapy_loader import (
    get_predicted_instances_meta,
    has_predicted_instances_any,
    list_prediction_sets,
    predicted_instances_to_bytes,
)
from services.sample_list import SampleEntry, find_sample, parse_sample_list, sample_zarr_path
from services.zarr_reader import (
    AxisKind,
    VolumeKind,
    get_volume_meta,
    mip_to_png,
    slice_to_png,
    volume_to_bytes,
)

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
    """List BiaPy prediction sets (run dirs) the viewer can overlay."""
    return list_prediction_sets()


@app.get("/api/samples")
def list_samples():
    entries = _cached_samples()
    return [
        {
            "split": e.split,
            "name": e.name,
            "dataset": e.dataset,
            "path_exists": e.path_exists,
            "has_predicted": has_predicted_instances_any(e.name),
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
    return entry, zarr_path


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


@app.get("/api/samples/{name}/volume.bin")
async def sample_volume(
    name: str,
    volume: VolumeKind = Query("raw"),
    channel: str = Query("0"),
    max_size: int = Query(256, ge=64, le=512),
    prediction_set: str | None = Query(None),
):
    try:
        if volume == "predicted":
            result = await asyncio.to_thread(
                predicted_instances_to_bytes,
                name,
                max_size,
                prediction_set,
            )
        else:
            _, zarr_path = _resolve_zarr_or_404(name)
            result = await asyncio.to_thread(
                volume_to_bytes,
                str(zarr_path),
                volume,
                _validate_channel_param(channel),
                max_size,
            )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    z, y, x = result.shape
    oz, oy, ox = result.original_shape
    headers = {
        "X-Volume-Shape": f"{z},{y},{x}",
        "X-Original-Shape": f"{oz},{oy},{ox}",
        "X-Downsample-Factor": str(result.downsample_factor),
        "X-Volume-Components": str(result.components),
    }
    return Response(
        content=result.data,
        media_type="application/octet-stream",
        headers=headers,
    )
