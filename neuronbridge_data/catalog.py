"""neuPrint catalog: client, neuron listing, skeleton bounding boxes."""

from __future__ import annotations

from typing import Any

import numpy as np
from cloudvolume import CloudVolume
from neuprint import Client, NeuronCriteria, fetch_neurons

from neuronbridge_data.auth import resolve_token
from neuronbridge_data.datasets import DATASETS, NEUPRINT_SERVER


def make_client(dataset_key: str, token: str | None) -> Client:
    cfg = DATASETS[dataset_key]
    return Client(
        NEUPRINT_SERVER,
        dataset=cfg["neuprint"],
        token=resolve_token(token),
    )


def query_neurons(
    client: Client,
    type_pattern: str | None,
    limit: int,
) -> Any:
    if type_pattern:
        criteria = NeuronCriteria(type=type_pattern, regex=True)
    else:
        criteria = NeuronCriteria(status="Traced")
    neuron_df, _roi_df = fetch_neurons(criteria, client=client)
    if neuron_df is None or len(neuron_df) == 0:
        return neuron_df
    return neuron_df.head(limit)


def print_neuron_table(neuron_df: Any) -> None:
    if neuron_df is None or len(neuron_df) == 0:
        print("No neurons matched.")
        return
    cols = [
        c
        for c in ("bodyId", "type", "instance", "status", "pre", "post")
        if c in neuron_df.columns
    ]
    print(neuron_df[cols].to_string(index=False))
    print(f"\n{len(neuron_df)} neuron(s)")


def skeleton_points_mip0(client: Client, body_id: int) -> np.ndarray:
    """Return Nx3 mip0 voxel coordinates from the neuPrint skeleton."""
    skel = client.fetch_skeleton(body_id, format="pandas")
    if skel is None or len(skel) == 0:
        raise RuntimeError(f"No skeleton for bodyId={body_id}")
    return skel[["x", "y", "z"]].to_numpy(dtype=np.float64)


def skeleton_bbox_mip0(
    client: Client,
    body_id: int,
    vol: CloudVolume,
    pad: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return inclusive-exclusive mip0 voxel bbox (lo, hi) from skeleton + pad.

    neuPrint hemibrain/MANC/MaleCNS skeletons expose x/y/z in mip0 voxel units
    (not nanometers), matching CloudVolume index space at mip 0.
    """
    xyz_vx = skeleton_points_mip0(client, body_id)
    lo = np.floor(xyz_vx.min(axis=0)).astype(np.int64) - pad
    hi = np.ceil(xyz_vx.max(axis=0)).astype(np.int64) + pad + 1

    bmin = vol.meta.bounds(0).minpt
    bmax = vol.meta.bounds(0).maxpt
    lo = np.maximum(lo, [bmin.x, bmin.y, bmin.z])
    hi = np.minimum(hi, [bmax.x, bmax.y, bmax.z])
    if np.any(hi <= lo):
        raise RuntimeError(
            f"Empty bbox after clamp for bodyId={body_id}: lo={lo} hi={hi}"
        )
    return (
        (int(lo[0]), int(lo[1]), int(lo[2])),
        (int(hi[0]), int(hi[1]), int(hi[2])),
    )
