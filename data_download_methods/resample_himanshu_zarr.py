#!/usr/bin/env python3
"""Resample prepared himanshu zarrs (63x MCFO) to the FISBe voxel size for PPP inference.

``prepare_himanshu_for_ppp.py`` keeps the native 63x sampling (0.19 x 0.19 x 0.38 um).
PPP checkpoints are trained on FISBe at 0.44 um isotropic (FISBe paper, sec. 2.1), so
neurites at native resolution look ~2.3x thicker in XY than anything the model saw.
This script reads ``<src>/*.zarr/volumes/raw`` plus the root attr ``voxel_size_um``
([vx, vy, vz], microns) and writes NEW zarrs ``<dst>/<same stem>.zarr`` resampled to
``--target-um`` (default 0.44) on every axis. The source zarrs are only read.
Existing outputs are never touched (skipped); delete them by hand to redo one.

* per channel: ``skimage.transform.resize(order=1, anti_aliasing=True,
  preserve_range=True)`` in float32 (Gaussian pre-smoothing, then linear zoom),
  rounded back to uint16;
* ``volumes/raw`` + ``volumes/raw_normalized`` are written with
  ``prepare_himanshu_for_ppp.write_zarr`` (same chunks, compressor, clipmax
  normalisation and raw_normalized dtype as the native prep);
* root attrs: the source attrs, ``voxel_size_um = [t, t, t]``,
  ``effective_voxel_size_um`` (exact, after rounding the output shape),
  ``source_voxel_size_um``, ``source_shape``, ``resampled_from``, ``resample_method``;
* ``<dst>/manifest.csv``: one row per volume (shapes, voxel sizes, status).

Example (repo root, inside the ppp container)::

    python data_download_methods/resample_himanshu_zarr.py \\
        --src fisbe/himanshu/test --dst fisbe/himanshu_r044/test --target-um 0.44

sbatch wrapper: ``sbatch/ppp/prep/himanshu/resample_himanshu_sbatch.sh``.
The output dataset name (``himanshu_r044``) must be in ``KNOWN_DATASETS``
(PatchPerPix/experiments/run_ppp.py) for list-mode inference.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from prepare_himanshu_for_ppp import write_zarr  # noqa: E402

import zarr  # noqa: E402
from skimage.transform import resize  # noqa: E402

RESAMPLE_METHOD = "skimage.transform.resize(order=1, anti_aliasing=True, preserve_range=True), float32 -> rint uint16"

MANIFEST_FIELDS = [
    "stem", "source_zarr", "source_shape", "shape",
    "source_voxel_size_um", "voxel_size_um", "effective_voxel_size_um", "status",
]


def zoom_factors(voxel_size_um_xyz, target_um: float) -> tuple[float, float, float]:
    """[vx, vy, vz] (X, Y, Z order, microns) -> (z, y, x) zoom factors to reach target_um."""
    vx, vy, vz = (float(v) for v in voxel_size_um_xyz)
    return (vz / target_um, vy / target_um, vx / target_um)


def output_shape(shape_zyx, factors_zyx) -> tuple[int, int, int]:
    return tuple(max(1, int(round(s * f))) for s, f in zip(shape_zyx, factors_zyx))


def resample_channel(chan: np.ndarray, out_shape) -> np.ndarray:
    """One (Z, Y, X) uint16 channel -> uint16 at out_shape (anti-aliased linear)."""
    out = resize(
        chan.astype(np.float32, copy=False),
        out_shape,
        order=1,
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.clip(np.rint(out), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def process_one(src: Path, dst_dir: Path, target_um: float, clipmax: int, raw_dtype: str) -> dict:
    stem = src.name[: -len(".zarr")]
    out_path = dst_dir / src.name
    row = {k: "" for k in MANIFEST_FIELDS}
    row.update(stem=stem, source_zarr=str(src))
    if out_path.exists():
        row["status"] = "skip_exists"
        return row
    try:
        root = zarr.open(str(src), mode="r")
        src_attrs = dict(root.attrs)
        vox = src_attrs.get("voxel_size_um")
        if not vox or len(vox) != 3:
            raise ValueError(f"missing/invalid voxel_size_um attr: {vox!r}")
        raw_arr = root["volumes/raw"]
        c, *zyx = raw_arr.shape
        factors = zoom_factors(vox, target_um)
        out_zyx = output_shape(zyx, factors)
        eff_xyz = [zyx[2] * float(vox[0]) / out_zyx[2],
                   zyx[1] * float(vox[1]) / out_zyx[1],
                   zyx[0] * float(vox[2]) / out_zyx[0]]
        out = np.zeros((c,) + out_zyx, dtype=np.uint16)
        for ci in range(c):
            t0 = time.time()
            out[ci] = resample_channel(raw_arr[ci], out_zyx)
            print(f"  c{ci}: {tuple(zyx)} -> {out_zyx} ({time.time() - t0:.0f} s)", flush=True)
        attrs = {k: v for k, v in src_attrs.items() if k != "voxel_size_um"}
        attrs.update(
            source_voxel_size_um=[float(v) for v in vox],
            effective_voxel_size_um=[round(v, 6) for v in eff_xyz],
            source_shape=[int(c)] + [int(s) for s in zyx],
            resampled_from=str(src),
            resample_method=RESAMPLE_METHOD,
        )
        status = write_zarr(
            out_path, out, clipmax=clipmax, overwrite=False,
            voxel_size_um=[target_um] * 3, attrs=attrs, raw_dtype=raw_dtype,
        )
        row.update(
            source_shape=str((c, *zyx)),
            shape=str(tuple(out.shape)),
            source_voxel_size_um="x".join(f"{float(v):g}" for v in vox),
            voxel_size_um="x".join(f"{target_um:g}" for _ in range(3)),
            effective_voxel_size_um="x".join(f"{v:.4f}" for v in eff_xyz),
            status=status,
        )
    except Exception as e:  # keep going; the manifest records the failure
        row["status"] = f"error:{e}"
        print(f"  error: {src.name}: {e}", flush=True)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", type=Path, default=Path("fisbe/himanshu/test"))
    p.add_argument("--dst", type=Path, default=Path("fisbe/himanshu_r044/test"))
    p.add_argument("--target-um", type=float, default=0.44)
    p.add_argument("--clipmax", type=int, default=1500)
    p.add_argument("--raw-dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--limit", type=int, default=None, help="process only the first N volumes (sorted)")
    args = p.parse_args()

    if args.src.resolve() == args.dst.resolve():
        sys.exit("--dst must differ from --src")
    srcs = sorted(args.src.glob("*.zarr"))
    if args.limit is not None:
        srcs = srcs[: args.limit]
    if not srcs:
        sys.exit(f"no .zarr under {args.src}")
    args.dst.mkdir(parents=True, exist_ok=True)
    print(f"src={args.src} dst={args.dst} target_um={args.target_um} clipmax={args.clipmax} "
          f"raw_dtype={args.raw_dtype} n={len(srcs)}", flush=True)

    manifest = args.dst / "manifest.csv"
    old = {}
    if manifest.exists():
        with manifest.open(newline="") as f:
            old = {r["stem"]: r for r in csv.DictReader(f)}

    rows = []
    for i, src in enumerate(srcs, 1):
        print(f"[{i}/{len(srcs)}] {src.name}", flush=True)
        row = process_one(src, args.dst, args.target_um, args.clipmax, args.raw_dtype)
        if row["status"] == "skip_exists" and row["stem"] in old:
            row = old[row["stem"]]
        print(f"  -> {row['status']} {row.get('shape', '')}", flush=True)
        rows.append(row)

    done = {r["stem"] for r in rows}
    rows += [r for s, r in old.items() if s not in done]
    rows.sort(key=lambda r: r["stem"])
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(r["status"] == "ok" for r in rows)
    n_err = sum(r["status"].startswith("error") for r in rows)
    print(f"Done. ok={n_ok} errors={n_err} manifest={manifest}", flush=True)
    if n_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
