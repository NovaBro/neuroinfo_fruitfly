#!/usr/bin/env python3
"""Convert MCFO .h5j stacks from a download_mcfo.py manifest to PPP inference zarrs.

Takes every ``status == selected`` row of ``--manifest`` and writes
``<out>/<stem>.zarr`` with ``volumes/raw`` (u2, exactly 3 channels, CZYX) and
``volumes/raw_normalized`` (clip(raw, 0, clipmax) / clipmax). The native voxel grid is
kept (no resampling); its size goes to the root attr ``voxel_size_um`` [vx, vy, vz]
(from the h5j ``voxel_size`` attr, else the manifest's metadata ``voxel_size``).
Optional resampling to the FISBe grid is a separate step (resample_himanshu_zarr.py).

``--raw-dtype float16`` (default) is lossless for the clip levels: every k/1500 stays
distinct and round(x * clipmax) recovers k. PPP predict casts raw to float32.

Needs h5py, zarr and PyAV (``av``; sbatch/ppp/env/ppp_env_install_av_sbatch.sh puts it
in the ppp overlay). Usually run via sbatch/ppp/prep/himanshu/himanshu_sbatch.sh;
by hand (repo root, ppp container)::

    python data_download_methods/prepare_himanshu_for_ppp.py \\
        --manifest downloads/mcfo_manifest_20x.tsv --out fisbe/himanshu_20x/test --limit 1
"""
from __future__ import annotations

import argparse
import csv
import io
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import zarr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

_REPO_ROOT = Path(__file__).resolve().parents[1]
SPATIAL_CHUNKS = (64, 64, 64)
NUM_CHANNELS = 3
MANIFEST_FIELDS = ["stem", "ss", "image_id", "stack", "source_h5j", "shape", "alignment_space",
                   "voxel_size_um", "voxel_size_source", "n_signal_channels", "status"]


def _str(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def read_h5j(path: Path):
    """Decode a Janelia .h5j (HDF5, one H.265 stream per channel), reference channels dropped.

    Returns (raw CZYX uint16, channel kinds, voxel_size [x, y, z] or None, alignment_space).
    """
    import av

    with h5py.File(path, "r") as f:
        ch = f["Channels"]
        h, w, n = (int(ch.attrs[k][0]) for k in ("height", "width", "frames"))
        names = sorted(k for k in ch.keys() if k.startswith("Channel_"))
        kinds = [_str(ch[k].attrs.get("content_type", b"signal")) for k in names]
        keep = [i for i, kind in enumerate(kinds) if kind != "reference"]
        raw = np.zeros((len(keep), n, h, w), np.uint16)
        for j, i in enumerate(keep):
            c = av.open(io.BytesIO(ch[names[i]][()].tobytes()))
            try:
                for z, frame in enumerate(c.decode(video=0)):
                    if z >= n:
                        break
                    a = frame.to_ndarray()
                    raw[j, z] = (a if a.ndim == 2 else a[..., 0])[:h, :w]
            finally:
                c.close()
        vox = f.attrs.get("voxel_size")
        vox = [float(v) for v in np.ravel(vox)[:3]] if vox is not None else None
        space = _str(f.attrs.get("alignment_space", b""))
    return raw, [kinds[i] for i in keep], vox, space


def read_manifest(path: Path) -> list[dict]:
    """Rows of a download_mcfo.py manifest (``#`` comment lines skipped)."""
    with path.open(newline="") as f:
        return list(csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t"))


def parse_voxel_size(text: str) -> list[float] | None:
    """``"0.52x0.52x1.00"`` (X x Y x Z, microns) -> [0.52, 0.52, 1.0]."""
    try:
        vals = [float(v) for v in (text or "").lower().split("x")]
    except ValueError:
        return None
    return vals if len(vals) == 3 and all(v > 0 for v in vals) else None


def stem_from_h5j(name: str) -> str:
    """``…-unaligned_stack.h5j`` -> ``…-unaligned``; ``…-aligned_stack.h5j`` -> ``…``."""
    for suffix, repl in (("-unaligned_stack.h5j", "-unaligned"), ("-aligned_stack.h5j", "")):
        if name.endswith(suffix):
            return name[: -len(suffix)] + repl
    return Path(name).stem


def force_three_channels(raw: np.ndarray) -> np.ndarray:
    """(C,Z,Y,X) -> C=3: extra channels dropped, missing ones zero-filled."""
    if raw.shape[0] >= NUM_CHANNELS:
        return raw[:NUM_CHANNELS]
    out = np.zeros((NUM_CHANNELS,) + raw.shape[1:], dtype=raw.dtype)
    out[: raw.shape[0]] = raw
    return out


def write_zarr(out_path: Path, raw: np.ndarray, clipmax: int, overwrite: bool, *,
               voxel_size_um, attrs: dict | None = None, raw_dtype: str = "float16") -> str:
    """Write volumes/raw + volumes/raw_normalized and root attrs. Returns a status string.

    Root attrs: ``voxel_size_um`` [vx, vy, vz] microns plus non-empty ``attrs``.
    Also used by resample_himanshu_zarr.py.
    """
    if out_path.exists():
        if not overwrite:
            return "skip_exists"
        shutil.rmtree(out_path)
    vox = [float(v) for v in np.ravel(voxel_size_um)]
    if len(vox) != 3:
        raise ValueError(f"voxel_size_um must have 3 values, got {vox}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(out_path), mode="w")
    root.attrs["voxel_size_um"] = vox
    for k, v in (attrs or {}).items():
        if v not in ("", None):
            root.attrs[k] = v
    from numcodecs import Blosc

    kw = {"chunks": (1,) + SPATIAL_CHUNKS,
          "compressor": Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)}
    volumes = root.create_group("volumes")
    volumes.create_dataset("raw", data=raw, dtype="u2", **kw)
    norm = (np.clip(raw, 0, clipmax).astype(np.float32) / float(clipmax)).astype(raw_dtype)
    volumes.create_dataset("raw_normalized", data=norm, dtype=raw_dtype, **kw)
    return "ok"


def process_one(mrow: dict, downloads: Path, out_dir: Path, clipmax: int, overwrite: bool,
                raw_dtype: str) -> dict:
    keys = [k for k in (mrow.get("s3_keys") or "").split(";") if k]
    row = {"ss": mrow.get("ss", ""), "image_id": mrow.get("image_id", ""),
           "stack": mrow.get("stack", "")}
    if len(keys) != 1 or not keys[0].endswith(".h5j"):
        row["status"] = f"error:expected 1 .h5j key, got {keys}"
        return row
    h5j = downloads / keys[0]
    stem = stem_from_h5j(h5j.name)
    out_path = out_dir / f"{stem}.zarr"
    row.update(stem=stem, source_h5j=str(h5j))
    if not h5j.is_file():
        row["status"] = "missing_file"
        return row
    if out_path.exists() and not overwrite:
        row["status"] = "skip_exists"
        return row
    try:
        raw, kinds, vox, space = read_h5j(h5j)
        vox_source = "h5j"
        if vox is None:
            vox, vox_source = parse_voxel_size(mrow.get("voxel_size", "")), "metadata"
        if vox is None:
            raise ValueError("no voxel size in h5j attrs or manifest")
        if len(kinds) != NUM_CHANNELS:
            print(f"  warn: {h5j.name}: {len(kinds)} signal channel(s) {kinds}; "
                  f"forcing C={NUM_CHANNELS}", flush=True)
        raw3 = force_three_channels(raw)
        row["status"] = write_zarr(
            out_path, raw3, clipmax, overwrite, voxel_size_um=vox, raw_dtype=raw_dtype,
            attrs={"alignment_space": space, "publishing_name": mrow.get("ss", ""),
                   "image_id": mrow.get("image_id", ""), "release": mrow.get("release", ""),
                   "slide_code": mrow.get("slide_code", ""), "objective": mrow.get("objective", ""),
                   "area": mrow.get("area", ""), "stack": mrow.get("stack", ""),
                   "source_s3_key": keys[0], "voxel_size_source": vox_source})
        row.update(shape=str(tuple(raw3.shape)), alignment_space=space,
                   voxel_size_um="x".join(f"{v:g}" for v in vox),
                   voxel_size_source=vox_source, n_signal_channels=str(len(kinds)))
    except Exception as e:  # keep going; manifest.csv records the failure
        row["status"] = f"error:{e}"
        print(f"  error: {h5j.name}: {e}", flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=_REPO_ROOT / "downloads" / "mcfo_manifest_20x.tsv")
    ap.add_argument("--downloads", type=Path, default=_REPO_ROOT / "downloads",
                    help="root the manifest's s3_keys are relative to (download_mcfo.py --out)")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "fisbe" / "himanshu_20x" / "test")
    ap.add_argument("--clipmax", type=int, default=1500)
    ap.add_argument("--raw-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--overwrite", action="store_true", help="replace existing .zarr outputs")
    ap.add_argument("--clean", action="store_true",
                    help="first delete every *.zarr and manifest.csv in --out (also ones from other manifests)")
    ap.add_argument("--limit", type=int, default=None, help="first N stacks only (smoke test)")
    args = ap.parse_args()

    rows_in = read_manifest(args.manifest)
    selected = [r for r in rows_in if r.get("status") == "selected"]
    print(f"Manifest: {args.manifest}  selected={len(selected)}  out={args.out}  "
          f"raw_dtype={args.raw_dtype}", flush=True)
    for r in rows_in:
        if r.get("status") != "selected":
            print(f"  skipped by download: {r.get('ss')}: {r.get('status')} ({r.get('note', '')})")

    if args.clean and args.out.is_dir():
        old = sorted(args.out.glob("*.zarr"))
        print(f"--clean: deleting {len(old)} zarr(s) + manifest.csv in {args.out}", flush=True)
        for p in old:
            shutil.rmtree(p)
        (args.out / "manifest.csv").unlink(missing_ok=True)
    expected = {stem_from_h5j(Path(r.get("s3_keys", "")).name) for r in selected}
    stale = sorted(p.name for p in args.out.glob("*.zarr") if p.name[:-5] not in expected)
    if stale:
        print(f"WARNING: {len(stale)} zarr(s) in {args.out} are not in this manifest but PPP will "
              f"still use them (rerun with --clean): {stale[:5]}", flush=True)

    if args.limit is not None:
        selected = selected[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, mrow in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {mrow.get('ss')} {mrow.get('image_id')} "
              f"{mrow.get('objective')} {mrow.get('stack')}", flush=True)
        rows.append(process_one(mrow, args.downloads, args.out, args.clipmax, args.overwrite,
                                args.raw_dtype))
        print(f"  -> {rows[-1]['status']}  {rows[-1].get('shape', '')}  "
              f"{rows[-1].get('voxel_size_um', '')}", flush=True)

    with (args.out / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    status = {}
    for r in rows:
        key = r["status"].split(":", 1)[0]
        status[key] = status.get(key, 0) + 1
    print(f"Done: {status}  manifest: {args.out / 'manifest.csv'}", flush=True)
    if any(r["status"].startswith(("error", "missing")) for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
