#!/usr/bin/env python3
"""Build fisbe/himanshu/test/*.zarr from Gen1 MCFO aligned stacks for PPP inference.

Reads downloads/hemidrivers.tsv, resolves Gen1 MCFO folders, converts every
aligned *.h5j (brain + VNC, all templates; excludes unaligned) to FISBe-layout
zarrs with exactly 3 channels and volumes/raw + volumes/raw_normalized.

PPP wiring (inference-only; no train/val needed)::

    fisbe_root = "/scratch/wmz2007/neuroinfo_fruitfly/fisbe"
    test_datasets = ["himanshu"]

Requires ``h5py``, ``zarr``, and ``av`` (PyAV). The ``ppp`` overlay may lack
``av``; install once into a writable overlay or conda env::

    pip install "av>=10"

Example (repo root)::

    python data_download_methods/prepare_himanshu_for_ppp.py \\
      --tsv downloads/hemidrivers.tsv \\
      --downloads "downloads/Gen1 MCFO" \\
      --out fisbe/himanshu/test \\
      --limit 1
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

# Line-buffer stdout so nohup/log redirects show progress immediately.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# emlabel.nb.read_h5j without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_EMLABEL_ROOT = _REPO_ROOT / "lm2lineage" / "emlabel"
if str(_EMLABEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EMLABEL_ROOT))

from emlabel.nb import read_h5j  # noqa: E402

import zarr  # noqa: E402

SPATIAL_CHUNKS = (64, 64, 64)
NUM_CHANNELS = 3


def _compressor():
    try:
        from numcodecs import Blosc

        return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    except ImportError:
        return None


def hemidriver_code(raw: str) -> str | None:
    """Extract enhancer code from an AD/DBD metadata string."""
    raw = (raw or "").strip()
    if not raw:
        return None
    for part in re.split(r"[;]", raw):
        part = part.strip()
        m = re.match(r"^(VT\d+|\d+[A-Z]\d+)\s*-", part, re.I)
        if m:
            return m.group(1)
    head = raw.split("-", 1)[0].strip()
    # Reject bare tokens that are not VT / GMR-style codes (e.g. "dsx").
    if head and re.match(r"^(VT\d+|\d+[A-Z]\d+)$", head, re.I):
        return head
    return None


def to_gen1_folder(code: str) -> str:
    code = code.strip()
    if code.upper().startswith("VT"):
        m = re.match(r"^(VT)(\d+)$", code, re.I)
        if m:
            return f"VT{m.group(2)}"
        return code
    if code.upper().startswith("R") and re.match(r"^R\d+[A-Z]\d+$", code, re.I):
        return "R" + code[1:]
    return f"R{code}"


def folders_from_tsv(tsv_path: Path) -> tuple[list[str], list[str]]:
    """Return (unique Gen1 folders, skipped raw strings)."""
    folders: list[str] = []
    seen: set[str] = set()
    skipped: list[str] = []

    with tsv_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("ss\t") or line.lower() == "ss\trole\traw":
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            raw = parts[2]
            code = hemidriver_code(raw)
            if not code:
                skipped.append(raw)
                continue
            folder = to_gen1_folder(code)
            if folder not in seen:
                seen.add(folder)
                folders.append(folder)
    return folders, skipped


def is_aligned_stack(path: Path) -> bool:
    name = path.name
    return name.endswith("aligned_stack.h5j") and "unaligned" not in name


def collect_h5j(downloads: Path, folders: list[str]) -> tuple[list[Path], list[str]]:
    """Return (h5j paths, missing folder names)."""
    paths: list[Path] = []
    missing: list[str] = []
    for folder in folders:
        d = downloads / folder
        if not d.is_dir():
            missing.append(folder)
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and is_aligned_stack(p):
                paths.append(p)
    return paths, missing


def stem_from_h5j(path: Path) -> str:
    name = path.name
    suffix = "-aligned_stack.h5j"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def force_three_channels(raw: np.ndarray) -> tuple[np.ndarray, int]:
    """Force (C,Z,Y,X) to C=3. Returns (array, original_C)."""
    if raw.ndim != 4:
        raise ValueError(f"expected CZYX, got shape {raw.shape}")
    c = int(raw.shape[0])
    if c == NUM_CHANNELS:
        return raw.astype(np.uint16, copy=False), c
    if c > NUM_CHANNELS:
        return raw[:NUM_CHANNELS].astype(np.uint16, copy=False), c
    out = np.zeros((NUM_CHANNELS,) + raw.shape[1:], dtype=np.uint16)
    out[:c] = raw
    return out, c


def write_zarr(
    out_path: Path,
    raw: np.ndarray,
    clipmax: int,
    overwrite: bool,
    *,
    voxel_size_um: np.ndarray | list[float] | tuple[float, ...],
    alignment_space: str = "",
) -> str:
    """Write volumes/raw and volumes/raw_normalized. Returns status.

    Root attrs: ``voxel_size_um`` as [vx, vy, vz] microns (X,Y,Z order from
    h5j) and ``alignment_space`` when provided.
    """
    if out_path.exists() and not overwrite:
        return "skip_exists"

    if out_path.exists() and overwrite:
        import shutil

        shutil.rmtree(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    compressor = _compressor()
    root = zarr.open(str(out_path), mode="w")
    vox = [float(v) for v in np.asarray(voxel_size_um, dtype=float).ravel()[:3]]
    if len(vox) != 3:
        raise ValueError(f"voxel_size_um must have 3 values, got {vox}")
    root.attrs["voxel_size_um"] = vox
    if alignment_space:
        root.attrs["alignment_space"] = alignment_space
    volumes = root.create_group("volumes")

    chunks = (1,) + SPATIAL_CHUNKS
    kw = {"chunks": chunks}
    if compressor is not None:
        kw["compressor"] = compressor

    volumes.create_dataset("raw", data=raw, dtype="u2", **kw)

    raw_normalized = np.clip(raw, 0, clipmax).astype(np.float32) / float(clipmax)
    volumes.create_dataset("raw_normalized", data=raw_normalized, dtype="f4", **kw)
    return "ok"


def process_one(
    h5j: Path,
    out_dir: Path,
    clipmax: int,
    overwrite: bool,
) -> dict:
    stem = stem_from_h5j(h5j)
    out_path = out_dir / f"{stem}.zarr"
    row = {
        "stem": stem,
        "source_h5j": str(h5j),
        "shape": "",
        "alignment_space": "",
        "n_signal_channels": "",
        "status": "",
    }

    if out_path.exists() and not overwrite:
        row["status"] = "skip_exists"
        return row

    try:
        with open(h5j, "rb") as f:
            raw, kinds, vox, space = read_h5j(f, with_reference=False)
        n_sig = len(kinds)
        if n_sig != NUM_CHANNELS:
            print(
                f"  warn: {h5j.name}: {n_sig} signal channel(s) "
                f"{kinds}; forcing C={NUM_CHANNELS}",
                flush=True,
            )
        raw3, orig_c = force_three_channels(raw)
        status = write_zarr(
            out_path,
            raw3,
            clipmax=clipmax,
            overwrite=overwrite,
            voxel_size_um=vox,
            alignment_space=space or "",
        )
        row["shape"] = str(tuple(raw3.shape))
        row["alignment_space"] = space or ""
        row["n_signal_channels"] = str(orig_c)
        row["status"] = status
    except Exception as e:
        row["status"] = f"error:{e}"
        print(f"  error: {h5j.name}: {e}", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Gen1 MCFO aligned h5j stacks (from hemidrivers.tsv) "
            "to fisbe/himanshu/test/*.zarr for PPP inference."
        )
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=_REPO_ROOT / "downloads" / "hemidrivers.tsv",
        help="Hemidrivers TSV (SS, role, raw)",
    )
    parser.add_argument(
        "--downloads",
        type=Path,
        default=_REPO_ROOT / "downloads" / "Gen1 MCFO",
        help="Root of Gen1 MCFO line folders",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "fisbe" / "himanshu" / "test",
        help="Output directory for *.zarr",
    )
    parser.add_argument("--clipmax", type=int, default=1500)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing .zarr outputs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N stacks (smoke test)",
    )
    args = parser.parse_args()

    try:
        import av  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Missing dependency 'av' (PyAV), required to decode .h5j stacks.\n"
            "  Install with:  pip install 'av>=10'\n"
            f"  Import error: {e}"
        ) from e

    if not args.tsv.is_file():
        raise SystemExit(f"TSV not found: {args.tsv}")
    if not args.downloads.is_dir():
        raise SystemExit(f"Downloads root not found: {args.downloads}")

    folders, skipped = folders_from_tsv(args.tsv)
    print(f"Hemidriver folders: {len(folders)}", flush=True)
    for f in folders:
        print(f"  - {f}", flush=True)
    if skipped:
        print(f"Skipped unparseable raw value(s): {len(skipped)}", flush=True)
        for s in skipped:
            print(f"  - {s!r}", flush=True)

    h5js, missing = collect_h5j(args.downloads, folders)
    if missing:
        print(f"Missing Gen1 folders: {len(missing)}", flush=True)
        for m in missing:
            print(f"  - {m}", flush=True)

    print(f"Aligned h5j stacks found: {len(h5js)}", flush=True)
    if args.limit is not None:
        h5js = h5js[: max(0, args.limit)]
        print(f"Limited to first {len(h5js)} stack(s)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # Manifest rows for missing folders (no h5j to process).
    for m in missing:
        rows.append(
            {
                "stem": "",
                "source_h5j": str(args.downloads / m),
                "shape": "",
                "alignment_space": "",
                "n_signal_channels": "",
                "status": "missing_folder",
            }
        )

    for i, h5j in enumerate(h5js, 1):
        print(f"[{i}/{len(h5js)}] {h5j.name}", flush=True)
        rows.append(
            process_one(
                h5j,
                out_dir=args.out,
                clipmax=args.clipmax,
                overwrite=args.overwrite,
            )
        )
        status = rows[-1]["status"]
        print(f"  → {status}  {rows[-1]['stem']}.zarr", flush=True)

    manifest_path = args.out / "manifest.csv"
    fieldnames = [
        "stem",
        "source_h5j",
        "shape",
        "alignment_space",
        "n_signal_channels",
        "status",
    ]
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    skip = sum(1 for r in rows if r["status"] == "skip_exists")
    miss = sum(1 for r in rows if r["status"] == "missing_folder")
    err = sum(1 for r in rows if str(r["status"]).startswith("error:"))
    print("", flush=True)
    print(f"Done. ok={ok} skip_exists={skip} missing_folder={miss} error={err}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Output:   {args.out}", flush=True)


if __name__ == "__main__":
    main()
