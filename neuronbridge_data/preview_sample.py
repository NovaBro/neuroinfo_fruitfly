#!/usr/bin/env python3
"""Headless MIP preview for a neuron-only binary seg.npy (no popup)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with:\n"
        "  pip install numpy matplotlib\n"
        f"Original error: {exc}"
    ) from exc

_PACKAGE_DIR = Path(__file__).resolve().parent
_SAMPLES_ROOT = _PACKAGE_DIR / "samples"


def resolve_sample_dir(args: argparse.Namespace) -> Path:
    if args.sample_dir is not None:
        return Path(args.sample_dir)
    if args.dataset is None or args.body_id is None:
        raise SystemExit("Provide --sample-dir, or both --dataset and --body-id.")
    return _SAMPLES_ROOT / args.dataset / str(args.body_id)


def load_meta(sample_dir: Path) -> dict:
    meta_path = sample_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text())


def make_preview(seg: np.ndarray, axis: int, title: str, out_path: Path) -> None:
    if seg.ndim != 3:
        raise SystemExit(f"Expected 3D seg array; got shape={seg.shape}")
    mip = (seg > 0).max(axis=axis).astype(np.float32)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mip.T, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Save a headless MIP PNG of neuron-only binary seg.npy. No popup."
        )
    )
    p.add_argument("--sample-dir", type=Path, default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--body-id", type=int, default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: <sample-dir>/preview_seg.png)",
    )
    p.add_argument(
        "--axis",
        type=int,
        default=2,
        choices=(0, 1, 2),
        help="Axis to collapse for MIP (default: 2 = XY along Z)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sample_dir = resolve_sample_dir(args)
    seg_path = sample_dir / "seg.npy"
    if not seg_path.is_file():
        raise SystemExit(
            f"Missing seg.npy in {sample_dir}. Re-run download for this body first."
        )
    seg = np.load(seg_path)
    meta = load_meta(sample_dir)
    body = meta.get("bodyId", sample_dir.name)
    n_vox = int(np.count_nonzero(seg))
    title = f"bodyId={body}  shape={tuple(seg.shape)}  n_voxels={n_vox}  axis={args.axis}"
    out_path = args.out if args.out is not None else sample_dir / "preview_seg.png"
    make_preview(seg, args.axis, title, out_path)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
