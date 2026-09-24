"""Load PatchPerPix prediction volumes for the web viewer.

Each PatchPerPix experiment under ``PPP_EXPERIMENTS_BASE`` can expose two kinds
of overlay, discovered independently as separate "prediction sets" under both
the ``test/`` and ``val/`` split dirs:

* **numinst** — the per-voxel overlap-count map written by ``predict``:
  ``{split}/processed/<ckpt>/<stem>.zarr`` → ``volumes/pred_numinst``, shape
  ``(3, Z, Y, X)`` float16 (channels = P(0 instances), P(1), P(2+)). This is a
  foreground/count probability map, *not* per-neuron labels, so it renders as a
  two-colour foreground after argmax (1-instance = cyan, 2+ overlap = magenta).
* **instances** — the final vote-instances labels:
  ``{split}/instanced/<ckpt>/<params...>/<stem>.hdf`` → dataset ``vote_instances``,
  a ``(Z, Y, X)`` integer label volume coloured per neuron like the BiaPy/GT
  overlays. Requires ``h5py``; when it is missing, instances sets are omitted
  from discovery and loads raise a clear error.

Set ids are source-prefixed (``ppp-numinst:`` / ``ppp-inst:``) so the dispatcher
in ``services/predictions.py`` can route to the right loader without colliding
with the (bare, relative-path) BiaPy ids.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import zarr

from config import PPP_EXPERIMENTS_BASE
from services.volume_pipeline import (
    VolumeBytesResult,
    compute_downsample_factor,
    volume_array_to_bytes,
)

NUMINST_PREFIX = "ppp-numinst:"
INST_PREFIX = "ppp-inst:"

_NUMINST_KEY = "volumes/pred_numinst"
_VOTE_DATASET = "vote_instances"
_SPLITS = ("test", "val")


# --- discovery -------------------------------------------------------------

def _base() -> Path:
    return PPP_EXPERIMENTS_BASE.resolve()


def _rel(root: Path) -> str:
    return str(root.resolve().relative_to(_base()))


def _numinst_dir_has_data(processed_dir: Path) -> bool:
    return any(
        (p / "volumes" / "pred_numinst" / ".zarray").is_file()
        for p in processed_dir.glob("*.zarr")
    )


def _h5py_available() -> bool:
    try:
        import h5py  # noqa: F401
    except ImportError:
        return False
    return True


_DEFAULT_NUMINST_THRESHS = "numinst_threshs__0_9_0_1_"


def _tf_flag(part: str, prefix: str) -> str | None:
    """Map ``{prefix}_True`` / ``{prefix}_False`` → ``T`` / ``F``."""
    true_s, false_s = f"{prefix}_True", f"{prefix}_False"
    if part == true_s:
        return "T"
    if part == false_s:
        return "F"
    return None


def _format_numinst_name(exp: str, split: str, ckpt_dir: Path) -> str:
    """Human label for a processed/<ckpt> numinst set."""
    return f"{exp} · {split} · ckpt {ckpt_dir.name} · numinst"


def _format_instances_name(
    exp: str, split: str, param_dir: Path, instanced_base: Path
) -> str:
    """Human label encoding ckpt + VI params from the instanced path.

    Example::

        ppp_basic_long8h · test · ckpt 70000 · instances · th0.3 · mws=T · skel=T

    Falls back to the leaf folder name when the path layout is unexpected.
    """
    try:
        parts = param_dir.resolve().relative_to(instanced_base.resolve()).parts
    except ValueError:
        return f"{exp} · {split} · instances ({param_dir.name})"

    if not parts:
        return f"{exp} · {split} · instances ({param_dir.name})"

    bits: list[str] = [exp, split]
    idx = 0
    if parts[0].isdigit():
        bits.append(f"ckpt {parts[0]}")
        idx = 1
    bits.append("instances")

    th: str | None = None
    mws: str | None = None
    skel: str | None = None
    ccs: str | None = None
    ni_label: str | None = None

    for part in parts[idx:]:
        if part.startswith("patch_threshold_"):
            th = part[len("patch_threshold_") :].replace("_", ".", 1)
        elif (flag := _tf_flag(part, "mws")) is not None:
            mws = flag
        elif (flag := _tf_flag(part, "skeletonize_foreground")) is not None:
            skel = flag
        elif (flag := _tf_flag(part, "includeSinglePatchCCS")) is not None:
            ccs = flag
        elif part.startswith("numinst_threshs_") and part != _DEFAULT_NUMINST_THRESHS:
            # numinst_threshs__0_5_0_5_ → ni=0.5/0.5
            raw = part[len("numinst_threshs_") :].strip("_")
            toks = [t for t in raw.split("_") if t]
            paired: list[str] = []
            i = 0
            while i + 1 < len(toks):
                paired.append(f"{toks[i]}.{toks[i + 1]}")
                i += 2
            if paired:
                ni_label = "ni=" + "/".join(paired)
            elif toks:
                ni_label = "ni=" + "/".join(toks)

    if th is None and mws is None and skel is None and ccs is None:
        # Unrecognised layout under instanced/ — keep old leaf label.
        return f"{exp} · {split} · instances ({param_dir.name})"

    if th is not None:
        bits.append(f"th{th}")
    if mws is not None:
        bits.append(f"mws={mws}")
    if skel is not None:
        bits.append(f"skel={skel}")
    if ccs is not None:
        bits.append(f"CCS={ccs}")
    if ni_label is not None:
        bits.append(ni_label)

    return " · ".join(bits)


def discover_prediction_sets() -> list[dict]:
    """Discover every PatchPerPix numinst and instances set under the base.

    Each entry is ``{"id", "name", "path", "default", "source", "kind"}``.
    ``default`` is always False — the viewer defaults to a BiaPy set and the
    user opts into a PatchPerPix overlay via the dropdown.

    Instances sets are omitted when ``h5py`` is not installable in the server
    env (the HDF loader would fail at request time otherwise).

    Display ``name`` encodes ckpt and VI params (e.g.
    ``… · ckpt 70000 · instances · th0.3 · mws=T · skel=T``); set ``id`` remains
    the full relative path so selections stay stable across renames.
    """
    base = _base()
    sets: list[dict] = []
    if not base.is_dir():
        return sets

    can_read_hdf = _h5py_available()

    for exp_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        exp = exp_dir.name

        for split in _SPLITS:
            # numinst sets: one per {split}/processed/<ckpt> dir with pred_numinst.
            processed_base = exp_dir / split / "processed"
            if processed_base.is_dir():
                for ckpt_dir in sorted(
                    p for p in processed_base.iterdir() if p.is_dir()
                ):
                    if not _numinst_dir_has_data(ckpt_dir):
                        continue
                    sets.append(
                        {
                            "id": NUMINST_PREFIX + _rel(ckpt_dir),
                            "name": _format_numinst_name(exp, split, ckpt_dir),
                            "path": str(ckpt_dir),
                            "default": False,
                            "source": "ppp",
                            "kind": "numinst",
                        }
                    )

            # instances sets: one per leaf param dir containing *.hdf vote outputs.
            if not can_read_hdf:
                continue
            instanced_base = exp_dir / split / "instanced"
            if instanced_base.is_dir():
                param_dirs = sorted(
                    {hdf.parent for hdf in instanced_base.glob("**/*.hdf")}
                )
                for param_dir in param_dirs:
                    sets.append(
                        {
                            "id": INST_PREFIX + _rel(param_dir),
                            "name": _format_instances_name(
                                exp, split, param_dir, instanced_base
                            ),
                            "path": str(param_dir),
                            "default": False,
                            "source": "ppp",
                            "kind": "instances",
                        }
                    )

    return sets


def _resolve_root(set_id: str) -> tuple[str, Path]:
    """Map a PatchPerPix set id back to ``(kind, root_dir)``.

    Raises ``ValueError`` for ids that are not PatchPerPix ids or that escape
    the base, and ``FileNotFoundError`` when the directory is missing.
    """
    if set_id.startswith(NUMINST_PREFIX):
        kind, rel = "numinst", set_id[len(NUMINST_PREFIX):]
    elif set_id.startswith(INST_PREFIX):
        kind, rel = "instances", set_id[len(INST_PREFIX):]
    else:
        raise ValueError(f"Not a PatchPerPix prediction set id: {set_id!r}")

    base = _base()
    root = (base / rel).resolve()
    if root != base and base not in root.parents:
        raise ValueError(f"Prediction set {set_id!r} is outside {base}")
    if not root.is_dir():
        raise FileNotFoundError(f"No directory for prediction set {set_id!r}")
    return kind, root


def is_ppp_set(set_id: str | None) -> bool:
    return bool(set_id) and (
        set_id.startswith(NUMINST_PREFIX) or set_id.startswith(INST_PREFIX)
    )


# --- per-sample lookup -----------------------------------------------------

def _numinst_zarr(root: Path, stem: str) -> Path:
    return root / f"{stem}.zarr"


def _inst_hdf(root: Path, stem: str) -> Path:
    return root / f"{stem}.hdf"


def has_predicted_instances(stem: str, set_id: str | None = None) -> bool:
    """Return True when the given PatchPerPix set has output for ``stem``."""
    if not set_id:
        return False
    try:
        kind, root = _resolve_root(set_id)
    except (FileNotFoundError, ValueError):
        return False
    if kind == "numinst":
        return (_numinst_zarr(root, stem) / "volumes" / "pred_numinst" / ".zarray").is_file()
    return _inst_hdf(root, stem).is_file()


def _set_stems(kind: str, root: Path) -> set[str]:
    if kind == "numinst":
        return {
            p.name[: -len(".zarr")]
            for p in root.glob("*.zarr")
            if (p / "volumes" / "pred_numinst" / ".zarray").is_file()
        }
    return {p.stem for p in root.glob("*.hdf")}


def stems_with_predictions_any() -> set[str]:
    """Stems with output in *any* PatchPerPix set (one scan per set)."""
    stems: set[str] = set()
    for s in discover_prediction_sets():
        try:
            kind, root = _resolve_root(s["id"])
        except (FileNotFoundError, ValueError):
            continue
        stems |= _set_stems(kind, root)
    return stems


def get_predicted_instances_meta(
    stem: str, set_id: str | None = None
) -> dict | None:
    """Return shape/dtype metadata for ``stem`` in the given set, or None."""
    if not set_id:
        return None
    try:
        kind, root = _resolve_root(set_id)
    except (FileNotFoundError, ValueError):
        return None

    if kind == "numinst":
        path = _numinst_zarr(root, stem)
        if not (path / "volumes" / "pred_numinst" / ".zarray").is_file():
            return None
        arr = zarr.open(str(path), mode="r", path=_NUMINST_KEY)
        # (C, Z, Y, X) → report the spatial (Z, Y, X) shape the overlay renders.
        shape = [int(s) for s in arr.shape[1:]]
        return {"shape": shape, "dtype": str(arr.dtype)}

    hdf = _inst_hdf(root, stem)
    if not hdf.is_file():
        return None
    if not _h5py_available():
        return None
    import h5py

    with h5py.File(hdf, "r") as f:
        dset = f[_VOTE_DATASET]
        return {"shape": [int(s) for s in dset.shape], "dtype": str(dset.dtype)}


# --- rendering -------------------------------------------------------------

# Fixed categorical palette for numinst argmax classes (not seeded random).
_NUMINST_RGB_SINGLE = (0, 220, 255)    # cyan — one instance
_NUMINST_RGB_OVERLAP = (255, 40, 200)  # magenta — 2+ overlap


def _encode_numinst_rgb(labels: np.ndarray) -> np.ndarray:
    """Map numinst argmax labels ``{0,1,2}`` to Z,Y,X,3 RGB (fixed palette)."""
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    rgb[labels == 1] = _NUMINST_RGB_SINGLE
    rgb[labels == 2] = _NUMINST_RGB_OVERLAP
    return rgb


def _numinst_argmax_labels(zarr_path: Path, max_size: int) -> tuple[np.ndarray, tuple[int, int, int], int]:
    """Downsample ``pred_numinst`` to a Z,Y,X argmax-of-count label volume.

    Reads the ``(3, Z, Y, X)`` array one output-Z block at a time (all three
    channels for a bounded Z range) so the ~0.5 GB float16 volume is never fully
    materialised. Within each block the count channels are max-pooled spatially,
    then ``argmax`` picks the dominant count → ``{0: background, 1: single,
    2: overlap}``.
    """
    arr = zarr.open(str(zarr_path), mode="r", path=_NUMINST_KEY)
    c, z, y, x = arr.shape
    original_shape = (int(z), int(y), int(x))
    factor = compute_downsample_factor(original_shape, max_size)

    nz, ny, nx = z // factor, y // factor, x // factor
    if nz == 0 or ny == 0 or nx == 0:
        raise ValueError(f"pred_numinst too small to downsample by {factor}")
    ty, tx = ny * factor, nx * factor

    labels = np.zeros((nz, ny, nx), dtype=np.int32)
    for oz in range(nz):
        block = np.asarray(
            arr[:, oz * factor : (oz + 1) * factor, :ty, :tx]
        ).astype(np.float32)  # (C, factor, ty, tx)
        z_max = block.max(axis=1)  # (C, ty, tx)
        pooled = z_max.reshape(c, ny, factor, nx, factor).max(axis=(2, 4))  # (C, ny, nx)
        labels[oz] = pooled.argmax(axis=0)
    return labels, original_shape, factor


@lru_cache(maxsize=16)
def predicted_instances_to_bytes(
    stem: str, max_size: int, set_id: str | None = None
) -> VolumeBytesResult:
    """Load and downsample a PatchPerPix overlay for 3D rendering."""
    if not set_id:
        raise ValueError("PatchPerPix overlay requires a prediction_set id")
    kind, root = _resolve_root(set_id)

    if kind == "numinst":
        zarr_path = _numinst_zarr(root, stem)
        if not (zarr_path / "volumes" / "pred_numinst" / ".zarray").is_file():
            raise FileNotFoundError(f"No pred_numinst for {stem!r} in {root}")
        labels, original_shape, factor = _numinst_argmax_labels(zarr_path, max_size)
        rgb = _encode_numinst_rgb(labels)
        return VolumeBytesResult(
            data=rgb.tobytes(),
            shape=tuple(int(s) for s in rgb.shape[:3]),
            original_shape=original_shape,
            downsample_factor=factor,
            components=3,
        )

    if not _h5py_available():
        raise RuntimeError(
            "PatchPerPix instances overlays require h5py in the webdev env "
            "(pip install h5py into env/webdev.ext3)."
        )
    hdf = _inst_hdf(root, stem)
    if not hdf.is_file():
        raise FileNotFoundError(f"No vote_instances hdf for {stem!r} in {root}")
    import h5py

    with h5py.File(hdf, "r") as f:
        volume = np.asarray(f[_VOTE_DATASET])
    return volume_array_to_bytes(volume, max_size=max_size, encoding="labels_rgb")
