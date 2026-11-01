"""Load PatchPerPix prediction volumes for the web viewer.

Each PatchPerPix experiment under ``PPP_EXPERIMENTS_BASE`` can expose two kinds
of overlay. Matching leaves under ``val/`` and ``test/`` are merged into one
logical prediction set (same ckpt / VI params), so both FISBe splits appear on
one viewer page:

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

Set ids are source-prefixed (``ppp-numinst:`` / ``ppp-inst:``) and use a
**logical** relative path without the on-disk split segment, e.g.
``ppp-inst:ppp_basic_fmaps24/instanced/80000/…``. Legacy ids that still embed
``val/`` or ``test/`` resolve to the same merged set. Routing is by that prefix
so ``services/predictions.py`` can dispatch without colliding with bare BiaPy
ids.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from config import PPP_EXPERIMENTS_BASE
from services.local_stage import ensure_local
from services.ttl_cache import ttl_cache
from services.volume_pipeline import (
    VolumeBytesResult,
    compute_downsample_factor,
    volume_array_to_bytes,
    zero_small_labels,
)
from services.zarr_store import open_array as open_cached_array

NUMINST_PREFIX = "ppp-numinst:"
INST_PREFIX = "ppp-inst:"

_NUMINST_KEY = "volumes/pred_numinst"
_VOTE_DATASET = "vote_instances"
# Prefer val then test when a stem exists under both leaves and FISBe split is unknown.
_SPLITS = ("val", "test")
# List-mode nests HDFs under {dataset}/; set roots / evaluated leaves omit this.
_DATASET_DIRS = frozenset({
    "completely",
    "partly",
    "synth_vh",
    "synth_vh_u8",
    "1k_v1",
    "1k_v1_u8",
    "himanshu",
})


# --- discovery -------------------------------------------------------------

def _base() -> Path:
    return PPP_EXPERIMENTS_BASE.resolve()


def _rel(root: Path) -> str:
    return str(root.resolve().relative_to(_base()))


def _iter_sample_zarrs(processed_dir: Path):
    """Yield sample zarr dirs: flat ``*.zarr`` and nested ``{dataset}/*.zarr``."""
    yield from (p for p in processed_dir.glob("*.zarr") if p.is_dir())
    yield from (p for p in processed_dir.glob("*/*.zarr") if p.is_dir())


def _find_hdf_parents(base: Path) -> set[Path]:
    """Dirs under ``base`` that directly contain ``*.hdf``, skipping ``*.zarr``.

    Each vote-instances ``.hdf`` sits beside a same-named ``.zarr`` with
    thousands of chunk files; ``glob("**/*.hdf")`` would walk all of them
    (tens of seconds on NFS scratch).
    """
    parents: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(base):
        if any(f.endswith(".hdf") for f in filenames):
            parents.add(Path(dirpath))
        dirnames[:] = [d for d in dirnames if not d.endswith(".zarr")]
    return parents


def _numinst_dir_has_data(processed_dir: Path) -> bool:
    return any(
        (p / "volumes" / "pred_numinst" / ".zarray").is_file()
        for p in _iter_sample_zarrs(processed_dir)
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


def _format_numinst_name(exp: str, ckpt_dir: Path) -> str:
    """Human label for a processed/<ckpt> numinst set.

    Omits the on-disk PPP split (``test``/``val``) from the label; discovery
    merges both dirs into one logical set.
    """
    return f"{exp} · ckpt {ckpt_dir.name} · numinst"


def _format_instances_name(
    exp: str, param_dir: Path, instanced_base: Path
) -> str:
    """Human label encoding ckpt + VI params from the instanced path.

    Example::

        ppp_basic_long8h · ckpt 70000 · instances · th0.3 · mws=T · skel=T

    Omits the on-disk PPP split from the label (same as ``_format_numinst_name``).
    Falls back to the leaf folder name when the path layout is unexpected.
    """
    try:
        parts = param_dir.resolve().relative_to(instanced_base.resolve()).parts
    except ValueError:
        return f"{exp} · instances ({param_dir.name})"

    if not parts:
        return f"{exp} · instances ({param_dir.name})"

    bits: list[str] = [exp]
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
        return f"{exp} · instances ({param_dir.name})"

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


def _logical_rel(rel: str) -> str:
    """Strip ``<exp>/(val|test)/…`` → ``<exp>/…``; leave already-logical paths."""
    parts = Path(rel).parts
    if len(parts) >= 2 and parts[1] in _SPLITS:
        return str(Path(parts[0], *parts[2:]))
    return rel


def _parse_set_id(set_id: str) -> tuple[str, str]:
    """Return ``(kind, raw_rel)`` from a prefixed PatchPerPix set id."""
    if set_id.startswith(NUMINST_PREFIX):
        return "numinst", set_id[len(NUMINST_PREFIX) :]
    if set_id.startswith(INST_PREFIX):
        return "instances", set_id[len(INST_PREFIX) :]
    raise ValueError(f"Not a PatchPerPix prediction set id: {set_id!r}")


def _roots_for_logical(kind: str, logical: str) -> list[Path]:
    """Existing on-disk leaves for a logical rel under both splits (val, test)."""
    parts = Path(logical).parts
    if len(parts) < 2:
        raise ValueError(f"Invalid PatchPerPix logical path: {logical!r}")
    exp, rest = parts[0], parts[1:]
    expected_kind_dir = "processed" if kind == "numinst" else "instanced"
    if rest[0] != expected_kind_dir:
        raise ValueError(
            f"Logical path {logical!r} does not match kind {kind!r} "
            f"(expected …/{expected_kind_dir}/…)"
        )
    base = _base()
    roots: list[Path] = []
    for split in _SPLITS:
        candidate = (base / exp / split / Path(*rest)).resolve()
        if candidate != base and base not in candidate.parents:
            raise ValueError(f"Prediction path {candidate} is outside {base}")
        if candidate.is_dir():
            roots.append(candidate)
    return roots


def _resolve_roots(set_id: str) -> tuple[str, list[Path]]:
    """Map a PatchPerPix set id to ``(kind, [root_dirs…])`` spanning val/test.

    Accepts logical ids (no split) and legacy ids that embed ``val/`` or
    ``test/``; both resolve to the same merged root list.

    Raises ``ValueError`` for non-PPP ids or paths that escape the base, and
    ``FileNotFoundError`` when no matching leaf directory exists.
    """
    kind, raw_rel = _parse_set_id(set_id)
    logical = _logical_rel(raw_rel)
    roots = _roots_for_logical(kind, logical)
    if not roots:
        raise FileNotFoundError(f"No directory for prediction set {set_id!r}")
    return kind, roots


def _resolve_root(set_id: str) -> tuple[str, Path]:
    """Map a PatchPerPix set id to ``(kind, first_root)``.

    Prefer ``_resolve_roots`` / ``_root_for_stem`` for new code. Kept for
    callers that only need any existing leaf (first is ``val`` then ``test``).
    """
    kind, roots = _resolve_roots(set_id)
    return kind, roots[0]


def _split_of_root(root: Path) -> str | None:
    """Return ``val`` or ``test`` if ``root`` sits under that PPP split dir."""
    try:
        parts = root.resolve().relative_to(_base()).parts
    except ValueError:
        return None
    if len(parts) >= 2 and parts[1] in _SPLITS:
        return parts[1]
    return None


@lru_cache(maxsize=1)
def _stem_to_fisbe_split() -> dict[str, str]:
    """Map ``{dataset}/{name}`` (and bare name when unique) → FISBe split."""
    from services.sample_list import parse_sample_list

    out: dict[str, str] = {}
    bare_counts: dict[str, int] = {}
    for entry in parse_sample_list():
        out[f"{entry.dataset}/{entry.name}"] = entry.split
        bare_counts[entry.name] = bare_counts.get(entry.name, 0) + 1
    for entry in parse_sample_list():
        if bare_counts.get(entry.name, 0) == 1 and entry.name not in out:
            out[entry.name] = entry.split
    return out


def _stem_exists(kind: str, root: Path, stem: str) -> bool:
    if kind == "numinst":
        return (
            _numinst_zarr(root, stem) / "volumes" / "pred_numinst" / ".zarray"
        ).is_file()
    return _inst_hdf(root, stem).is_file()


def _root_for_stem(
    kind: str,
    roots: list[Path],
    stem: str,
    preferred_split: str | None = None,
) -> Path | None:
    """Pick the leaf that holds ``stem``, preferring ``preferred_split`` when set.

    If ``preferred_split`` is omitted, look up the FISBe sample list. When the
    stem exists under multiple leaves, prefer that split; otherwise walk roots
    in ``val``, ``test`` order.
    """
    if preferred_split is None:
        splits = _stem_to_fisbe_split()
        preferred_split = splits.get(stem) or splits.get(stem.rsplit("/", 1)[-1])

    hits = [r for r in roots if _stem_exists(kind, r, stem)]
    if not hits:
        return None
    if preferred_split:
        for r in hits:
            if _split_of_root(r) == preferred_split:
                return r
    return hits[0]


@ttl_cache(60)
def discover_prediction_sets() -> list[dict]:
    """Discover PatchPerPix numinst and instances sets under the base.

    Matching ``val/`` and ``test/`` leaves (same ckpt / VI params) collapse into
    one logical set. Each entry is
    ``{"id", "name", "path", "default", "source", "kind"}``.
    ``default`` is always False — the viewer defaults to a BiaPy set and the
    user opts into a PatchPerPix overlay via the dropdown.

    Instances sets are omitted when ``h5py`` is not installable in the server
    env (the HDF loader would fail at request time otherwise).

    Display ``name`` encodes ckpt and VI params; set ``id`` is the logical
    relative path (no split). Legacy split-embedded ids still resolve.
    """
    base = _base()
    if not base.is_dir():
        return []

    can_read_hdf = _h5py_available()

    # key → (kind, exp, after_split, [roots], name_source_root, name_instanced_base)
    groups: dict[tuple[str, str, str], dict] = {}

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
                    after = str(Path("processed") / ckpt_dir.name)
                    key = ("numinst", exp, after)
                    g = groups.get(key)
                    if g is None:
                        groups[key] = {
                            "kind": "numinst",
                            "exp": exp,
                            "after": after,
                            "roots": [ckpt_dir],
                            "name_root": ckpt_dir,
                        }
                    else:
                        g["roots"].append(ckpt_dir)

            # instances sets: one per leaf param dir containing *.hdf vote outputs.
            if not can_read_hdf:
                continue
            instanced_base = exp_dir / split / "instanced"
            if instanced_base.is_dir():
                param_dirs: set[Path] = set()
                for parent in _find_hdf_parents(instanced_base):
                    if parent.name in _DATASET_DIRS:
                        parent = parent.parent
                    param_dirs.add(parent)
                for param_dir in sorted(param_dirs):
                    try:
                        rel_parts = param_dir.resolve().relative_to(
                            instanced_base.resolve()
                        ).parts
                    except ValueError:
                        continue
                    after = str(Path("instanced", *rel_parts))
                    key = ("instances", exp, after)
                    g = groups.get(key)
                    if g is None:
                        groups[key] = {
                            "kind": "instances",
                            "exp": exp,
                            "after": after,
                            "roots": [param_dir],
                            "name_root": param_dir,
                            "instanced_base": instanced_base,
                        }
                    else:
                        g["roots"].append(param_dir)

    sets: list[dict] = []
    for key in sorted(groups.keys()):
        g = groups[key]
        kind = g["kind"]
        exp = g["exp"]
        after = g["after"]
        roots: list[Path] = g["roots"]
        # Stable val-then-test order (discovery already walks _SPLITS that way).
        logical = f"{exp}/{after}"
        if kind == "numinst":
            name = _format_numinst_name(exp, g["name_root"])
            prefix = NUMINST_PREFIX
        else:
            name = _format_instances_name(
                exp, g["name_root"], g["instanced_base"]
            )
            prefix = INST_PREFIX
        sets.append(
            {
                "id": prefix + logical,
                "name": name,
                "path": " | ".join(str(r) for r in roots),
                "default": False,
                "source": "ppp",
                "kind": kind,
            }
        )

    return sets


def is_ppp_set(set_id: str | None) -> bool:
    return bool(set_id) and (
        set_id.startswith(NUMINST_PREFIX) or set_id.startswith(INST_PREFIX)
    )


# --- per-sample lookup -----------------------------------------------------

def _numinst_zarr(root: Path, stem: str) -> Path:
    """Resolve flat or nested list-mode zarr for ``stem``.

    Prefers namespaced ``{dataset}/{name}.zarr`` when ``stem`` contains ``/``.
    Otherwise tries flat ``{stem}.zarr``, then a single nested match.
    """
    base = stem.rsplit("/", 1)[-1]
    if "/" in stem:
        nested = root / f"{stem}.zarr"
        if nested.is_dir():
            return nested
        # Fall through to basename only under the declared dataset dir.
        dataset, _ = stem.split("/", 1)
        under = root / dataset / f"{base}.zarr"
        if under.is_dir():
            return under
        return nested
    direct = root / f"{stem}.zarr"
    if direct.is_dir():
        return direct
    matches = sorted(p for p in root.glob(f"*/{base}.zarr") if p.is_dir())
    if len(matches) == 1:
        return matches[0]
    return direct


def _inst_hdf(root: Path, stem: str) -> Path:
    """Resolve flat or nested list-mode hdf for ``stem``."""
    base = stem.rsplit("/", 1)[-1]
    if "/" in stem:
        nested = root / f"{stem}.hdf"
        if nested.is_file():
            return nested
        dataset, _ = stem.split("/", 1)
        under = root / dataset / f"{base}.hdf"
        if under.is_file():
            return under
        return nested
    direct = root / f"{stem}.hdf"
    if direct.is_file():
        return direct
    matches = sorted(p for p in root.glob(f"*/{base}.hdf") if p.is_file())
    if len(matches) == 1:
        return matches[0]
    return direct


def has_predicted_instances(stem: str, set_id: str | None = None) -> bool:
    """Return True when the given PatchPerPix set has output for ``stem``."""
    if not set_id:
        return False
    try:
        kind, roots = _resolve_roots(set_id)
    except (FileNotFoundError, ValueError):
        return False
    return _root_for_stem(kind, roots, stem) is not None


def _set_stems(kind: str, root: Path) -> set[str]:
    """Return stems for viewer matching.

    Nested ``{dataset}/{stem}`` when under a known dataset dir; bare stem when
    flat at the set root. Always include the namespaced form for nested files
    so colliding basenames across datasets stay distinct.
    """
    stems: set[str] = set()
    if kind == "numinst":
        paths = [
            p
            for p in _iter_sample_zarrs(root)
            if (p / "volumes" / "pred_numinst" / ".zarray").is_file()
        ]
        for p in paths:
            bare = p.name[: -len(".zarr")]
            if p.parent != root and p.parent.name in _DATASET_DIRS:
                stems.add(f"{p.parent.name}/{bare}")
            else:
                stems.add(bare)
        return stems
    for p in root.glob("*.hdf"):
        stems.add(p.stem)
    for p in root.glob("*/*.hdf"):
        if p.parent.name in _DATASET_DIRS:
            stems.add(f"{p.parent.name}/{p.stem}")
        else:
            stems.add(p.stem)
    return stems


def stems_for_set(set_id: str) -> set[str]:
    """Union of stems with output across all roots of a logical PatchPerPix set."""
    try:
        kind, roots = _resolve_roots(set_id)
    except (FileNotFoundError, ValueError):
        return set()
    stems: set[str] = set()
    for root in roots:
        stems |= _set_stems(kind, root)
    return stems


@ttl_cache(60)
def stems_with_predictions_any() -> set[str]:
    """Stems with output in *any* PatchPerPix set (one scan per set)."""
    stems: set[str] = set()
    for s in discover_prediction_sets():
        stems |= stems_for_set(s["id"])
    return stems


def get_predicted_instances_meta(
    stem: str, set_id: str | None = None
) -> dict | None:
    """Return shape/dtype metadata for ``stem`` in the given set, or None."""
    if not set_id:
        return None
    try:
        kind, roots = _resolve_roots(set_id)
    except (FileNotFoundError, ValueError):
        return None
    root = _root_for_stem(kind, roots, stem)
    if root is None:
        return None

    if kind == "numinst":
        path = _numinst_zarr(root, stem)
        if not (path / "volumes" / "pred_numinst" / ".zarray").is_file():
            return None
        arr = open_cached_array(path, _NUMINST_KEY)
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
    arr = open_cached_array(zarr_path, _NUMINST_KEY)
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
    stem: str,
    max_size: int,
    set_id: str | None = None,
    min_size: int = 0,
) -> VolumeBytesResult:
    """Load and downsample a PatchPerPix overlay for 3D rendering.

    ``min_size`` drops instance labels with fewer than that many voxels
    (display-only). Ignored for ``numinst`` overlays (not per-body IDs).
    """
    if not set_id:
        raise ValueError("PatchPerPix overlay requires a prediction_set id")
    kind, roots = _resolve_roots(set_id)
    root = _root_for_stem(kind, roots, stem)
    if root is None:
        raise FileNotFoundError(
            f"No PatchPerPix overlay for {stem!r} in set {set_id!r}"
        )

    if kind == "numinst":
        zarr_path = _numinst_zarr(root, stem)
        if not (zarr_path / "volumes" / "pred_numinst" / ".zarray").is_file():
            raise FileNotFoundError(f"No pred_numinst for {stem!r} in {root}")
        labels, original_shape, factor = _numinst_argmax_labels(
            ensure_local(zarr_path), max_size
        )
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

    with h5py.File(ensure_local(hdf), "r") as f:
        volume = np.asarray(f[_VOTE_DATASET])
    volume = zero_small_labels(volume, min_size)
    return volume_array_to_bytes(volume, max_size=max_size, encoding="labels_rgb")
