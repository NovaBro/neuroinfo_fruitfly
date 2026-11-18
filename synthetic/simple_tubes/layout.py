"""Names, splits and the FISBe-style symlink tree for simple_v1.

Standard library only, so the skeleton step runs on any host without numpy or zarr.
See md_guides/dataset/simple_dataset_plan.md.
"""
from __future__ import annotations

import json
import os

DATASET = "simple_v1"
MOUNT = "/simple_v1"          # source SquashFS parts mount here (never under /synth_MCFO)
SPLITS = ("train", "val", "test")
DEFAULT_SPLIT_COUNTS = (800, 100, 100)

# Root .zattrs shared by every sample (same keys as the 1k_v1.1 zarrs).
BASE_ATTRS = {
    "voxel_size_um": [0.44, 0.44, 0.44],
    "axes": "CZYX",
    "channels": ["ch0", "ch1", "ch2"],
}


def sample_name(index: int) -> str:
    return f"tube_{index:05d}"


def parse_split_counts(text: str) -> tuple[int, int, int]:
    parts = [int(v) for v in text.split(",")]
    if len(parts) != 3 or min(parts) < 0:
        raise ValueError(f"--splits needs three non-negative counts, got {text!r}")
    return tuple(parts)


def split_of(index: int, counts=DEFAULT_SPLIT_COUNTS) -> str:
    """Fixed split by index: the first counts[0] are train, the next counts[1] val, ..."""
    if index < 0:
        raise ValueError(f"negative index {index}")
    edge = 0
    for split, n in zip(SPLITS, counts):
        edge += n
        if index < edge:
            return split
    raise ValueError(f"index {index} is past the total {sum(counts)}")


def zarr_relpath(index: int, counts=DEFAULT_SPLIT_COUNTS) -> str:
    return os.path.join(split_of(index, counts), sample_name(index) + ".zarr")


def sample_list_text(counts=DEFAULT_SPLIT_COUNTS, dataset=DATASET, section=1,
                     max_train=None) -> str:
    """sample_list_per_split.txt in the format of fisbe/1k_v1.1. max_train limits the
    train names listed (the shared eval/viewer list need not carry 50k names)."""
    lines = [f"({section}) samples for {dataset}:"]
    start = 0
    for split, n in zip(SPLITS, counts):
        shown = n if (split != "train" or max_train is None) else min(n, max_train)
        lines += ["", f"{split}:"]
        lines += [sample_name(i) for i in range(start, start + shown)]
        start += n
    return "\n".join(lines) + "\n"


# Arrays each zarr carries for PatchPerPix (aliases are relative links inside the image).
ARRAYS = ("raw", "gt_instances", "gt_numinst", "gt_fg_rm_5", "raw_normalized",
          "gt_instances_rm_5")


def _link(link, target, overwrite):
    if os.path.islink(link):
        if os.readlink(link) == target:
            return
        if not overwrite:
            raise FileExistsError(f"{link} -> {os.readlink(link)}, expected {target}")
        os.remove(link)
    elif os.path.exists(link):
        raise FileExistsError(f"{link} exists and is not a symlink")
    os.symlink(target, link)


def write_skeleton(dataset_dir: str, counts=DEFAULT_SPLIT_COUNTS, mount=MOUNT,
                   dataset=DATASET, overwrite=False, fg_dir=None) -> int:
    """Lay out fisbe/<dataset> for a dataset whose zarrs live in SquashFS parts mounted
    at `mount` (listed in <dataset_dir>/source_sqf.txt). Returns the loose zarrs written.

    - train/          -> symlink to <mount>/train. Overlay mounts merge the parts, so the
                         container lists every train zarr; the host keeps no per-sample
                         inodes (50k samples would otherwise cost ~550k).
    - val/, test/     loose zarr skeletons whose volumes/<array> link into <mount>, so
                         host-side tools (ppp_submit_test.sh) can list them.
    - meta/           -> symlink to <mount>/meta
    - fg_dir          -> symlink to <mount>/fg_points (e.g. fisbe/fg_points/<dataset>)
    """
    os.makedirs(dataset_dir, exist_ok=True)
    written = 0
    start = 0
    for split, n in zip(SPLITS, counts):
        if split == "train":
            if n:
                _link(os.path.join(dataset_dir, "train"), os.path.join(mount, "train"),
                      overwrite)
            start += n
            continue
        for index in range(start, start + n):
            rel = zarr_relpath(index, counts)
            zdir = os.path.join(dataset_dir, rel)
            vol = os.path.join(zdir, "volumes")
            os.makedirs(vol, exist_ok=True)
            _write_json(os.path.join(zdir, ".zgroup"), {"zarr_format": 2})
            _write_json(os.path.join(zdir, ".zattrs"), BASE_ATTRS)
            _write_json(os.path.join(vol, ".zgroup"), {"zarr_format": 2})
            for arr in ARRAYS:
                _link(os.path.join(vol, arr), os.path.join(mount, rel, "volumes", arr),
                      overwrite)
            written += 1
        start += n
    _link(os.path.join(dataset_dir, "meta"), os.path.join(mount, "meta"), overwrite)
    if fg_dir:
        os.makedirs(os.path.dirname(os.path.abspath(fg_dir)), exist_ok=True)
        _link(fg_dir, os.path.join(mount, "fg_points"), overwrite)
    with open(os.path.join(dataset_dir, "sample_list_per_split.txt"), "w") as f:
        f.write(sample_list_text(counts, dataset))
    return written


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
