"""Parse sample_list_per_split.txt into structured sample entries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import FISBE_ROOT, SAMPLE_LIST_PATH

# Folder names under fisbe/ (and nested PPP prediction dirs).
KNOWN_DATASETS = (
    "completely",
    "partly",
    "synth_vh",
    "synth_vh_u8",
    "1k_v1",
    "1k_v1_u8",
    "himanshu",
)

# sample_list section headers → dataset folder name.
_SECTION_HEADERS = (
    ("(1) samples for FlyLight completely:", "completely"),
    ("(2) samples for FlyLight partly:", "partly"),
    ("(3) samples for synth_vh:", "synth_vh"),
    ("(4) samples for synth_vh_u8:", "synth_vh_u8"),
    ("(5) samples for himanshu:", "himanshu"),
    ("(6) samples for 1k_v1:", "1k_v1"),
    ("(7) samples for 1k_v1_u8:", "1k_v1_u8"),
)


@dataclass(frozen=True)
class SampleEntry:
    split: str
    name: str
    dataset: str
    path_exists: bool


def _resolve_zarr_path(split: str, name: str, dataset: str) -> Path:
    if dataset == "completely":
        root = FISBE_ROOT
    else:
        root = FISBE_ROOT.parent / dataset
    return root / split / f"{name}.zarr"


def parse_sample_list(list_path: Path | None = None) -> list[SampleEntry]:
    """Parse train/val/test sections from the FISBe sample list file."""
    list_path = list_path or SAMPLE_LIST_PATH
    entries: list[SampleEntry] = []
    current_split: str | None = None
    current_dataset = "completely"

    with open(list_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            matched_header = False
            for header, dataset in _SECTION_HEADERS:
                if header in line:
                    current_dataset = dataset
                    current_split = None
                    matched_header = True
                    break
            if matched_header:
                continue

            if line in ("train:", "val:", "test:"):
                current_split = line[:-1]
                continue

            if current_split is None:
                continue

            zarr_path = _resolve_zarr_path(current_split, line, current_dataset)
            entries.append(
                SampleEntry(
                    split=current_split,
                    name=line,
                    dataset=current_dataset,
                    path_exists=zarr_path.is_dir(),
                )
            )

    return entries


def find_sample(
    name: str,
    entries: list[SampleEntry] | None = None,
    *,
    dataset: str | None = None,
) -> SampleEntry | None:
    """Return the matching sample entry by ``(dataset, name)`` when dataset is set."""
    entries = entries if entries is not None else parse_sample_list()
    if dataset is not None:
        for entry in entries:
            if entry.name == name and entry.dataset == dataset:
                return entry
        return None
    for entry in entries:
        if entry.name == name:
            return entry
    return None


def sample_zarr_path(entry: SampleEntry) -> Path:
    return _resolve_zarr_path(entry.split, entry.name, entry.dataset)


def namespaced_stem(entry: SampleEntry) -> str:
    """``{dataset}/{name}`` for prediction / metrics matching."""
    return f"{entry.dataset}/{entry.name}"
