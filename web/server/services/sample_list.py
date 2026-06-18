"""Parse sample_list_per_split.txt into structured sample entries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import FISBE_ROOT, SAMPLE_LIST_PATH


@dataclass(frozen=True)
class SampleEntry:
    split: str
    name: str
    dataset: str  # "completely" or "partly"
    path_exists: bool


def _resolve_zarr_path(split: str, name: str, dataset: str) -> Path:
    if dataset == "completely":
        root = FISBE_ROOT
    else:
        root = FISBE_ROOT.parent / "partly"
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

            if "(1) samples for FlyLight completely:" in line:
                current_dataset = "completely"
                current_split = None
                continue
            if "(2) samples for FlyLight partly:" in line:
                current_dataset = "partly"
                current_split = None
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


def find_sample(name: str, entries: list[SampleEntry] | None = None) -> SampleEntry | None:
    """Return the first matching sample entry by name."""
    entries = entries if entries is not None else parse_sample_list()
    for entry in entries:
        if entry.name == name:
            return entry
    return None


def sample_zarr_path(entry: SampleEntry) -> Path:
    return _resolve_zarr_path(entry.split, entry.name, entry.dataset)
