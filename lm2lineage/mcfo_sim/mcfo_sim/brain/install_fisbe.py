"""Symlink rendered synthetic zarrs into fisbe/<dataset>/{train,val,test}/.

Uses manifest_images.parquet ``split`` and ``image_id`` columns. Only creates
links for image_ids that already have ``<rendered>/<image_id>.zarr``.
"""
from __future__ import annotations

import os
import re
from collections import Counter

import pandas as pd


def install_fisbe(
    manifest_dir: str,
    rendered_dir: str,
    fisbe_root: str,
    dataset: str = "synthetic",
    force: bool = False,
    dry_run: bool = False,
    update_sample_list: str | None = None,
    log=print,
):
    imgs = pd.read_parquet(os.path.join(manifest_dir, "manifest_images.parquet"))
    if "image_id" not in imgs.columns or "split" not in imgs.columns:
        raise ValueError("manifest_images.parquet needs image_id and split columns")

    counts: Counter = Counter()
    linked = {s: [] for s in ("train", "val", "test")}
    skipped_missing = 0
    skipped_exist = 0

    for row in imgs.itertuples(index=False):
        image_id = str(row.image_id)
        split = str(row.split)
        if split not in linked:
            log(f"warn: unknown split {split!r} for {image_id}, skip")
            continue
        src = os.path.abspath(os.path.join(rendered_dir, f"{image_id}.zarr"))
        if not os.path.isdir(src):
            skipped_missing += 1
            continue
        dest_dir = os.path.join(fisbe_root, dataset, split)
        dest = os.path.join(dest_dir, f"{image_id}.zarr")
        if os.path.lexists(dest):
            if os.path.islink(dest):
                cur = os.path.realpath(dest)
                if cur == os.path.realpath(src) and not force:
                    counts[f"{split}_already"] += 1
                    linked[split].append(image_id)
                    continue
                if not force:
                    log(f"refuse: {dest} is a symlink to {cur} (use --force)")
                    skipped_exist += 1
                    continue
                if not dry_run:
                    os.unlink(dest)
            else:
                if not force:
                    log(f"refuse: {dest} exists and is not a symlink (use --force)")
                    skipped_exist += 1
                    continue
                if not dry_run:
                    # do not delete real directories
                    log(f"refuse: will not remove non-symlink directory {dest}")
                    skipped_exist += 1
                    continue

        if dry_run:
            log(f"[dry-run] {dest} -> {src}")
        else:
            os.makedirs(dest_dir, exist_ok=True)
            os.symlink(src, dest)
            log(f"link {dest} -> {src}")
        counts[split] += 1
        linked[split].append(image_id)

    log(
        f"install: linked "
        + ", ".join(f"{s}={counts[s]}" for s in ("train", "val", "test"))
        + f"; already={sum(counts[f'{s}_already'] for s in ('train','val','test'))}"
        + f"; missing_render={skipped_missing}; refused={skipped_exist}"
        + (" [dry-run]" if dry_run else "")
    )

    if update_sample_list:
        _update_sample_list(update_sample_list, dataset, linked, dry_run=dry_run, log=log)

    return dict(counts=dict(counts), linked=linked,
                skipped_missing=skipped_missing, skipped_exist=skipped_exist)


def _update_sample_list(path, dataset, linked, dry_run=False, log=print):
    """Replace or append the FlyLight <dataset> section with train/val/test stems."""
    if os.path.isfile(path):
        text = open(path).read()
    else:
        text = ""

    body_parts = []
    for split in ("train", "val", "test"):
        ids = sorted(linked[split])
        if not ids:
            continue
        body_parts.append(f"{split}:\n" + "\n".join(ids))
    body = "\n\n".join(body_parts)
    n = sum(len(linked[s]) for s in ("train", "val", "test"))
    section = f"({n}) samples for FlyLight {dataset}:\n\n{body}\n"

    # Find existing section for this dataset
    pattern = re.compile(
        rf"^\(\d+\)\s+samples for FlyLight\s+{re.escape(dataset)}\s*:\s*\n"
        rf"(?:.*\n)*?"
        rf"(?=^\(\d+\)\s+samples for FlyLight\s|\Z)",
        re.MULTILINE,
    )
    if pattern.search(text):
        new_text = pattern.sub(section + ("\n" if not section.endswith("\n") else ""), text)
        # avoid double blank at EOF
        new_text = re.sub(r"\n{3,}", "\n\n", new_text)
    else:
        new_text = text.rstrip() + ("\n\n" if text.strip() else "") + section

    if dry_run:
        log(f"[dry-run] would update sample list {path} ({n} stems)")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(new_text)
    log(f"updated sample list {path} ({n} stems for {dataset})")
