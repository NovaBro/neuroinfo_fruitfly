#!/usr/bin/env python3
"""Download split lines' own MCFO .h5j stacks from the public FlyLight S3 bucket.

For each line in ``--lines`` the script:
  1. finds every release folder that holds the line (a line sits in 3-4 releases);
  2. syncs the folders' ``*metadata.json`` to ``<out>/<release>/<SS>/``;
  3. picks merged MCFO images matching objective / sex / area / slide_code with at
     least ``--min-signal-channels`` (3) signal channels whose requested stack file
     exists on S3, sorted by (release, slide_code, image id), first ``--per-line`` per line;
  4. writes ``--manifest`` (TSV, one row per line; ``status`` = selected | no_match |
     no_mcfo | not_found) and downloads the selected stacks to ``<out>/<s3 key>``.
     Files already on disk with the S3 size are kept.

``--lines`` format (``#`` comments; optional header ``ss sex area slide_code``)::

    SS00043  f  brain  20130412_19_F3     # sex/area/slide_code per line; blank = any
    SS31997                               # bare line name: the --sex/--area defaults

Stacks (``--stack``): ``unaligned`` (native sampling, default), ``gendered``
(JRC2018_FEMALE/MALE aligned), ``unisex`` (JRC2018_Unisex_20x_HR aligned).

S3 metadata facts this relies on:
  * releases with ``<SS>.metadata.json`` list all images; others only have
    per-image ``*-metadata.json`` (no ``imaging_project``/``id``: MCFO is detected
    by the effector ``hsPESTOPT...``, the id comes from the file name);
  * a merged sample record has the stack keys; 63x tile records (LSM only) do not;
  * some Omnibus Broad folders hold metadata only (no stacks) -> no_match.

Runs on the host (python3 + aws CLI, anonymous access). Example (repo root)::

    python3 data_download_methods/download_mcfo.py --objective 20x --dry-run \\
        --lines data_download_methods/himanshu_lines.tsv \\
        --manifest downloads/mcfo_manifest_20x.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote

BUCKET = "janelia-flylight-imagery"
STACK_KEYS = {
    "unaligned": "unaligned_stack",
    "gendered": "aligned_jrc2018_gender_stack",
    "unisex": "aligned_jrc2018_unisex_hr_stack",
}
MCFO_EFFECTOR = re.compile(r"hsPESTOPT|MCFO", re.I)
MANIFEST_COLS = [
    "ss", "status", "release", "image_id", "slide_code", "sex", "objective",
    "area", "tile", "effector", "channel_spec", "dimensions", "voxel_size",
    "stack", "s3_keys", "bytes", "note",
]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def aws(*args: str) -> str:
    return subprocess.run(["aws", *args, "--no-sign-request"], check=True,
                          capture_output=True, text=True).stdout


def s3_ls(prefix: str) -> tuple[list[str], dict[str, int]]:
    """(sub-folder names, {file name: bytes}) directly under ``prefix``."""
    dirs, files = [], {}
    for line in aws("s3", "ls", f"s3://{BUCKET}/{prefix}").splitlines():
        line = line.rstrip("\r")
        if line.strip().startswith("PRE "):
            dirs.append(line.strip()[4:].rstrip("/"))
            continue
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[2].isdigit():
            files[parts[3]] = int(parts[2])
    return dirs, files


def hr(n: int) -> str:
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


def fname(value) -> str:
    """File name from a full https URL (line-level JSON) or bare name (per-image JSON)."""
    return unquote(str(value).rsplit("/", 1)[-1]) if value else ""


def read_lines(path: Path, sex: str, area: str) -> list[dict]:
    wanted = []
    for raw in path.read_text().splitlines():
        cols = raw.split("#", 1)[0].split()
        if not cols or cols[0].lower() == "ss":
            continue
        cols += [""] * (4 - len(cols))
        wanted.append({"ss": cols[0], "sex": (cols[1] or sex).lower(),
                       "area": (cols[2] or area).lower(), "slide_code": cols[3]})
    seen = set()
    return [w for w in wanted if not (w["ss"] in seen or seen.add(w["ss"]))]


def load_images(folder: Path, ss: str) -> list[dict]:
    """Image records of one release folder: line-level JSON if present, else per-image JSONs."""
    line_json = folder / f"{ss}.metadata.json"
    if line_json.is_file():
        return [dict(im) for imgs in json.loads(line_json.read_text()).values() for im in (imgs or [])]
    images = []
    for p in sorted(folder.glob("*-metadata.json")):
        try:
            im = json.loads(p.read_text())
        except ValueError as e:
            print(f"  warn: skip {p}: {e}", file=sys.stderr)
            continue
        m = re.search(r"-(\d+)-metadata\.json$", p.name)
        if m:
            im.setdefault("id", int(m.group(1)))
        images.append(im)
    return images


def is_mcfo(im: dict) -> bool:
    proj = str(im.get("imaging_project") or "").strip().lower()
    return proj == "mcfo" if proj else bool(MCFO_EFFECTOR.search(str(im.get("effector") or "")))


def scan_folder(release: str, ss: str, out: Path, stack_key: str) -> list[dict]:
    """List one release/SS folder, sync its metadata, return merged MCFO candidate records."""
    prefix = f"{release}/{ss}/"
    _, listing = s3_ls(prefix)
    aws("s3", "sync", f"s3://{BUCKET}/{prefix}", str(out / prefix),
        "--exclude", "*", "--include", "*metadata.json", "--only-show-errors")
    recs = []
    for im in load_images(out / prefix, ss):
        if not is_mcfo(im) or not any(k in im for k in STACK_KEYS.values()):
            continue  # not MCFO, or a 63x tile record (LSM only)
        name = fname(im.get(stack_key))
        recs.append({
            "ss": ss, "release": release, "image_id": str(im.get("id", "")),
            "slide_code": str(im.get("slide_code") or ""), "sex": str(im.get("gender") or ""),
            "objective": str(im.get("objective") or ""), "area": str(im.get("area") or ""),
            "tile": str(im.get("tile") or ""), "effector": str(im.get("effector") or ""),
            "channel_spec": str(im.get("channel_spec") or ""),
            "dimensions": str(im.get("dimensions") or ""), "voxel_size": str(im.get("voxel_size") or ""),
            "key": prefix + name if name in listing else "", "bytes": listing.get(name, 0),
        })
    return recs


def sort_key(r: dict):
    iid = r["image_id"]
    return (r["release"], r["slide_code"], int(iid) if iid.isdigit() else 0, iid)


def n_signal(r: dict) -> int:
    """Signal channels from metadata ``channel_spec`` (``sssr`` -> 3). Blank counts as 0: the
    2013 20x MCFO screen images have no channel_spec and are 1 signal + reference (``sr``)."""
    return r["channel_spec"].lower().count("s")


def select(want: dict, recs: list[dict], objective: str, per_line: int,
           min_signal: int) -> tuple[list[dict], str, str]:
    """(picked records, status, note) for one requested line."""
    if not recs:
        return [], "no_mcfo", "no merged MCFO images"

    def ok(r, check_channels=True):
        return ((objective == "any" or r["objective"].lower() == objective)
                and (want["sex"] == "any" or r["sex"].lower() == want["sex"])
                and (want["area"] == "any" or r["area"].lower() == want["area"])
                and (not want["slide_code"] or r["slide_code"] == want["slide_code"])
                and (not check_channels or n_signal(r) >= min_signal))

    hits, ids = [], set()
    for r in sorted(filter(ok, recs), key=sort_key):
        if r["key"] and r["image_id"] not in ids:  # same image can sit in several releases
            ids.add(r["image_id"])
            hits.append(r)
    if not hits:
        avail = Counter(f"{r['objective']}/{r['sex']}/{r['area']}" for r in recs)
        note = "available: " + ", ".join(f"{k}x{v}" for k, v in sorted(avail.items()))
        lacking = sum(1 for r in recs if ok(r) and not r["key"])
        if lacking:
            note += f"; {lacking} match but the stack file is not on S3"
        few = sum(1 for r in recs if ok(r, check_channels=False) and not ok(r))
        if few:
            note += f"; {few} match but have < {min_signal} signal channels"
        return [], "no_match", note
    return hits[:per_line], "selected", f"{len(hits)} match(es), picked {min(per_line, len(hits))}"


def download(key: str, size: int, out: Path) -> str:
    dest = out / key
    if dest.is_file() and dest.stat().st_size == size:
        return "exists"
    dest.parent.mkdir(parents=True, exist_ok=True)
    aws("s3", "cp", f"s3://{BUCKET}/{key}", str(dest), "--only-show-errors")
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lines", type=Path, default=Path("data_download_methods/himanshu_lines.tsv"))
    ap.add_argument("--out", type=Path, default=Path("downloads"), help="download root (mirrors S3 keys)")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="manifest TSV (default <out>/mcfo_manifest_<objective>.tsv)")
    ap.add_argument("--objective", default="20x", help="20x | 40x | 63x | any (default 20x)")
    ap.add_argument("--stack", default="unaligned", choices=sorted(STACK_KEYS))
    ap.add_argument("--sex", default="any", choices=["f", "m", "any"], help="default for lines without one")
    ap.add_argument("--area", default="any", choices=["brain", "vnc", "any"], help="default for lines without one")
    ap.add_argument("--per-line", type=int, default=1)
    ap.add_argument("--min-signal-channels", type=int, default=3,
                    help="skip images with fewer signal channels in metadata channel_spec "
                         "(default 3 = MCFO colours; 0 disables)")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="sync metadata + write manifest, no stacks")
    args = ap.parse_args()
    objective = args.objective.lower()
    manifest = args.manifest or args.out / f"mcfo_manifest_{objective}.tsv"

    wanted = read_lines(args.lines, args.sex, args.area)
    print(f"Lines: {len(wanted)} from {args.lines}  objective={objective} stack={args.stack} "
          f"per_line={args.per_line} dry_run={args.dry_run}")

    releases, _ = s3_ls("")
    with ThreadPoolExecutor(args.parallel) as ex:
        children = dict(zip(releases, ex.map(lambda r: set(s3_ls(r + "/")[0]), releases)))
    folders = [(rel, w["ss"]) for w in wanted for rel in releases if w["ss"] in children[rel]]
    print(f"Indexed {sum(map(len, children.values()))} line folders in {len(releases)} releases; "
          f"{len(folders)} release folder(s) hold the requested lines")

    def scan(folder):
        try:
            return folder, scan_folder(*folder, args.out, STACK_KEYS[args.stack]), None
        except subprocess.CalledProcessError as e:
            return folder, [], (e.stderr or "").strip() or f"exit {e.returncode}"

    recs, failed = {w["ss"]: [] for w in wanted}, {}
    with ThreadPoolExecutor(args.parallel) as ex:
        for (rel, ss), found, err in ex.map(scan, folders):
            recs[ss] += found
            if err:
                failed.setdefault(ss, []).append(rel)
                print(f"  warn: {rel}/{ss}: {err}", file=sys.stderr)

    rows, picked = [], []
    for w in wanted:
        ss = w["ss"]
        if not any(f[1] == ss for f in folders):
            hits, status, note = [], "not_found", "line not on S3"
        else:
            hits, status, note = select(w, recs[ss], objective, args.per_line,
                                        args.min_signal_channels)
        if failed.get(ss):
            note += f"; listing failed: {', '.join(failed[ss])}"
        if not hits:
            rows.append({"ss": ss, "status": status, "note": note})
            print(f"  -- {ss:9s} {status:9s} ({note})")
        for r in hits:
            rows.append({**r, "status": status, "stack": args.stack, "s3_keys": r["key"],
                         "bytes": r["bytes"], "note": note})
            picked.append(r)
            print(f"  ok {ss:9s} {r['release']:26s} {r['image_id']:>8s} "
                  f"{r['objective']}/{r['sex']}/{r['area']:5s} {r['slide_code']:15s} "
                  f"{r['voxel_size']:15s} {hr(r['bytes'])}")

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        f.write(f"# mcfo_manifest lines={args.lines} objective={objective} stack={args.stack} "
                f"per_line={args.per_line} min_signal_channels={args.min_signal_channels}\n")
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS, delimiter="\t", extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)
    total = sum(r["bytes"] for r in picked)
    status = Counter(r["status"] for r in rows)
    print(f"Selected {len({r['ss'] for r in picked})}/{len(wanted)} lines, {len(picked)} stack(s), "
          f"{hr(total)}  {dict(status)}")
    print(f"Manifest: {manifest}")
    if not picked:
        sys.exit("nothing selected")
    if args.dry_run:
        return

    def fetch(r):
        try:
            return r["key"], download(r["key"], r["bytes"], args.out)
        except subprocess.CalledProcessError as e:
            return r["key"], f"FAILED {(e.stderr or '').strip()}"

    with ThreadPoolExecutor(min(4, args.parallel)) as ex:
        results = list(ex.map(fetch, picked))
    for key, res in results:
        print(f"  {res:8s} {key}")
    n_fail = sum(1 for _, res in results if res.startswith("FAILED"))
    print(f"Downloaded to {args.out}/<release>/<SS>/: "
          f"{Counter(res.split()[0] for _, res in results)}")
    if n_fail:
        sys.exit(f"{n_fail} download(s) failed")


if __name__ == "__main__":
    main()
