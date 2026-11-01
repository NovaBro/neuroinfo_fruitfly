"""CLI for listing neuPrint neurons and downloading neuron-only segmentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    from neuronbridge_data.catalog import (
        make_client,
        print_neuron_table,
        query_neurons,
    )
    from neuronbridge_data.datasets import DATASETS
    from neuronbridge_data.volumes import download_body
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with:\n"
        "  pip install neuprint-python cloud-volume numpy\n"
        f"Original error: {exc}"
    ) from exc

_PACKAGE_DIR = Path(__file__).resolve().parent


def add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="hemibrain",
        help="FlyEM dataset (default: hemibrain)",
    )
    p.add_argument(
        "--token",
        default=None,
        help="neuPrint API token (else env or repo-root .env neuprint_key)",
    )
    p.add_argument(
        "--type",
        dest="type_pattern",
        default=None,
        help="Neuron type regex, e.g. 'MBON.*'",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max neurons to list/download (download default: 3 when using --type)",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "List neuPrint neurons and download neuron-only binary segmentation "
            "(tight seg.npy mask; no EM tissue)."
        )
    )
    sub = p.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list", help="List available neuron samples")
    add_shared_args(list_p)

    dl_p = sub.add_parser(
        "download",
        help="Download neuron-only binary segmentation for selected neurons",
    )
    add_shared_args(dl_p)
    dl_p.add_argument(
        "--body-ids",
        type=int,
        nargs="+",
        default=None,
        help="Explicit body IDs to download",
    )
    dl_p.add_argument(
        "--mip",
        type=int,
        default=5,
        help=(
            "Resolution level: CloudVolume mip or DVID sparsevol scale "
            "(higher = coarser/faster; default: 5)"
        ),
    )
    dl_p.add_argument(
        "--pad",
        type=int,
        default=32,
        help="Padding around skeleton bbox in mip0 voxels (CloudVolume modes; default: 32)",
    )
    dl_p.add_argument(
        "--outdir",
        type=Path,
        default=_PACKAGE_DIR / "samples",
        help="Output root (default: neuronbridge_data/samples)",
    )
    return p


def _json_friendly_neuron_meta(row: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for k, v in row.items():
        if k not in ("type", "instance", "status"):
            continue
        if hasattr(v, "__float__") and v != v:  # NaN
            meta[k] = None
        elif hasattr(v, "item"):
            meta[k] = v.item()
        elif v is not None and not isinstance(v, (str, int, float, bool)):
            meta[k] = str(v)
        else:
            meta[k] = v
    return meta


def cmd_list(args: argparse.Namespace) -> int:
    limit = args.limit if args.limit is not None else 20
    client = make_client(args.dataset, args.token)
    print(
        f"dataset={args.dataset} ({DATASETS[args.dataset]['neuprint']}) limit={limit}"
    )
    df = query_neurons(client, args.type_pattern, limit)
    print_neuron_table(df)
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    cfg = DATASETS[args.dataset]
    client = make_client(args.dataset, args.token)
    out_root = args.outdir / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    neuron_info: dict[int, dict[str, Any]] = {}
    if args.body_ids:
        body_ids = list(args.body_ids)
    else:
        limit = args.limit if args.limit is not None else 3
        if not args.type_pattern and args.limit is None:
            raise SystemExit(
                "download requires --body-ids, or --type / --limit to select neurons."
            )
        df = query_neurons(client, args.type_pattern, limit)
        if df is None or len(df) == 0:
            raise SystemExit("No neurons matched; nothing to download.")
        body_ids = [int(b) for b in df["bodyId"].tolist()]
        for _, row in df.iterrows():
            bid = int(row["bodyId"])
            neuron_info[bid] = _json_friendly_neuron_meta(row)

    print(
        f"dataset={args.dataset} mode={cfg.get('seg_mode')} "
        f"mip={args.mip} pad={args.pad} bodies={body_ids}"
    )

    errors: list[str] = []
    for body_id in body_ids:
        try:
            download_body(
                client,
                cfg,
                body_id,
                args.mip,
                args.pad,
                out_root,
                neuron_meta=neuron_info.get(body_id),
            )
        except Exception as exc:  # noqa: BLE001 — report per-body, continue
            msg = f"bodyId={body_id} failed: {exc}"
            print(msg, file=sys.stderr)
            errors.append(msg)

    if errors:
        raise SystemExit(f"{len(errors)}/{len(body_ids)} download(s) failed")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "download":
        return cmd_download(args)
    raise SystemExit(f"Unknown command: {args.command}")
