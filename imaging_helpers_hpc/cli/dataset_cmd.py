import argparse
from pathlib import Path

from imaging_helpers_hpc.commands.dataset import (
    DEFAULT_INVENTORY_CSV,
    DEFAULT_MIP_ARRAYS,
    parse_csv_list,
    run_inventory,
    run_mip,
)
from imaging_helpers_hpc.paths import AnalysisOutputPaths, MetricPaths


def _add_common(
    parser: argparse.ArgumentParser,
    *,
    arrays_default: str,
) -> None:
    parser.add_argument(
        "--datasets",
        default="1k_v1,1k_v1_u8",
        help="Comma-separated folder names under fisbe/ "
        "(default: 1k_v1,1k_v1_u8)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=("train", "val", "test"),
        help="Splits to scan (default: train val test)",
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="One brain stem, with or without .zarr",
    )
    parser.add_argument(
        "--arrays",
        default=arrays_default,
        help="Comma-separated names under volumes/ "
        f"(default: {arrays_default})",
    )


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "dataset",
        help="Inventory or MIP sqf-backed datasets under fisbe/",
    )
    modes = parser.add_subparsers(dest="mode", required=True)

    inventory = modes.add_parser(
        "inventory",
        help="Write shape and dtype for each array; no voxel reads",
    )
    _add_common(inventory, arrays_default="raw,gt_instances")
    inventory.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_INVENTORY_CSV,
        help=f"CSV path (default: {DEFAULT_INVENTORY_CSV})",
    )
    inventory.set_defaults(func=run)

    mip = modes.add_parser(
        "mip",
        help="Write a Z-projection PNG for each requested volumes/ array",
    )
    _add_common(mip, arrays_default=DEFAULT_MIP_ARRAYS)
    mip.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Max brains per dataset (default: 3)",
    )
    mip.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI for MIP figures (default: 100)",
    )
    mip.set_defaults(func=run)


def run(
    args: argparse.Namespace,
    analysis_output_paths: AnalysisOutputPaths,
    metric_paths: MetricPaths,
) -> None:
    del metric_paths
    datasets = parse_csv_list(args.datasets)
    arrays = parse_csv_list(args.arrays)
    if not datasets:
        raise SystemExit("error: --datasets is empty")
    if not arrays:
        raise SystemExit("error: --arrays is empty")

    if args.mode == "inventory":
        run_inventory(
            datasets,
            list(args.splits),
            args.sample,
            arrays,
            args.out,
        )
    elif args.mode == "mip":
        run_mip(
            datasets,
            list(args.splits),
            args.sample,
            arrays,
            args.limit,
            args.dpi,
            analysis_output_paths,
        )
