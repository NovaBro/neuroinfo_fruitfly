import argparse

from imaging_helpers_hpc.commands.ppp import run_snapshot_mip
from imaging_helpers_hpc.paths import AnalysisOutputPaths, MetricPaths


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("ppp")
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        choices=("snapshot-mip",),
        help="Mode for PPP data",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Snapshot HDF or a directory of snapshot_*.hdf",
    )
    parser.set_defaults(func=run)


def run(
    args: argparse.Namespace,
    analysis_output_paths: AnalysisOutputPaths,
    metric_paths: MetricPaths,
) -> None:
    del metric_paths
    if args.mode == "snapshot-mip":
        run_snapshot_mip(args.input_file, analysis_output_paths)
