import argparse

from imaging_helpers_hpc.commands.any import run_any_mip
from imaging_helpers_hpc.paths import AnalysisOutputPaths, MetricPaths


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("any")
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Path to a TIFF file to project",
    )
    parser.set_defaults(func=run)


def run(
    args: argparse.Namespace,
    analysis_output_paths: AnalysisOutputPaths,
    metric_paths: MetricPaths,
) -> None:
    del metric_paths
    run_any_mip(args.input_file, analysis_output_paths)
