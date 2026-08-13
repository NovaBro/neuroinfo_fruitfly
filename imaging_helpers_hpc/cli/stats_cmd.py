import argparse
from pathlib import Path

from imaging_helpers_hpc.commands.stats import run_stats
from imaging_helpers_hpc.paths import AnalysisOutputPaths, MetricPaths


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("stats")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="File or directory path for stats",
    )
    parser.set_defaults(func=run)


def run(
    args: argparse.Namespace,
    analysis_output_paths: AnalysisOutputPaths,
    metric_paths: MetricPaths,
) -> None:
    del analysis_output_paths, metric_paths
    run_stats(Path(args.input))
