import argparse
import logging
import sys

from imaging_helpers_hpc.cli import any_cmd, biapy_cmd, fisbe_cmd, stats_cmd
from imaging_helpers_hpc.paths import AnalysisOutputPaths, MetricPaths


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)-8s: %(message)s",
        handlers=[
            logging.FileHandler("imaging_helpers_hpc.txt", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger(__name__).setLevel(logging.DEBUG)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Specify the source of data of interest",
    )
    any_cmd.add_parser(subparsers)
    biapy_cmd.add_parser(subparsers)
    fisbe_cmd.add_parser(subparsers)
    stats_cmd.add_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    analysis_output_paths = AnalysisOutputPaths("imaging_helpers_hpc/output")
    metric_paths = MetricPaths()
    args.func(args, analysis_output_paths, metric_paths)
