import argparse

from imaging_helpers_hpc.commands.fisbe import (
    parse_split_sample,
    run_channel,
    run_mip_gt,
    run_rotate,
)
from imaging_helpers_hpc.paths import AnalysisOutputPaths, MetricPaths


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("fisbe")
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        choices=("mip-gt", "rotate", "channel"),
        help="Mode for what to do with Fisbe data",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Specify the split and sample [train / test / val]/[sample]. Do not include extension",
    )
    parser.add_argument(
        "-v",
        "--volume",
        default="raw",
        help="Specify what part of the image volume, e.g raw or gt_instance",
    )
    parser.set_defaults(func=run)


def run(
    args: argparse.Namespace,
    analysis_output_paths: AnalysisOutputPaths,
    metric_paths: MetricPaths,
) -> None:
    del metric_paths
    split, sample = parse_split_sample(args.input_file)
    if args.mode == "mip-gt":
        run_mip_gt(split, sample, analysis_output_paths)
    elif args.mode == "rotate":
        run_rotate(split, sample, analysis_output_paths)
    elif args.mode == "channel":
        run_channel(split, sample, analysis_output_paths)
