import argparse

from imaging_helpers_hpc.commands.biapy import run_4pane_mip, run_watershed
from imaging_helpers_hpc.paths import AnalysisOutputPaths, MetricPaths


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("biapy")
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        choices=("4-pane-mip", "watershed"),
        help="Mode for what to do with BiaPy data",
    )
    parser.add_argument(
        "-c",
        "--bia-config-name",
        default="biapy-v1-no-aug",
        help="Select config file for source of data within biapy",
    )
    parser.add_argument(
        "-r",
        "--run",
        default="0",
        help="Select run within config stem",
    )
    parser.add_argument(
        "-s",
        "--sample",
        default="JRC_SS04989-20160318_24_A2.zarr",
        help="What sample to view. Give empty string for dir mode",
    )
    parser.add_argument(
        "-w",
        "--watershed",
        default="seed_map",
        help="What part of each watershed sample",
    )
    parser.set_defaults(func=run)


def run(
    args: argparse.Namespace,
    analysis_output_paths: AnalysisOutputPaths,
    metric_paths: MetricPaths,
) -> None:
    if args.mode == "4-pane-mip":
        run_4pane_mip(args.bia_config_name, analysis_output_paths)
    elif args.mode == "watershed":
        run_watershed(
            args.bia_config_name,
            args.sample,
            args.watershed,
            analysis_output_paths,
            metric_paths=metric_paths,
            run=args.run
        )
