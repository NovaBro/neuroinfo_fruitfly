import argparse

from imaging_helpers_hpc.commands.any import run_any_mip
from imaging_helpers_hpc.paths import AnalysisOutputPaths, MetricPaths


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("any")
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Path to a TIFF, NRRD, NPY, or HDF (.hdf/.h5/.hdf5) file to project",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=None,
        help="HDF internal dataset path (e.g. volumes/raw). Required when the "
        "HDF has multiple datasets under /volumes/",
    )
    parser.set_defaults(func=run)


def run(
    args: argparse.Namespace,
    analysis_output_paths: AnalysisOutputPaths,
    metric_paths: MetricPaths,
) -> None:
    del metric_paths
    run_any_mip(args.input_file, analysis_output_paths, dataset=args.dataset)
