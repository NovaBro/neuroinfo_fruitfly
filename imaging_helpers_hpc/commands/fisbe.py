import logging

import numpy as np

from imaging_helpers_hpc.imaging import (
    gen_basic_mip,
    gen_instance_projection,
    gen_rotations_and_projections,
)
from imaging_helpers_hpc.loading import load_fisbe_completely
from imaging_helpers_hpc.paths import AnalysisOutputPaths, FisbeDataPaths

logger = logging.getLogger(__name__)


def parse_split_sample(input_file: str) -> tuple[str, str]:
    parts = input_file.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Expected --input-file as split/sample (e.g. train/JRC_...), got: {input_file!r}"
        )
    return parts[0], parts[1]


def _load(split: str, sample: str, fisbe_paths: FisbeDataPaths | None):
    paths = fisbe_paths if fisbe_paths is not None else FisbeDataPaths()
    return load_fisbe_completely(sample, paths, split)


def run_mip_gt(
    split: str,
    sample: str,
    output_paths: AnalysisOutputPaths,
    fisbe_paths: FisbeDataPaths | None = None,
) -> None:
    raw_np, gt_instance_np = _load(split, sample, fisbe_paths)
    gen_basic_mip(raw_np, f"raw_{split}_{sample}", output_paths)
    gen_instance_projection(gt_instance_np, f"gt_{split}_{sample}", output_paths)


def run_rotate(
    split: str,
    sample: str,
    output_paths: AnalysisOutputPaths,
    fisbe_paths: FisbeDataPaths | None = None,
) -> None:
    raw_np, gt_instance_np = _load(split, sample, fisbe_paths)
    rand_axis_int = np.random.randint(0, 3)
    k_rotations = np.random.randint(1, 4)
    logger.info(
        f"shared rotation: rand_axis_int={rand_axis_int}, k_rotations={k_rotations}"
    )
    logger.info("rotating raw")
    gen_rotations_and_projections(
        raw_np,
        f"raw_{split}_{sample}",
        output_paths,
        volume="raw",
        rand_axis_int=rand_axis_int,
        k_rotations=k_rotations,
    )
    logger.info("rotating gt")
    gen_rotations_and_projections(
        gt_instance_np,
        f"gt_{split}_{sample}",
        output_paths,
        volume="gt_instance",
        rand_axis_int=rand_axis_int,
        k_rotations=k_rotations,
    )


def run_channel(
    split: str,
    sample: str,
    output_paths: AnalysisOutputPaths,
    fisbe_paths: FisbeDataPaths | None = None,
) -> None:
    raw_np, _ = _load(split, sample, fisbe_paths)
    random_order = np.random.permutation(3)
    shuffled_image = raw_np[random_order, ...]
    gen_basic_mip(
        shuffled_image,
        f"raw_shuffled_channel_{split}_{sample}",
        output_paths,
    )
