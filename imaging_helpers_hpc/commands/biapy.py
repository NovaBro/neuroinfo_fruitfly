import logging
from pathlib import Path

import numpy as np
import tifffile

from imaging_helpers_hpc.analysis import get_stats_in_one_image
from imaging_helpers_hpc.imaging import (
    gen_biapy_mip_4panel,
    gen_instance_projection,
    gen_topographic_projection,
)
from imaging_helpers_hpc.loading import get_sample_stem, load_biapy_test_sample
from imaging_helpers_hpc.paths import AnalysisOutputPaths, BiapyDataPaths, MetricPaths

logger = logging.getLogger(__name__)


def _resolve_watershed_sample_dir(watershed_root: Path, sample_name: str) -> Path:
    direct = watershed_root / sample_name
    if direct.is_dir():
        return direct
    stem = get_sample_stem(Path(sample_name))
    for name in (f"{stem}.zarr", f"{stem}.zarr.tiff", stem):
        candidate = watershed_root / name
        if candidate.is_dir():
            return candidate
    return direct  # preserve original path for error messages


def run_4pane_mip(
    config_name: str,
    output_paths: AnalysisOutputPaths,
    sample_index: int = 2,
) -> None:
    biapy_paths = BiapyDataPaths(config_name)
    sample_names = sorted(get_sample_stem(p) for p in biapy_paths.per_image.glob("*.tif"))
    logger.info(f"{len(sample_names)} test volumes of BiaPy outputs: {sample_names}")
    sample = sample_names[sample_index]

    raw_vol, prob_vol, inst_vol = load_biapy_test_sample(sample, biapy_paths)
    gen_biapy_mip_4panel(
        raw_vol,
        prob_vol,
        inst_vol,
        output_paths,
        title_prefix=f"{sample}",
    )


def run_watershed(
    config_name: str,
    sample: str,
    watershed: str,
    output_paths: AnalysisOutputPaths,
    metric_paths: MetricPaths | None = None,
    run:str = '0',
) -> None:
    if metric_paths is None:
        metric_paths = MetricPaths()

    watershed_root = metric_paths.metric_biapy / config_name / f"results/{config_name}_{run}/watershed"
    watershed_products = ("growth_mask", "seed_map", "topografic_surface")
    dir_mode = sample == ""

    if dir_mode:
        samples = sorted(p.name for p in watershed_root.iterdir() if p.is_dir())
        products = list(watershed_products)
        logger.info(
            f"watershed dir mode: {len(samples)} samples × "
            f"{len(products)} products under {watershed_root}"
        )
    else:
        samples = [sample]
        products = [watershed]
        logger.info(
            f"watershed single-file mode: sample={sample}, product={watershed}"
        )

    jobs = [(s, product) for s in samples for product in products]
    config_out_root = output_paths.output_root / config_name / run

    for sample_name, product in jobs:
        sample_dir = _resolve_watershed_sample_dir(watershed_root, sample_name)
        resolved_name = sample_dir.name
        in_path = sample_dir / f"{product}.tif"
        if not in_path.is_file():
            if dir_mode:
                logger.warning(f"Missing watershed TIF, skipping: {in_path}")
                continue
            logger.error(f"Missing watershed TIF: {in_path}")
            raise FileNotFoundError(in_path)

        if not dir_mode:
            get_stats_in_one_image(in_path)

        watershed_image = tifffile.imread(in_path)[np.newaxis, ...]
        logger.info(
            f"watershed_image shape: {watershed_image.shape} "
            f"({resolved_name}/{product})"
        )

        out_path = config_out_root / resolved_name / f"{product}.png"
        output_file_name = f"{product}_{resolved_name}"
        if product == "topografic_surface":
            gen_topographic_projection(
                watershed_image,
                output_file_name,
                output_paths,
                output_path=out_path,
            )
        else:
            gen_instance_projection(
                watershed_image,
                output_file_name,
                output_paths,
                output_path=out_path,
            )
