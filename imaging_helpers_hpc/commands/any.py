from pathlib import Path

from imaging_helpers_hpc.imaging import gen_basic_mip
from imaging_helpers_hpc.loading import load_any_tif
from imaging_helpers_hpc.paths import AnalysisOutputPaths


def run_any_mip(input_file: str | Path, output_paths: AnalysisOutputPaths) -> None:
    input_path = Path(input_file)
    tif_image = load_any_tif(input_path)
    gen_basic_mip(
        tif_image,
        f"any_sample_{input_path.stem}",
        output_paths,
        axis=0,
    )
