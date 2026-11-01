from __future__ import annotations

from pathlib import Path

from imaging_helpers_hpc.imaging import gen_basic_mip
from imaging_helpers_hpc.loading import load_any_volume
from imaging_helpers_hpc.paths import AnalysisOutputPaths


def run_any_mip(
    input_file: str | Path,
    output_paths: AnalysisOutputPaths,
    dataset: str | None = None,
) -> None:
    input_path = Path(input_file)
    volume = load_any_volume(input_path, dataset=dataset)
    gen_basic_mip(
        volume,
        f"any_sample_{input_path.stem}",
        output_paths,
        axis=0,
    )
