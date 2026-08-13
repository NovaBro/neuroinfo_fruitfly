from pathlib import Path

from imaging_helpers_hpc.analysis import get_stats_in_dir, get_stats_in_one_image


def run_stats(input_path: Path) -> None:
    if input_path.is_file():
        print(f"Getting Stats at file {input_path}")
        get_stats_in_one_image(input_path)
    else:
        print(f"Getting Stats at dir {input_path}")
        get_stats_in_dir(input_path)
