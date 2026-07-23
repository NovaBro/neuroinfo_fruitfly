import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BiapyDataPaths:
    def __init__(self, config_name:str | Path):
        # Zarr to Tiff converted input files for training BiaPy model
        self.BIAPY_TEST_RAW_DIR = Path("fisbe/biapy/test/raw")

        # Getting the images data of the model results
        self.result_root = Path(f"BiaPy/results/{config_name}/results/{config_name}_1")
        self.per_image = self.result_root / "per_image"
        self.per_image_instances = self.result_root / "per_image_instances"
        self.watershed = self.result_root / "watershed"

class FisbeDataPaths:
    def __init__(self, root:str | Path = './'):
        # Zarr to Tiff converted input files for training BiaPy model
        self.completely_root = Path(root) / "fisbe/completely"
        
        self.completely_test = self.completely_root / "test"
        self.completely_train = self.completely_root / "train"
        self.completely_val = self.completely_root / "val"
        
        self.paths = {
            'root': self.completely_root,
            'train': self.completely_train,
            'test': self.completely_test,
            'val': self.completely_val,
        }

class AnalysisOutputPaths():
    def __init__(self, output_root:str):
        self.output_root = Path(output_root)
        self.output_images = self.output_root / "images"
        
        self.output_images.mkdir(parents=True, exist_ok=True)
