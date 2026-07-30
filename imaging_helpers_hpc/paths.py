import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class MetricPaths:
    def __init__(self):
        self.metric_root = Path("metrics")
        self.metric_biapy = self.metric_root / 'biapy'
        self.metric_ppp = self.metric_root / 'ppp'

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
    def __init__(self, project_root:str | Path = './', data_root: str | Path = 'completely'):
        # Zarr to Tiff converted input files for training BiaPy model
        self.root = Path(project_root) / 'fisbe' / data_root
        
        self.test_dir = self.root / 'test'
        self.train_dir = self.root / 'train'
        self.val_dir = self.root / 'val'
        
        self.paths = {
            'root': self.root,
            'train': self.train_dir,
            'test': self.test_dir,
            'val': self.val_dir,
        }

class AnalysisOutputPaths():
    def __init__(self, output_root:str):
        self.output_root = Path(output_root)
        self.output_images = self.output_root / "images"
        
        self.output_images.mkdir(parents=True, exist_ok=True)
