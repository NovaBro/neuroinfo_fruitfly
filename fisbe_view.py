import json
import napari
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from imagej_helper.projection import z_project_and_save

DATASET = Path('./fisbe/completely').resolve()
train_path = DATASET / 'train'
file_name = "R38F04-20181005_63_G3.zarr"
OUT_DIR = Path('./fisbe_images')

import zarr

raw = zarr.open(train_path / file_name, mode='r', path="volumes/raw")
seg = zarr.open(train_path / file_name, mode='r', path="volumes/gt_instances")

# optional:
import numpy as np
raw_np = np.array(raw)
# Normalize raw_np to (0, 255) as uint8
raw_np = raw_np.astype(np.float32)
raw_min = raw_np.min()
raw_max = raw_np.max()
raw_np = (raw_np - raw_min) / (raw_max - raw_min)
raw_np = (raw_np * 255).astype(np.uint8)
raw_np = 255 - raw_np

z_project_and_save(raw[0], output_dir=OUT_DIR, out_path="test_projection_0")
z_project_and_save(raw[1], output_dir=OUT_DIR, out_path="test_projection_1")
z_project_and_save(raw[2], output_dir=OUT_DIR, out_path="test_projection_2")
