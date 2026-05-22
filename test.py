import os
# Force PyImageJ to use Conda's Java 21, ignoring Mac system defaults
if "CONDA_PREFIX" in os.environ:
    os.environ["JAVA_HOME"] = os.environ["CONDA_PREFIX"]

import imagej
import numpy as np

# 1. Initialize headless PyImageJ
print("Initializing headless Fiji environment...")
ij = imagej.init('/Applications/Fiji', mode='headless')

# ... the rest of your script stays exactly the same ...

# Define your file path
file_path = "/Users/williamzheng/Documents/2025_stuff/NewYork/NYU/neuroinfomatics/R78H12-20190409_62_D3-m-40x-central-GAL4-unaligned_stack.h5j"

# 2. Open the h5j file using ImageJ's SCIFIO / Bio-Formats wrapper
print(f"Opening {file_path}...")
j_image = ij.io().open(file_path)

# 3. Convert the Java image object directly into a Python NumPy array
# ij.py.from_java returns an xarray DataArray; adding .values extracts the raw NumPy array
print("Converting Java image to NumPy array...")
np_image = ij.py.from_java(j_image).values

# 4. Check the metadata layout to ensure it fits your model's target input shape
print(f"Extraction successful!")
print(f"Array Type: {type(np_image)}")
print(f"Data Type (dtype): {np_image.dtype}")
print(f"Array Shape: {np_image.shape}")