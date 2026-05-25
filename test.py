import os
# Force PyImageJ to use Conda's Java, ignoring system defaults
if "CONDA_PREFIX" in os.environ:
    os.environ["JAVA_HOME"] = os.environ["CONDA_PREFIX"]

import imagej
import numpy as np
import scyjava

print("Initializing headless Fiji environment...")
ij = imagej.init('/home/william-zheng/Downloads/Fiji.app', mode='headless')
print(f"ImageJ version: {ij.getVersion()}")

file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "R78H12-20190409_62_D3-m-40x-central-GAL4-unaligned_stack.h5j",
)

# The H5J format is Janelia's HDF5 + JPEG2000 container. It is handled by
# the H5J_Loader_Plugin (an IJ1 plugin), not by the SciJava IO service.
# Route through the legacy IJ.openImage so the plugin is invoked.
print(f"Opening {file_path}...")
IJ = scyjava.jimport("ij.IJ")
imp = IJ.openImage(file_path)
if imp is None:
    raise RuntimeError(
        f"IJ.openImage returned null for {file_path}. "
        "Check that the H5J_Loader_Plugin jar is present in Fiji.app/plugins."
    )

print(f"Opened ImagePlus: title={imp.getTitle()}, "
      f"dims={imp.getWidth()}x{imp.getHeight()}x{imp.getNSlices()} "
      f"channels={imp.getNChannels()} frames={imp.getNFrames()}")

print("Converting ImagePlus to NumPy array...")
np_image = ij.py.from_java(imp)
if hasattr(np_image, "values"):
    np_image = np_image.values

print("Extraction successful!")
print(f"Array Type: {type(np_image)}")
print(f"Data Type (dtype): {np_image.dtype}")
print(f"Array Shape: {np_image.shape}")