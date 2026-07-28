"""Convert BiaPy-compatible TIFF files to evalinstseg input zarr volumes to."""
import argparse
import os
from pathlib import Path
import shutil
import numpy as np
import tifffile
import zarr

def tiff_to_zarr(tiff_path, zarr_path, zarr_key='volumes/raw'):
    """
    Load a TIFF file and save it as a Zarr file.

    Args:
        tiff_path (str): Path to the input TIFF file.
        zarr_path (str): Path to the output Zarr file.
        zarr_key (str): The dataset key to use inside the Zarr file (default: 'volumes/raw').
    """
    # Load TIFF file as numpy array
    arr = tifffile.imread(tiff_path)
    print(f"Shape: {arr.shape}")

    # # Remove an extra trailing singleton dimension, if present (e.g. (1,...) shape)
    # arr = np.squeeze(arr)

    # Create or open the Zarr file and store the array
    z = zarr.open_group(zarr_path, mode="w", zarr_format=2)
    group, dataset = os.path.split(zarr_key)
    if group:
        g = z.require_group(group)
    else:
        g = z

    # Remove dataset if it already exists (overwrite)
    if dataset in g.array_keys():
        del g[dataset]

    # g.create_dataset(dataset, data=arr, overwrite=True)
    g.create_array(dataset, data=arr, overwrite=True)
    # zarr.create_group(dataset, data=arr, overwrite=True)

    print(f"Converted {tiff_path} -> {zarr_path}:{zarr_key}")

def convert_folder(tiff_folder_path:str | Path, zarr_folder_path:str | Path, zarr_key='volumes/raw'):
    tiff_folder_path = Path(tiff_folder_path)
    zarr_folder_path = Path(zarr_folder_path)
    
    if zarr_folder_path.exists(): shutil.rmtree(zarr_folder_path)
    os.makedirs(zarr_folder_path, exist_ok=True)

    tiff_files = os.listdir(tiff_folder_path)
    for t_file in tiff_files:
        tiff_to_zarr(tiff_folder_path / t_file, zarr_folder_path / Path(t_file).stem, zarr_key=zarr_key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a',
        '--'
    )
    
    job_name = "train_Dn_3d_instance_segmentation"
    # job_name = "train_Dn_0707"
    # input_folder = "per_image_instances"
    input_folder = "per_image_post_processing"
    tiff_path = f"BiaPy/results/{job_name}/results/{job_name}_1/{input_folder}"
    zarr_path = f"BiaPy/results/{job_name}/results/{job_name}_1/per_image_instances_zarr"
    zarr_key = "volumes/pred_instance"
    convert_folder(tiff_path, zarr_path, zarr_key)
