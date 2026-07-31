import os
from pathlib import Path
import uuid
import math

from scipy.special import expit

from tqdm.auto import tqdm

import navis
import navis.interfaces.neuprint as neu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools

import pint

import colorcet as cc


import cajal.sample_swc
import cajal.swc

from cajal.sample_swc import compute_icdm_all_geodesic, read_swc, icdm_geodesic
from cajal.run_gw import compute_gw_distance_matrix
import shutil


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# FIJI_APP = "/home/william-zheng/Downloads/Fiji.app"
# PROJECT_DIR = "/home/william-zheng/Documents/Programming/Python/NeuroInformatics/summer_2026/neuroinfo_fruitfly"
FIJI_APP = "/Users/vuhepola/Desktop/Fiji"
PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
FISBe_DIR = DATA_DIR / "FISBe"
FlyLight_DIR = DATA_DIR / "FlyLight"
FANC_DIR = DATA_DIR / "FANC"
MANC_DIR = DATA_DIR / "MANC"

# print(FIJI_APP)
# print(PROJECT_DIR)
# print(DATA_DIR)
# print(FISBe_DIR)
# print(FlyLight_DIR)
# print(FANC_DIR)
# print(MANC_DIR)

def test_swc_files(swc_dir):
    # 1. Setup paths
    bad_swc_dir = swc_dir.parent / "Bad_Skeletons"
    bad_swc_dir.mkdir(exist_ok=True)
    
    bad_files = []
    swc_files = list(swc_dir.glob("*.swc"))
    
    print(f"Testing {len(swc_files)} skeletons for CAJAL compatibility...")
    
    # 2. Test each file individually to catch the timeouts
    for i, swc_file in enumerate(tqdm(swc_files)):
        if i % 500 == 0:
            print(f"Checked {i}/{len(swc_files)}...")
            
        try:
            # read_swc returns (forest, lookup_dict)
            forest, lookup = read_swc(str(swc_file))
            
            # Grab the largest connected component (the main tree)
            tree = forest[0]
            
            # Test the core geodesic math that triggers the timeout
            _ = icdm_geodesic(tree, 50)
            
        except Exception as e:
            # Catch genuine timeouts (or empty/fragmented skeletons)
            print(f"Failed: {swc_file.name} - {e}")
            bad_files.append(swc_file)
    
    # 3. Move the bad files out of the main directory
    print(f"\nFound {len(bad_files)} problematic skeletons.")
    for bad_file in bad_files:
        shutil.move(str(bad_file), str(bad_swc_dir / bad_file.name))
        
    print(f"Moved bad skeletons to {bad_swc_dir}.")
    print("You can now safely re-run compute_icdm_all_geodesic()!")


def main():
    # --- 2. Dynamically fetch SLURM cores ---
    # This ensures your script scales perfectly with what you requested in SLURM
    # Defaults to 4 if running locally/outside SLURM
    num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
    print(f"Using {num_cores} cores")

    dataset_name = "FANC"

    FANC_SKEL_DIR = FANC_DIR / 'Skeletons'
    
    # test_swc_files(swc_dir = Path(MANC_SKEL_DIR))
    

    # --- 1. Clearly define all input/output paths ---
    out_csv = FANC_DIR / f"{dataset_name}_compute_icdm_all_geodesic.csv"
    out_node_types = FANC_DIR / f"{dataset_name}_node_types.npy"
    
    # Create separate variables for the CSV distances and the NPZ coupling matrices
    gw_dist_csv = FANC_DIR / f"{dataset_name}_morphological_gw_distances.csv"
    gw_coupling_npz = FANC_DIR / f"{dataset_name}_gw_coupling_matrices.npz"

    
    # --- 2. Compute Intracellular Distance Matrices (ICDMs) ---
    print("Computing ICDMs...")
    compute_icdm_all_geodesic(
        infolder=FANC_SKEL_DIR,        # Safely cast to string
        out_csv=str(out_csv),
        out_node_types=str(out_node_types),
        # num_processes=int(num_cores//2),
        num_processes=num_cores,      # <-- FIX 1: Don't hardcode 32
        n_sample=50  
    )

    # --- 3. Compute Pairwise Gromov-Wasserstein Distances ---
    print("Computing GW Distances...")
    compute_gw_distance_matrix(
        intracell_csv_loc=str(out_csv),
        gw_dist_csv_loc=str(gw_dist_csv),             
        gw_coupling_mat_npz_loc=str(gw_coupling_npz), 
        num_processes=num_cores
    )
    
    print("Done Computing GW Distances")
    
    # # --- 4. Load and view the results ---
    # print("Loading results...")
    # # We now correctly load the CSV file
    # gw_distances = pd.read_csv(gw_dist_csv, index_col=0)
    
    # print(f"Shape of distance matrix: {gw_distances.shape}")
    # print(gw_distances.head())
    

if __name__ == "__main__":
    main()

