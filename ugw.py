from cajal.utilities import cell_iterator_csv
from cajal.ugw import _multicore, UGW
from cajal.sample_swc import compute_icdm_all_geodesic, compute_icdm_all_euclidean

import os
from os.path import join
from pathlib import Path
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

import colorcet as cc

auth_token = "4e1d16bc3043f74ec92c5c9478ffdec8945ff4559c643301e56164807bb95fb3"

client = neu.Client(
    "https://neuprint.janelia.org/",
    token=auth_token,
    dataset="manc:v1.2.3",
)

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
FISBe_DIR = DATA_DIR / "FISBe"
FlyLight_DIR = DATA_DIR / "FlyLight"
FANC_DIR = DATA_DIR / "FANC"
MANC_DIR = DATA_DIR / "MANC"

print(PROJECT_DIR)
print(DATA_DIR)
print(FISBe_DIR)
print(FlyLight_DIR)
print(FANC_DIR)
print(MANC_DIR)

def main():
    neuropil_regex  = "T[1]"
    side_regex  = "[LR]"
    instance_file_name = f"{neuropil_regex}_{side_regex}"
    instance_regex =f".*{neuropil_regex}_{side_regex}.*"
    regex=True

    print(instance_file_name)
    print(instance_regex)

    # t1 = neu.fetch_roi(roi)
    
    
    criteria = neu.NeuronCriteria(
        # rois=roi,
        # target=target,
        instance=instance_regex,
        # class_ = "motor neuron",
        regex=True,
    )

    nl, roi_info = neu.fetch_neurons(
        criteria
    )

    nl = nl[nl['hemilineage'] != "TBD"]

    print(len(nl))

    # nl = nl[nl.bodyId.isin(target_id_list)]
    

    # --- 1. Define all input/output paths for GEODESIC ---
    out_csv_geo = MANC_DIR / f"{instance_file_name}_icdm_geodesic.csv"
    print(out_csv_geo)

    
    out_csv_geo_df = pd.read_csv(out_csv_geo)
    
    print(len(out_csv_geo_df))
    
    print("out_csv_geo_df['cell_id']")
    print(out_csv_geo_df['cell_id'])
    print()
    print("nl.bodyId")
    print(nl.bodyId)
    
    nl = nl[nl.bodyId.isin(out_csv_geo_df['cell_id'])]
    
    print(len(nl.bodyId))


    num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
    print(f"Using {num_cores} cores")

    print("Initializing UGW multicore backend...")
    UGW_multicore = UGW(_multicore)
    print("Successfully initialized UGW multicore backend!")

    # --- CONFIG ---
    MASS_KEPT = 0.80
    EPS = 100.0

    # 2. Compute Unbalanced GW
    print(f"\nComputing Unbalanced Geodesic GW (mass_kept={MASS_KEPT})...")
    try:
        # Prefer passing num_processes so the job uses the CPUs it requested,
        # if this installed cajal version's multicore backend supports it.
        ugw_dmat_raw = UGW_multicore.ugw_armijo_pairwise(
            mass_kept=MASS_KEPT,
            eps=EPS,
            dmats=str(out_csv_geo),  # Your existing Geodesic ICDM CSV path
            num_processes=num_cores,
        )
    except TypeError:
        ugw_dmat_raw = UGW_multicore.ugw_armijo_pairwise(
            mass_kept=MASS_KEPT,
            eps=EPS,
            dmats=str(out_csv_geo),  # Your existing Geodesic ICDM CSV path
        )

    np.save(MANC_DIR / f"{instance_file_name}_UGW_dmat_mass_80_eps_100.npy", ugw_dmat_raw) # This file is available, pre-computed, in the folder linked above.

    # 3. Map the raw Numpy array back to your specific Body IDs
    cells, _ = zip(*cell_iterator_csv(intracell_csv_loc=str(out_csv_geo)))
    ugw_dist_matrix_geo = pd.DataFrame(ugw_dmat_raw, index=cells, columns=cells)

    # 4. Filter to match your target list
    target_id_list = ugw_dist_matrix_geo.index.intersection(list(nl.bodyId))
    dist_matrix_ugw = ugw_dist_matrix_geo.loc[target_id_list, target_id_list]

    # Fill diagonal with zeros
    np.fill_diagonal(dist_matrix_ugw.values, 0.0)

    print(f"\nShape of final Unbalanced Geodesic matrix: {dist_matrix_ugw.shape}")

    out_csv_ugw = MANC_DIR / f"{instance_file_name}_ugw_dist_mass80_eps100.csv"
    dist_matrix_ugw.to_csv(out_csv_ugw)
    print(f"Saved UGW distance matrix to {out_csv_ugw}")

if __name__ == "__main__":
    main()