"""Compute the common set of MANC T1_[LR] body IDs covered by all four
distance/similarity matrices (geodesic GW, euclidean GW, NBLAST, UGW), plus
their hemilineage labels, so the classification comparison evaluates every
method on an identical neuron set.

Saves data/MANC/results/common_ids.npy and data/MANC/results/id_to_class.pkl.
"""

import pickle
from pathlib import Path

import navis.interfaces.neuprint as neu
import numpy as np
import pandas as pd

import ugw_common as common

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
MANC_DIR = DATA_DIR / "MANC"
RESULTS_DIR = MANC_DIR / "results"

auth_token = "4e1d16bc3043f74ec92c5c9478ffdec8945ff4559c643301e56164807bb95fb3"

client = neu.Client(
    "https://neuprint.janelia.org/",
    token=auth_token,
    dataset="manc:v1.2.3",
)


def main():
    neuropil_regex = "T[1]"
    side_regex = "[LR]"
    instance_file_name = f"{neuropil_regex}_{side_regex}"
    instance_regex = f".*{neuropil_regex}_{side_regex}.*"

    criteria = neu.NeuronCriteria(instance=instance_regex, regex=True)
    nl, _ = neu.fetch_neurons(criteria)
    nl = nl[nl["hemilineage"] != "TBD"]
    print(f"Neurons with known hemilineage: {len(nl)}")

    geo_ids = set(
        pd.read_csv(MANC_DIR / f"{instance_file_name}_icdm_geodesic.csv")["cell_id"]
    )
    euc_ids = set(
        pd.read_csv(MANC_DIR / f"{instance_file_name}_icdm_euclidean.csv")["cell_id"]
    )
    nblast_ids = set(
        pd.read_csv(MANC_DIR / f"{instance_file_name}_nblast_dist.csv", index_col=0).index
    )
    ugw_ids = set(
        pd.read_csv(common.FINAL_UGW_CSV, index_col=0).index
    )

    print(f"Geodesic GW IDs: {len(geo_ids)}")
    print(f"Euclidean GW IDs: {len(euc_ids)}")
    print(f"NBLAST IDs: {len(nblast_ids)}")
    print(f"UGW IDs: {len(ugw_ids)}")

    common_ids = set(nl["bodyId"]) & geo_ids & euc_ids & nblast_ids & ugw_ids
    common_ids = np.array(sorted(common_ids))
    print(f"Common body IDs across all four methods + hemilineage labels: {len(common_ids)}")

    id_to_class = dict(zip(nl["bodyId"], nl["hemilineage"]))
    id_to_class = {k: v for k, v in id_to_class.items() if k in set(common_ids)}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_DIR / "common_ids.npy", common_ids)
    with open(RESULTS_DIR / "id_to_class.pkl", "wb") as f:
        pickle.dump(id_to_class, f)

    print(f"Saved common_ids.npy and id_to_class.pkl to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
