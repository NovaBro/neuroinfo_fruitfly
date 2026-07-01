import zarr
import numpy as np
from pathlib import Path
import os

import kimimaro
import matplotlib.pyplot as plt

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
FISBe_DIR = DATA_DIR / "FISBe"
FlyLight_DIR = DATA_DIR / "FlyLight"
FANC_DIR = DATA_DIR / "FANC"

target_name = "R38F04-20181005_63_G3"
output_path = FISBe_DIR / "swc_output"
os.makedirs(output_path, exist_ok=True)


store = zarr.open(FISBe_DIR / "completely/train" / f"{target_name}.zarr")
gt = store['volumes/gt_instances']
raw = store['volumes/raw']



neuron_tuples = []

# print(gt.shape[0])

for i in range(gt.shape[0]):
    new_neuron = (np.array(gt[i]) == i+1).astype(np.uint8)
    
    new_tuple = (f'neuron{i+1}', new_neuron, i)
    
    neuron_tuples.append(new_tuple)

def save_swc(skel, filepath):
    """Save a kimimaro skeleton to SWC format"""
    with open(filepath, 'w') as f:
        f.write("# SWC format: id type x y z radius parent_id\n")
        # Build parent mapping from edges
        parent_map = {}
        for e in skel.edges:
            parent_map[e[1]] = e[0]  # child -> parent

        for i, v in enumerate(skel.vertices):
            node_id = i + 1
            x, y, z = v[2], v[1], v[0]  # SWC order is x,y,z
            radius = skel.radii[i] if hasattr(skel, 'radii') else 1.0
            parent_id = parent_map.get(i, -1)
            parent_id = parent_id + 1 if parent_id != -1 else -1
            f.write(f"{node_id} 0 {x:.2f} {y:.2f} {z:.2f} {radius:.2f} {parent_id}\n")
    print(f"Saved: {filepath}")

for name, mask, raw_ch in neuron_tuples:
    print(f"\nSkeletonizing {name}...")
    skels = kimimaro.skeletonize(
        mask,
        teasar_params={'scale': 2, 'const': 200},
        anisotropy=(1, 1, 1),
        parallel=1
    )
    skel = list(skels.values())[0]
    
    # Save as SWC
    save_swc(skel, output_path / rf"{name}.swc")

    # Compare with raw image
    z = mask.shape[0] // 2
    raw_slice = np.array(raw[raw_ch, z])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(name)
    axes[0].imshow(raw_slice, cmap='gray', vmax=raw_slice.max()*0.5)
    axes[0].set_title('Raw Image')
    axes[1].imshow(mask[z], cmap='gray')
    axes[1].set_title('GT Mask')
    axes[2].imshow(raw_slice, cmap='gray', vmax=raw_slice.max()*0.5)
    skel_points = [(int(v[1]), int(v[2])) for v in skel.vertices if int(v[0]) == z]
    if skel_points:
        ys, xs = zip(*skel_points)
        axes[2].scatter(xs, ys, c='red', s=8)
    axes[2].set_title('Skeleton on Raw Image')
    plt.tight_layout()
    # plt.show()