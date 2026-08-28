import zarr
import numpy as np
import matplotlib.pyplot as plt
import os

base_path = r"C:\Users\gaura\Downloads\fisbe_v1.0_completely\completely\train"
files = os.listdir(base_path)
print(f"Total train files: {len(files)}")
print("Files:", files[:5])

# Loop through first 3 files and visualize middle slice
for fname in files[:3]:
    path = os.path.join(base_path, fname)
    store = zarr.open(path)
    raw = store['volumes/raw']
    gt = store['volumes/gt_instances']
    
    z = raw.shape[1] // 2  # middle slice
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(fname)
    axes[0].imshow(raw[0, z], cmap='gray', vmax=raw[0, z].max()*0.3)
    axes[0].set_title('Raw Ch 0')
    axes[1].imshow(raw[1, z], cmap='gray', vmax=raw[1, z].max()*0.3)
    axes[1].set_title('Raw Ch 1')
    axes[2].imshow(gt[0, z], cmap='tab20')
    axes[2].set_title('GT Instances')
    plt.tight_layout()
    plt.show()