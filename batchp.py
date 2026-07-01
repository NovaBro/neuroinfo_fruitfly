import zarr
import numpy as np
from pathlib import Path
import kimimaro
import os
import csv

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
FISBe_DIR = DATA_DIR / "FISBe"
FlyLight_DIR = DATA_DIR / "FlyLight"
FANC_DIR = DATA_DIR / "FANC"

base_path = FISBe_DIR / "completely/train"
output_path = FISBe_DIR / "swc_output"
os.makedirs(output_path, exist_ok=True)

def save_swc(skel, filepath):
    with open(filepath, 'w') as f:
        f.write("# SWC format: id type x y z radius parent_id\n")
        parent_map = {}
        for e in skel.edges:
            parent_map[e[1]] = e[0]
        for i, v in enumerate(skel.vertices):
            node_id = i + 1
            x, y, z = v[2], v[1], v[0]
            radius = skel.radii[i] if hasattr(skel, 'radii') else 1.0
            parent_id = parent_map.get(i, -1)
            parent_id = parent_id + 1 if parent_id != -1 else -1
            f.write(f"{node_id} 0 {x:.2f} {y:.2f} {z:.2f} {radius:.2f} {parent_id}\n")

def evaluate_skeleton_on_instance(skel, instance_mask, label_value):
    inside = sum(
        1 for v in skel.vertices
        if 0 <= int(v[0]) < instance_mask.shape[0]
        and 0 <= int(v[1]) < instance_mask.shape[1]
        and 0 <= int(v[2]) < instance_mask.shape[2]
        and instance_mask[int(v[0]), int(v[1]), int(v[2])] == label_value
    )
    total = len(skel.vertices)
    return inside, total

files = sorted(os.listdir(base_path))
log_path = os.path.join(output_path, "summary_log.csv")

# Load existing log if resuming, so we don't lose previous results
log = []
if os.path.exists(log_path):
    with open(log_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        log = list(reader)

print(f"Found {len(files)} files to process\n")

for idx, fname in enumerate(files):
    sample_name = fname.replace('.zarr', '')
    sample_out_dir = os.path.join(output_path, sample_name)

    # Skip if already processed
    if os.path.exists(sample_out_dir) and len(os.listdir(sample_out_dir)) > 0:
        print(f"[{idx+1}/{len(files)}] Skipping {fname} (already processed)")
        continue

    print(f"[{idx+1}/{len(files)}] Processing {fname}...")
    os.makedirs(sample_out_dir, exist_ok=True)

    try:
        store = zarr.open(os.path.join(base_path, fname))
        gt = store['volumes/gt_instances']
        n_channels = gt.shape[0]

        for ch in range(n_channels):
            gt_ch = np.array(gt[ch])
            unique_labels = np.unique(gt_ch)
            unique_labels = unique_labels[unique_labels != 0]  # skip background

            for label in unique_labels:
                mask = (gt_ch == label).astype(np.uint8)
                voxel_count = mask.sum()
                if voxel_count < 50:  
                    continue

                skels = kimimaro.skeletonize(
                    mask,
                    teasar_params={'scale': 2, 'const': 200},
                    anisotropy=(1, 1, 1),
                    parallel=1
                )
                if not skels:
                    continue
                skel = list(skels.values())[0]

                swc_name = f"ch{ch}_label{label}.swc"
                swc_path = os.path.join(sample_out_dir, swc_name)
                save_swc(skel, swc_path)

                inside, total = evaluate_skeleton_on_instance(skel, gt_ch, label)
                pct = (inside/total*100) if total > 0 else 0

                log.append({
                    'sample': sample_name,
                    'channel': ch,
                    'label': int(label),
                    'vertices': len(skel.vertices),
                    'edges': len(skel.edges),
                    'pct_inside_mask': round(pct, 1)
                })
                print(f"  ch{ch} label{label}: {len(skel.vertices)} verts, {pct:.1f}% inside mask")

        # Save log after each sample completes, so progress is never lost
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample', 'channel', 'label', 'vertices', 'edges', 'pct_inside_mask'])
            writer.writeheader()
            writer.writerows(log)

    except Exception as e:
        print(f"  ERROR on {fname}: {e}")
        continue

print(f"\nDone! Processed {len(files)} files.")
print(f"SWC files saved to: {output_path}")
print(f"Summary log saved to: {log_path}")