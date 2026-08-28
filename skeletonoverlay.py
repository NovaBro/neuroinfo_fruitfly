import zarr
import numpy as np
import napari

def load_swc(filepath):
    nodes = {}
    edges = []
    with open(filepath) as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            nid = int(parts[0])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            parent = int(parts[6])
            nodes[nid] = (x, y, z)
            if parent != -1:
                edges.append((nid, parent))
    return nodes, edges

# Load raw image
store = zarr.open(r"C:\Users\gaura\Downloads\fisbe_v1.0_completely\completely\train\R38F04-20181005_63_G3.zarr")
raw = np.array(store['volumes/raw'][1])  # channel 1

# Load skeletons as paths
swc_files = {
    'neuron1': r"C:\Users\gaura\Downloads\swc_output\R38F04-20181005_63_G3\ch0_label1.swc",
    'neuron2': r"C:\Users\gaura\Downloads\swc_output\R38F04-20181005_63_G3\ch1_label2.swc"
}

viewer = napari.Viewer()

# Add raw image as 3D volume
viewer.add_image(raw, name='raw', colormap='gray', contrast_limits=[0, int(np.percentile(raw, 99.5))])

# Add skeletons as paths
colors = {'neuron1': 'red', 'neuron2': 'blue'}
for name, filepath in swc_files.items():
    nodes, edges = load_swc(filepath)
    paths = []
    for n1, n2 in edges:
        if n1 in nodes and n2 in nodes:
            x0, y0, z0 = nodes[n1]
            x1, y1, z1 = nodes[n2]
            paths.append([[z0, y0, x0], [z1, y1, x1]])
    viewer.add_shapes(paths, shape_type='line', edge_color=colors[name],
                      edge_width=2, name=name)

napari.run()