"""Synthetic FISBe-style MCFO *brain* stacks from the male CNS connectome.

Pipeline (see README_brain.md):
    pool      -> neuron pool + coarse spatial footprints + tissue grid
    manifest  -> deterministic per-image design (crop, neuron ids, tags, imaging)
    stats     -> neuron-count / overlap / usage statistics over the manifest
    render    -> FISBe-format zarr for any subset of manifest rows
    validate  -> integrity checks, FISBe calibration and realism comparison
"""
