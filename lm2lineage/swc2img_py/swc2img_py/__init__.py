"""Drive the SWC2IMG Fiji plugin (Radojevic & Meijering) from Python.

High-level entry points::

    from swc2img_py import render_swc, render_v1_image

`render_swc` renders one SWC file to a float32 ZYX array. `render_v1_image` merges the
neurons of one synth_MCFO 1k_v1 image (from the neuron-set bundle) and renders them in that
image's field of view, so the result lines up voxel-for-voxel with the v1 render.
"""

from .fiji import FijiNotFound, SWC2IMGError, find_fiji, run_swc2img
from .render import RenderResult, render_swc, render_v1_image
from .swc import clip_to_box, read_swc, write_swc

__all__ = [
    "FijiNotFound",
    "RenderResult",
    "SWC2IMGError",
    "clip_to_box",
    "find_fiji",
    "read_swc",
    "render_swc",
    "render_v1_image",
    "run_swc2img",
    "write_swc",
]
