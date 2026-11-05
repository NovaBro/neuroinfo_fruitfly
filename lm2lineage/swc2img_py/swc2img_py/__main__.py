"""CLI.

    python -m swc2img_py swc FILE.swc [FILE2.swc ...] --out renders/
    python -m swc2img_py v1 brain_00049 [brain_00091 ...] --bundle D:/1k_v1_neuron_sets/1k_v1_neuron_sets --out renders/
"""

from __future__ import annotations

import argparse
import time

from .render import render_swc, render_v1_image, render_v1_image_rgb


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="swc2img_py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", required=True, help="output directory")
    common.add_argument("--voxel", type=float, default=0.44, help="voxel size in um (x, y and z); v1 uses 0.44")
    common.add_argument("--background", type=float, default=10.0, help="SWC2IMG background intensity")
    common.add_argument("--correlation", type=float, default=0.5, help="SWC2IMG noise correlation sigma (um)")
    common.add_argument("--snr", type=float, default=10.0, help="SWC2IMG signal-to-noise ratio")
    common.add_argument("--fiji", help="Fiji directory (default: $FIJI_DIR or D:/tools/Fiji)")
    common.add_argument("--heap", default="5g", help="Java max heap, e.g. 5g")

    p = sub.add_parser("swc", parents=[common], help="render SWC files one by one")
    p.add_argument("files", nargs="+")
    p = sub.add_parser("v1", parents=[common], help="render 1k_v1 images from the neuron-set bundle")
    p.add_argument("image_ids", nargs="+")
    p.add_argument("--bundle", required=True, help="folder with images.csv and swc/")
    p.add_argument("--no-crop", action="store_true", help="don't crop to the v1 field of view")
    p.add_argument("--radius-scale", type=float, default=1.0, help="v1 used 0.75")
    p.add_argument("--rgb", action="store_true",
                   help="MCFO colour render: one SWC2IMG channel per tag (HA, V5, FLAG); always cropped")
    p.add_argument("--tags", choices=["v1", "random"], default="v1",
                   help="with --rgb: colours from neurons.csv (v1) or new random MCFO tags")
    p.add_argument("--seed", type=int, default=0, help="with --tags random")
    a = ap.parse_args(argv)

    kw = dict(background=a.background, correlation_um=a.correlation, snr=a.snr,
              fiji_dir=a.fiji, max_heap=a.heap)
    jobs = a.files if a.cmd == "swc" else a.image_ids
    for item in jobs:
        t = time.time()
        if a.cmd == "swc":
            r = render_swc(item, a.out, voxel_um=a.voxel, **kw)
        elif a.rgb:
            r = render_v1_image_rgb(item, a.bundle, a.out, tags=a.tags, seed=a.seed,
                                    radius_scale=a.radius_scale, voxel_um=a.voxel, **kw)
        else:
            r = render_v1_image(item, a.bundle, a.out, crop=not a.no_crop,
                                radius_scale=a.radius_scale, voxel_um=a.voxel, **kw)
        print(f"{item}: {r.volume.shape} -> {r.tif.name}, {r.mip_png.name} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
