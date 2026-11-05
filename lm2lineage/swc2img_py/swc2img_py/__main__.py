"""CLI.

    python -m swc2img_py swc FILE.swc [FILE2.swc ...] --out renders/
    python -m swc2img_py v1 brain_00049 [brain_00091 ...] --bundle BUNDLE --out renders/ [--rgb] [--zarr]
    python -m swc2img_py zarr brain_00049 [...] --bundle BUNDLE --out renders/ [--rgb]     # convert existing renders
    python -m swc2img_py visibility brain_00049 [...] --bundle BUNDLE --out renders/ --csv visibility.csv
"""

from __future__ import annotations

import argparse
import time


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="swc2img_py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", required=True, help="output directory (renders are read from / written to here)")
    common.add_argument("--voxel", type=float, default=0.44, help="voxel size in um (x, y and z); v1 uses 0.44")
    common.add_argument("--background", type=float, default=10.0, help="SWC2IMG background intensity")
    common.add_argument("--correlation", type=float, default=0.5, help="SWC2IMG noise correlation sigma (um)")
    common.add_argument("--snr", type=float, default=10.0, help="SWC2IMG signal-to-noise ratio")
    common.add_argument("--fiji", help="Fiji directory (default: $FIJI_DIR or D:/tools/Fiji)")
    common.add_argument("--heap", default="5g", help="Java max heap, e.g. 5g")
    bundle = argparse.ArgumentParser(add_help=False)
    bundle.add_argument("image_ids", nargs="+")
    bundle.add_argument("--bundle", required=True, help="1k_v1 neuron-set folder (images.csv, neurons.csv, swc/)")

    p = sub.add_parser("swc", parents=[common], help="render SWC files one by one")
    p.add_argument("files", nargs="+")
    p = sub.add_parser("v1", parents=[common, bundle], help="render 1k_v1 images from the neuron-set bundle")
    p.add_argument("--no-crop", action="store_true", help="don't crop to the v1 field of view")
    p.add_argument("--radius-scale", type=float, default=1.0, help="v1 used 0.75")
    p.add_argument("--rgb", action="store_true",
                   help="MCFO colour render: one SWC2IMG channel per tag (HA, V5, FLAG); always cropped")
    p.add_argument("--tags", choices=["v1", "random"], default="v1",
                   help="with --rgb: colours from neurons.csv (v1) or new random MCFO tags")
    p.add_argument("--seed", type=int, default=0, help="with --tags random")
    p.add_argument("--zarr", action="store_true", help="also write a FISBe-layout zarr with per-neuron masks")
    p = sub.add_parser("zarr", parents=[bundle], help="convert existing cropped renders in --out to FISBe zarr")
    p.add_argument("--out", required=True)
    p.add_argument("--rgb", action="store_true", help="convert the colour render (<id>.swc2img_rgb.tif)")
    p = sub.add_parser("visibility", parents=[bundle],
                       help="per-neuron visible fraction: greyscale SWC2IMG render vs v1 example render")
    p.add_argument("--out", required=True, help="folder with the greyscale renders")
    p.add_argument("--csv", required=True, help="output CSV")
    p.add_argument("--k", type=float, default=2.0, help="visible = SNR >= k (default 2, as in mcfo_sim)")
    a = ap.parse_args(argv)

    if a.cmd == "zarr":
        from .fisbe_zarr import convert_render
        for item in a.image_ids:
            print(f"{item}: {convert_render(a.out, item, a.bundle, rgb=a.rgb)}")
        return
    if a.cmd == "visibility":
        from .visibility import compare_image, write_csv
        rows = []
        for item in a.image_ids:
            rows += compare_image(item, a.bundle, a.out, k=a.k)
            print(f"{item}: {sum(r['image_id'] == item for r in rows)} neurons")
        print(f"wrote {write_csv(rows, a.csv)}")
        return

    from .render import render_swc, render_v1_image, render_v1_image_rgb
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
        msg = f"{item}: {r.volume.shape} -> {r.tif.name}, {r.mip_png.name}"
        if a.cmd == "v1" and a.zarr:
            from .fisbe_zarr import convert_render
            del r
            msg += f", {convert_render(a.out, item, a.bundle, rgb=a.rgb).name}"
        print(f"{msg} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
