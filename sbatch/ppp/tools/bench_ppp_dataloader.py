"""Benchmark the PPP training data pipeline on CPU (no model, no GPU).

Answers "is the GPU starved, and by which node?" for a given TOML stack:

  1. per-variant single-process timing: sources only, each augmentation alone,
     and the full configured chain; per-node times from gunpowder profiling.
  2. handoff cost: pickle round-trip (what PreCache's mp.Queue does per
     sample) + np.stack for one training batch -- serial main-process work.
  3. throughput: full chain + PreCache(num_workers) + Stack(batch_size),
     batches/s as seen by the training loop.

Run inside the ppp container from PatchPerPix/experiments/ (see
bench_ppp_dataloader_sbatch.sh). Augmentation chain mirrors
flylight/setups/setup01/train.py:train_until -- keep in sync.
"""
import argparse
import logging
import os
import pickle
import statistics
import sys
import time
from glob import glob

import numpy as np
import toml
import gunpowder as gp
from gunpowder.profiling import ProfilingStats
import neurolight.gunpowder as nl

sys.path.insert(0, os.getcwd())
from flylight.setups.setup01.train import get_sources  # noqa: E402
from flylight.setups.setup01.random_hue import RandomHue  # noqa: E402
from flylight.setups.setup01.threaded_prefetch import ThreadedPrefetch  # noqa: E402

logger = logging.getLogger("bench")


def merge_dicts(sink, source):
    for k, v in source.items():
        if isinstance(v, dict) and isinstance(sink.get(k), dict):
            merge_dicts(sink[k], v)
        else:
            sink[k] = v
    return sink


def load_config(paths, fisbe_root, list_datasets=None):
    cfg = {}
    for p in paths:
        merge_dicts(cfg, toml.load(p))
    data = cfg["data"]
    if fisbe_root:
        data["fisbe_root"] = fisbe_root
    names = data.get("train_datasets") or []
    if names:
        files = []
        # list_datasets: dirs actually present (ppp_job.sh stages only $DATASETS;
        # run_ppp warns and skips missing dirs but keeps add_partly on)
        for n in (list_datasets or names):
            files += glob(os.path.join(data["fisbe_root"], n, "train", "*.zarr"))
        if any("partly" in n for n in names):
            data["add_partly"] = True
        data["train_data"] = os.path.join(data["fisbe_root"], names[0], "train")
    else:
        files = glob(os.path.join(data["train_data"], "*.zarr"))
    files = sorted(files)
    if data.get("max_train_samples") is not None:
        files = files[:data["max_train_samples"]]
    flat = {**cfg["data"], **cfg["model"], **cfg["training"],
            **cfg.get("preprocessing", {})}
    flat["data_files"] = files
    return flat


def enabled(augment, name):
    return augment.get(name) is not None and augment.get(name) is not False


def build_chain(config, keys, gt_max_num_inst, only=None):
    """Augmentation chain from train.py. only=None -> as configured;
    only='none' -> no augmentation; only=<name> -> just that augmentation."""
    raw, gt_instances, gt_numinst, loss_mask = keys
    augment = config.get("augmentation", {})
    is_3d = len(config["voxel_size"]) == 3
    add_partly = config.get("add_partly", False)

    def want(name, configured):
        return configured if only is None else only == name

    chain = []
    if want("elastic", enabled(augment, "elastic")):
        e = augment["elastic"] if enabled(augment, "elastic") else {
            "control_point_spacing": [20, 20, 20], "jitter_sigma": [1, 1, 1],
            "rotation_min": -45, "rotation_max": 45, "subsample": 4}
        chain.append(gp.ElasticAugment(
            e["control_point_spacing"], e["jitter_sigma"],
            [e["rotation_min"] * np.pi / 180.0, e["rotation_max"] * np.pi / 180.0],
            uniform_3d_rotation=is_3d, subsample=e.get("subsample", 1),
            spatial_dims=3 if is_3d else 2, temporal_dim=False))
    if want("simple", enabled(augment, "simple")):
        s = augment.get("simple") or {}
        chain.append(gp.SimpleAugment(
            mirror_only=s.get("mirror"), transpose_only=s.get("transpose")))
    if want("permute", config.get("probability_permute", 0) > 0):
        chain.append(nl.PermuteChannel(
            raw, config.get("probability_permute", 0) if only is None else 1.0))
    overlay_on = enabled(augment, "overlay")
    overlay_used = want("overlay", overlay_on)
    if overlay_used:
        p = augment["overlay"]["probability_overlay"] if overlay_on else 0.3
        chain.append(nl.OverlayAugment(
            raw, gt_instances, apply_probability=1.0 if only else p,
            overlay_background=True, numinst=gt_numinst,
            max_numinst=config["max_num_inst"],
            loss_mask=loss_mask if add_partly else None))
    chain.append(gp.ZeroPadChannels(
        gt_instances, gt_max_num_inst * 2 if overlay_used else gt_max_num_inst))
    if want("hue", config.get("probability_hue", 0) > 0):
        chain.append(RandomHue(raw, config["hue_max_change"],
                               config["probability_hue"] if only is None else 1.0))
    if want("intensity", enabled(augment, "intensity")):
        i = augment.get("intensity") or {"scale": [0.9, 1.1], "shift": [-0.1, 0.1]}
        chain.append(gp.IntensityAugment(
            raw, scale_min=i["scale"][0], scale_max=i["scale"][1],
            shift_min=i["shift"][0], shift_max=i["shift"][1],
            z_section_wise=False, clip=False))
    if config.get("reject_min_masked", 0) > 0:
        chain.append(gp.Reject(gt_numinst, min_masked=config["reject_min_masked"],
                               reject_probability=1, mask_filter=1))
    return chain


def make_request(config, keys, input_shape, output_shape):
    raw, gt_instances, gt_numinst, loss_mask = keys
    vs = gp.Coordinate(config["voxel_size"])
    out = gp.Coordinate(output_shape) * vs
    req = gp.BatchRequest()
    req.add(raw, gp.Coordinate(input_shape) * vs)
    if config.get("add_affinities", "cpu") == "loss":
        ps = config["patchshape"][-len(vs):]
        req.add(gt_instances, gp.Coordinate(o + p - 1 for o, p in zip(out, ps)))
    else:
        req.add(gt_instances, out)
    req.add(gt_numinst, out)
    if config.get("add_partly"):
        req.add(loss_mask, out)
    return req


def node_summary(stats):
    rows = []
    for (node, method), s in stats.get_timing_summaries().items():
        if s.counts() == 0:
            continue
        rows.append((s.mean() * s.counts(), node, method, s.mean(), s.counts()))
    rows.sort(reverse=True)
    return rows


def time_variant(name, source, chain, request, n, warmup):
    pipeline = source
    for node in chain:
        pipeline = pipeline + node
    stats = ProfilingStats()
    times = []
    with gp.build(pipeline):
        for i in range(warmup + n):
            t0 = time.perf_counter()
            batch = pipeline.request_batch(request)
            dt = time.perf_counter() - t0
            if i >= warmup:
                times.append(dt)
                stats.merge_with(batch.profiling_stats)
    print(f"\n## variant={name}: n={n} mean={statistics.mean(times):.3f}s "
          f"median={statistics.median(times):.3f}s max={max(times):.3f}s")
    for total, node, method, mean, cnt in node_summary(stats)[:8]:
        print(f"   {node:28s} {str(method):10s} mean={mean:.3f}s  n={cnt}")
    return statistics.mean(times), batch


def handoff_cost(batch, batch_size):
    blob = pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
    t0 = time.perf_counter()
    for _ in range(3):
        pickle.loads(pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL))
    rt = (time.perf_counter() - t0) / 3
    arrays = {k: a.data for k, a in batch.arrays.items()}
    t0 = time.perf_counter()
    for k, a in arrays.items():
        np.stack([a] * batch_size)
    st = time.perf_counter() - t0
    print(f"\n## handoff: one sample pickles to {len(blob) / 2**20:.1f} MiB; "
          f"pickle round-trip {rt * 1000:.0f} ms/sample -> "
          f"{rt * batch_size:.2f}s/batch(bs={batch_size}) + np.stack {st:.2f}s/batch "
          f"(serial in the main process, GPU idle)")
    for k, a in arrays.items():
        print(f"   {str(k):20s} {a.dtype} {a.shape} {a.nbytes / 2**20:.1f} MiB")


def throughput(source, chain, request, config, num_workers, cache_size,
               batch_size, n, prefetch=0, step=0.0, raw_fp16=None):
    """step > 0 sleeps that long per batch to stand in for the GPU train step
    (a CUDA sync releases the GIL like sleep does); times are then the wait
    for the next batch, i.e. GPU idle per step."""
    pipeline = source
    for node in chain:
        pipeline = pipeline + node
    if raw_fp16 is not None:  # mirrors train.py raw_transport_fp16
        pipeline = pipeline + gp.Cast(raw_fp16, dtype=np.float16)
    pipeline = (pipeline + gp.PreCache(cache_size=cache_size, num_workers=num_workers)
                + gp.Stack(batch_size))
    if prefetch > 0:
        pipeline = pipeline + ThreadedPrefetch(num_batches=prefetch)
    with gp.build(pipeline):
        t0 = time.perf_counter()
        pipeline.request_batch(request)
        print(f"\n## throughput: first batch (cache warm-up) {time.perf_counter() - t0:.1f}s")
        times = []
        for _ in range(n):
            if step > 0:
                time.sleep(step)
            t0 = time.perf_counter()
            pipeline.request_batch(request)
            times.append(time.perf_counter() - t0)
    if step > 0:
        print(f"## throughput: prefetch={prefetch} simulated_step={step}s -> "
              f"wait/idle per step {statistics.mean(times):.2f}s "
              f"({100 * statistics.mean(times) / (step + statistics.mean(times)):.0f}% of step)")
    print(f"## throughput: workers={num_workers} cache={cache_size} bs={batch_size}: "
          f"mean {statistics.mean(times):.2f}s/batch, median "
          f"{statistics.median(times):.2f}s, max {max(times):.2f}s "
          f"-> {batch_size / statistics.mean(times):.2f} samples/s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--fisbe-root", default=None)
    ap.add_argument("--list-datasets", nargs="+", default=None,
                    help="dataset dirs to list files from (default: train_datasets)")
    ap.add_argument("--input-shape", type=int, nargs=3, default=[140, 140, 140])
    ap.add_argument("--output-shape", type=int, nargs=3, default=[52, 52, 52],
                    help="U-Net valid output for --input-shape (52 for 140, fmaps-independent)")
    ap.add_argument("--n", type=int, default=20, help="samples per variant")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--variants", default="none,simple,permute,intensity,hue,overlay,elastic,full")
    ap.add_argument("--throughput-batches", type=int, default=15)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--prefetch", type=int, nargs="+", default=[0],
                    help="ThreadedPrefetch depths to compare in the throughput test")
    ap.add_argument("--raw-fp16", action="store_true",
                    help="cast RAW to float16 before PreCache (train.py raw_transport_fp16)")
    ap.add_argument("--simulate-step", type=float, default=0.0,
                    help="seconds to sleep per batch as a stand-in GPU step")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    config = load_config(args.config, args.fisbe_root, args.list_datasets)
    bs = args.batch_size or config["batch_size"]
    workers = args.num_workers or config["num_workers"]
    print(f"configs={args.config}\nfiles={len(config['data_files'])} "
          f"add_partly={config.get('add_partly')} bs={bs} workers={workers} "
          f"augmentation={config.get('augmentation')} hue={config.get('probability_hue')} "
          f"permute={config.get('probability_permute')}\n"
          f"cpus(affinity)={len(os.sched_getaffinity(0))} "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    assert config.get("overlapping_inst"), "bench covers overlapping_inst=true only"

    raw = gp.ArrayKey("RAW")
    gt_instances = gp.ArrayKey("GT_INSTANCES")
    gt_numinst = gp.ArrayKey("GT_NUMINST")
    loss_mask = gp.ArrayKey("LOSS_MASK")
    keys = (raw, gt_instances, gt_numinst, loss_mask)
    arrays = {"raw": raw, "gt_instances": gt_instances, "gt_numinst": gt_numinst}
    graphs = {"gt_close_to_overlap": gp.GraphKey("GT_CLOSE_TO_OVERLAP")}
    if config.get("add_partly"):
        arrays["gt_fgbg_extra"] = gp.ArrayKey("GT_FGBG_EXTRA")
        arrays["loss_mask"] = loss_mask
    if config.get("sample_fg_points"):
        graphs["gt_fg_points"] = gp.GraphKey("GT_FG_POINTS")
    else:
        arrays["gt_sample_mask"] = gp.ArrayKey("GT_SAMPLE_MASK")

    t0 = time.perf_counter()
    source, gt_max_num_inst = get_sources(
        config, arrays, gp.Coordinate(config["voxel_size"]),
        config["data_files"], val=False, graphs=graphs)
    print(f"get_sources: {time.perf_counter() - t0:.1f}s, gt_max_num_inst={gt_max_num_inst}")
    request = make_request(config, keys, args.input_shape, args.output_shape)

    last_batch = None
    for v in filter(None, args.variants.split(",")):
        chain = build_chain(config, keys, gt_max_num_inst,
                            only=None if v == "full" else v)
        _, last_batch = time_variant(v, source, chain, request, args.n, args.warmup)

    if last_batch is not None:
        handoff_cost(last_batch, bs)

    if args.throughput_batches > 0:
        chain = build_chain(config, keys, gt_max_num_inst, only=None)
        if args.raw_fp16:
            print("## throughput: RAW transported as float16")
        for pf in args.prefetch:
            throughput(source, chain, request, config, workers,
                       config["cache_size"], bs, args.throughput_batches,
                       prefetch=pf, step=args.simulate_step,
                       raw_fp16=raw if args.raw_fp16 else None)


if __name__ == "__main__":
    main()
