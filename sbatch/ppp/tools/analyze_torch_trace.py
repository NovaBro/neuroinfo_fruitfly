#!/usr/bin/env python3
"""Where does a real PPP training step go? Analyze a torch.profiler chrome trace written by
train.py's _TrainDiagnostics (training.torch_trace_iters; overlay configs/train/trace_train.toml).

Per traced iteration (user annotation `train_iter_N`, main thread):
  wall      CPU span of pipeline.request_batch (= the training step incl. H2D and loss fetch)
  gpu busy  union of kernels + memcpy + memset on the GPU inside that span
  idle      wall - busy, split into gaps; the largest gaps are listed with the CPU op that was
            running on the main thread at the gap's midpoint (what the GPU waited for)
  memcpy    HtoD / DtoH time and bytes (pageable copies are synchronous)
  syncs     cudaStreamSynchronize / cudaDeviceSynchronize / blocking cudaMemcpy* time
Plus kernel time by category and the top kernels.

  python3 sbatch/ppp/tools/analyze_torch_trace.py metrics/ppp/<exp>/train/torch_trace/trace_iters_201-210.json
"""
import argparse
import bisect
import json
import re
import sys
from collections import defaultdict

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
SYNC_RT = re.compile(r"^cuda(StreamSynchronize|DeviceSynchronize|EventSynchronize|Memcpy$|MemcpyAsync$)")
KCAT = [
    ("layout nchw<->nhwc", re.compile(r"nchwToNhwc|nhwcToNchw", re.I)),
    ("conv/gemm (tensor core)", re.compile(r"conv|cudnn|xmma|implicit_gemm|gemm|cutlass|wgrad|dgrad|fprop|sm\d+_", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|unrolled|CatArrayBatched|copy_kernel|fill", re.I)),
    ("reduce", re.compile(r"reduce|norm|softmax|sum_", re.I)),
    ("pool/upsample", re.compile(r"pool|upsample|interp", re.I)),
]


def kcat(name):
    for label, rx in KCAT:
        if rx.search(name):
            return label
    return "other"


def union_len(intervals, lo, hi):
    """Total covered length of [s,e) intervals clipped to [lo,hi), plus the gaps."""
    iv = sorted((max(s, lo), min(e, hi)) for s, e in intervals if e > lo and s < hi)
    busy, gaps, cur = 0.0, [], lo
    for s, e in iv:
        if s > cur:
            gaps.append((cur, s))
        if e > cur:
            busy += e - max(s, cur)
            cur = e
    if cur < hi:
        gaps.append((cur, hi))
    return busy, gaps


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--gaps", type=int, default=12, help="largest idle gaps to list")
    ap.add_argument("--top", type=int, default=15, help="top kernels to list")
    ap.add_argument("--whole", action="store_true",
                    help="no train_iter_* annotations (e.g. profile_ppp_step.py traces): treat "
                         "the longest main-thread span as one step")
    a = ap.parse_args(argv[1:])

    with open(a.trace) as f:
        ev = json.load(f)
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    ev = [e for e in ev if e.get("ph") == "X" and "dur" in e]

    iters = sorted((e for e in ev if e.get("cat") == "user_annotation"
                    and str(e.get("name", "")).startswith("train_iter_")),
                   key=lambda e: e["ts"])
    if not iters and a.whole:
        top = max((e for e in ev if e.get("cat") in ("cpu_op", "python_function", "user_annotation")),
                  key=lambda e: e["dur"])
        lo = min(e["ts"] for e in ev if e["tid"] == top["tid"])
        hi = max(e["ts"] + e["dur"] for e in ev if e["tid"] == top["tid"])
        iters = [{"name": "whole", "ts": lo, "dur": hi - lo, "tid": top["tid"]}]
    if not iters:
        sys.exit("no train_iter_* annotations in trace (use --whole for a step-profiler trace)")
    main_tid = iters[0]["tid"]
    gpu = [e for e in ev if e.get("cat") in GPU_CATS]
    gpu_iv = [(e["ts"], e["ts"] + e["dur"]) for e in gpu]
    rt = [e for e in ev if e.get("cat") == "cuda_runtime"]
    cpu_main = sorted((e for e in ev if e.get("cat") in ("cpu_op", "python_function", "user_annotation")
                       and e["tid"] == main_tid), key=lambda e: e["ts"])
    cpu_starts = [e["ts"] for e in cpu_main]

    def cpu_op_at(t):
        """Innermost main-thread op covering time t (shortest enclosing span)."""
        i = bisect.bisect_right(cpu_starts, t)
        best = None
        for e in cpu_main[max(0, i - 400):i]:
            if e["ts"] <= t <= e["ts"] + e["dur"] and (best is None or e["dur"] < best["dur"]):
                best = e
        return best["name"] if best else "(no op on main thread)"

    tot = defaultdict(float)
    gap_by_op = defaultdict(lambda: [0.0, 0])
    all_gaps = []
    copies = defaultdict(lambda: [0.0, 0])
    syncs = defaultdict(lambda: [0.0, 0])
    kern = defaultdict(lambda: [0.0, 0])
    kc = defaultdict(float)
    print(f"{'iter':>12s} {'wall ms':>8s} {'gpu busy':>9s} {'idle':>7s} {'idle%':>6s} "
          f"{'HtoD ms':>8s} {'sync ms':>8s}")
    for it in iters:
        lo, hi = it["ts"], it["ts"] + it["dur"]
        busy, gaps = union_len(gpu_iv, lo, hi)
        h2d = sum(e["dur"] for e in gpu if e.get("cat") == "gpu_memcpy" and lo <= e["ts"] < hi
                  and "HtoD" in e["name"])
        sy = sum(e["dur"] for e in rt if lo <= e["ts"] < hi and e["tid"] == main_tid
                 and SYNC_RT.search(e["name"]))
        wall = hi - lo
        print(f"{it['name']:>12s} {wall / 1e3:8.1f} {busy / 1e3:9.1f} {(wall - busy) / 1e3:7.1f} "
              f"{100 * (wall - busy) / wall:5.1f}% {h2d / 1e3:8.1f} {sy / 1e3:8.1f}")
        tot["wall"] += wall
        tot["busy"] += busy
        for s, e in gaps:
            op = cpu_op_at((s + e) / 2)
            gap_by_op[op][0] += e - s
            gap_by_op[op][1] += 1
            all_gaps.append((e - s, s - lo, op, it["name"]))
        for e in gpu:
            if lo <= e["ts"] < hi:
                if e.get("cat") == "gpu_memcpy":
                    kind = "HtoD" if "HtoD" in e["name"] else "DtoH" if "DtoH" in e["name"] else e["name"]
                    copies[kind][0] += e["dur"]
                    copies[kind][1] += int(e.get("args", {}).get("bytes", 0))
                elif e.get("cat") == "kernel":
                    kern[e["name"]][0] += e["dur"]
                    kern[e["name"]][1] += 1
                    kc[kcat(e["name"])] += e["dur"]
        for e in rt:
            if lo <= e["ts"] < hi and e["tid"] == main_tid and SYNC_RT.search(e["name"]):
                syncs[e["name"]][0] += e["dur"]
                syncs[e["name"]][1] += 1

    n = len(iters)
    per = lambda us: us / n / 1e3  # noqa: E731  (ms per iteration)
    print(f"\nmean over {n} iterations: wall {per(tot['wall']):.1f} ms, GPU busy {per(tot['busy']):.1f} ms, "
          f"idle {per(tot['wall'] - tot['busy']):.1f} ms ({100 * (1 - tot['busy'] / tot['wall']):.1f}%)")

    print("\n== GPU idle time by main-thread CPU op running during the gap (ms/iter)")
    for op, (us, c) in sorted(gap_by_op.items(), key=lambda kv: -kv[1][0])[:15]:
        print(f"  {per(us):8.2f}  {c / n:6.1f} gaps/iter  {op[:90]}")

    print(f"\n== {a.gaps} largest idle gaps")
    for d, off, op, name in sorted(all_gaps, reverse=True)[:a.gaps]:
        print(f"  {d / 1e3:7.2f} ms at +{off / 1e3:7.1f} ms in {name}: {op[:80]}")

    print("\n== memcpy (per iteration)")
    for kind, (us, b) in sorted(copies.items(), key=lambda kv: -kv[1][0]):
        bw = (b / (us * 1e-6) / 1e9) if us else 0
        print(f"  {kind:6s} {per(us):7.2f} ms  {b / n / 2**20:8.1f} MiB  {bw:6.1f} GB/s")

    print("\n== sync / blocking runtime calls on the main thread (per iteration)")
    for name, (us, c) in sorted(syncs.items(), key=lambda kv: -kv[1][0]):
        print(f"  {per(us):7.2f} ms  {c / n:6.1f} calls  {name}")

    ksum = sum(kc.values())
    print(f"\n== kernel time by category (ms/iter; total {per(ksum):.1f})")
    for cat, us in sorted(kc.items(), key=lambda kv: -kv[1]):
        print(f"  {per(us):8.1f}  {100 * us / ksum:5.1f}%  {cat}")
    print(f"\n== top {a.top} kernels (ms/iter)")
    for name, (us, c) in sorted(kern.items(), key=lambda kv: -kv[1][0])[:a.top]:
        print(f"  {per(us):8.1f}  {100 * us / ksum:5.1f}%  x{c / n:5.1f}  {name[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
