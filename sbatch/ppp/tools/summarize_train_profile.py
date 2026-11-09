#!/usr/bin/env python3
"""Summarize short PPP training profile runs (e.g. 300-it profile_noenc_*.toml jobs).

For each SLURM job id: steady-state s/iter from `Batch: iteration=N, time=T` lines
(wall per iteration, main process) and `Train process: ... time=T` (TorchTrainExt step),
skipping warm-up iterations; GPU util/memory from gpu_log/ppp/gpu_usage_log_<exp>_<job>.csv
over the same window (rows with memory in use and util > 0 after the first batch).

  python3 sbatch/ppp/tools/summarize_train_profile.py 22104455 22104461 22104466 22104468
  python3 sbatch/ppp/tools/summarize_train_profile.py --skip 50 JOBID...
"""
import argparse
import glob
import os
import re
import statistics as st
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BATCH_RE = re.compile(r"Batch: iteration=(\d+), time=([\d.]+)")
TRAIN_RE = re.compile(r"Train process: iteration=(\d+) .*time=([\d.]+)")
EXP_RE = re.compile(r"ppp_job exp=(\S+)")
CFG_RE = re.compile(r"configs=(.*?) ckpt=")


def job_log(job):
    hits = glob.glob(os.path.join(REPO, "sbatch/ppp/logs", f"*-{job}.out"))
    return hits[0] if hits else None


def gpu_stats(exp, job):
    path = os.path.join(REPO, "gpu_log/ppp", f"gpu_usage_log_{exp}_{job}.csv")
    if not os.path.isfile(path):
        return None
    util, mem = [], []
    with open(path) as f:
        next(f, None)
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                u = float(parts[3].split()[0])
                m = float(parts[5].split()[0])
            except ValueError:
                continue
            if m > 1000:  # model loaded
                util.append(u)
                mem.append(m)
    if not util:
        return None
    # drop leading zero-util samples (startup before the first step)
    while util and util[0] == 0:
        util.pop(0)
        mem.pop(0)
    return st.mean(util), max(mem) / 1024


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip", type=int, default=20, help="warm-up iterations to drop")
    ap.add_argument("jobs", nargs="+")
    a = ap.parse_args(argv[1:])
    rows = []
    for job in a.jobs:
        log = job_log(job)
        if not log:
            print(f"{job}: no log yet", file=sys.stderr)
            continue
        text = open(log, errors="replace").read()
        exp = (EXP_RE.search(text) or [None, "?"])[1]
        cfg = (CFG_RE.search(text) or [None, "?"])[1]
        batch = {int(i): float(t) for i, t in BATCH_RE.findall(text)}
        train = {int(i): float(t) for i, t in TRAIN_RE.findall(text)}
        b = [t for i, t in batch.items() if i > a.skip]
        t = [x for i, x in train.items() if i > a.skip]
        g = gpu_stats(exp, job)
        rows.append((job, exp, len(batch), b, t, g, cfg))
    print(f"{'job':>9} {'exp':<24} {'its':>4} {'s/iter med':>10} {'mean':>6} {'p90':>6} "
          f"{'train med':>9} {'gpu util%':>9} {'peak GB':>7}")
    for job, exp, n, b, t, g, cfg in rows:
        if not b:
            print(f"{job:>9} {exp:<24} {n:>4}  (no steady-state iterations yet)")
            continue
        p90 = sorted(b)[int(0.9 * (len(b) - 1))]
        tm = f"{st.median(t):.3f}" if t else "-"
        gu = f"{g[0]:.1f}" if g else "-"
        gm = f"{g[1]:.1f}" if g else "-"
        print(f"{job:>9} {exp:<24} {n:>4} {st.median(b):>10.3f} {st.mean(b):>6.3f} {p90:>6.3f} "
              f"{tm:>9} {gu:>9} {gm:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
