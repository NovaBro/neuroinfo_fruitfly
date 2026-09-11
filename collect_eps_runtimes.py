"""Collect per-block-pair wall-clock runtimes for the UGW eps sweep, from
SLURM's own job accounting (sacct). Runs on the bare login/compute node --
sacct is a host-only SLURM client tool, not present inside the Singularity
container that has pandas/matplotlib/cajal -- so this uses only the Python
standard library (no pandas/numpy needed here) and is invoked outside the
container, unlike every other script in this pipeline.

Writes data/MANC/results/eps_sweep_runtimes.csv (columns: eps, task_id,
elapsed_seconds), consumed by plot_eps_sweep.py (which runs inside the
container).
"""

import csv
import json
import re
import subprocess
from pathlib import Path

MANC_DIR = Path("data") / "MANC"
RESULTS_DIR = MANC_DIR / "results"
JOBS_JSON = RESULTS_DIR / "eps_sweep_jobs.json"
OUT_CSV = RESULTS_DIR / "eps_sweep_runtimes.csv"


def elapsed_to_seconds(elapsed):
    # SLURM Elapsed format: [D-]HH:MM:SS
    days = 0
    if "-" in elapsed:
        day_part, elapsed = elapsed.split("-", 1)
        days = int(day_part)
    h, m, s = (int(x) for x in elapsed.split(":"))
    return days * 86400 + h * 3600 + m * 60 + s


def main():
    with open(JOBS_JSON) as f:
        jobs = json.load(f)

    rows = []
    for eps, info in sorted(jobs.items(), key=lambda kv: float(kv[0])):
        jobid = info["array_jobid"]
        print(f"eps={eps}: querying sacct for array job {jobid}")
        result = subprocess.run(
            ["sacct", "-j", jobid, "--format=JobID,State,Elapsed", "--noheader", "-P"],
            capture_output=True, text=True, check=True,
        )
        n_rows = 0
        for line in result.stdout.splitlines():
            task_id, state, elapsed = line.split("|")
            if "." in task_id:
                continue  # skip .batch/.extern substeps
            if not re.match(rf"^{re.escape(jobid)}_\d+$", task_id):
                continue  # skip the array-summary row (e.g. "JOBID_[..]")
            if state != "COMPLETED":
                continue  # only successful blocks have a meaningful runtime
            rows.append({"eps": eps, "task_id": task_id, "elapsed_seconds": elapsed_to_seconds(elapsed)})
            n_rows += 1
        print(f"  {n_rows} completed tasks")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["eps", "task_id", "elapsed_seconds"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} runtime rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
