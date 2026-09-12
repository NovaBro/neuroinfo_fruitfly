"""Collect wall-clock runtimes for the pruning sweep, from SLURM's own job
accounting (sacct). Runs on the bare login/compute node -- sacct is a
host-only SLURM client tool, not present inside the Singularity container --
so this uses only the Python standard library, mirroring
collect_eps_runtimes.py's pattern exactly but generalized to a
(definition, severity) manifest instead of an eps manifest, and to several
job stages (icdm+GW, NBLAST, UGW's 231-block array) instead of just one.

Reads data/MANC/results/pruning_sweep_jobs.json (written as jobs are
submitted), writes data/MANC/results/pruning_sweep_runtimes.csv (columns:
tag, definition, severity, stage, task_id, elapsed_seconds), consumed by
plot_pruning_sweep.py (which runs inside the container).
"""

import csv
import json
import re
import subprocess
from pathlib import Path

MANC_DIR = Path("data") / "MANC"
RESULTS_DIR = MANC_DIR / "results"
JOBS_JSON = RESULTS_DIR / "pruning_sweep_jobs.json"
OUT_CSV = RESULTS_DIR / "pruning_sweep_runtimes.csv"


def elapsed_to_seconds(elapsed):
    # SLURM Elapsed format: [D-]HH:MM:SS
    days = 0
    if "-" in elapsed:
        day_part, elapsed = elapsed.split("-", 1)
        days = int(day_part)
    h, m, s = (int(x) for x in elapsed.split(":"))
    return days * 86400 + h * 3600 + m * 60 + s


def sacct_rows(jobid):
    result = subprocess.run(
        ["sacct", "-j", jobid, "--format=JobID,State,Elapsed", "--noheader", "-P"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.splitlines()


def single_job_elapsed(jobid):
    """For a plain (non-array) job: the top-level JobID row's own elapsed
    time (skip .batch/.extern substeps, which report the same wall time)."""
    for line in sacct_rows(jobid):
        task_id, state, elapsed = line.split("|")
        if task_id == jobid and state == "COMPLETED":
            return elapsed_to_seconds(elapsed)
    return None


def array_job_elapsed(jobid):
    """For a SLURM array job: one elapsed time per completed array task."""
    rows = []
    for line in sacct_rows(jobid):
        task_id, state, elapsed = line.split("|")
        if "." in task_id:
            continue  # skip .batch/.extern substeps
        if not re.match(rf"^{re.escape(jobid)}_\d+$", task_id):
            continue  # skip the array-summary row
        if state != "COMPLETED":
            continue
        rows.append((task_id, elapsed_to_seconds(elapsed)))
    return rows


def main():
    with open(JOBS_JSON) as f:
        jobs = json.load(f)

    rows = []
    for tag, info in jobs.items():
        definition, severity = tag.split("_", 1)
        print(f"{tag}: querying sacct...")

        for stage, key in [("icdm_gw", "icdm_gw_jobid"), ("nblast", "nblast_jobid")]:
            jobid = info.get(key)
            if not jobid:
                continue
            elapsed = single_job_elapsed(jobid)
            if elapsed is not None:
                rows.append({
                    "tag": tag, "definition": definition, "severity": severity,
                    "stage": stage, "task_id": jobid, "elapsed_seconds": elapsed,
                })

        array_jobid = info.get("array_jobid")
        if array_jobid:
            task_rows = array_job_elapsed(array_jobid)
            for task_id, elapsed in task_rows:
                rows.append({
                    "tag": tag, "definition": definition, "severity": severity,
                    "stage": "ugw_block", "task_id": task_id, "elapsed_seconds": elapsed,
                })
            print(f"  {len(task_rows)} completed ugw_block tasks")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["tag", "definition", "severity", "stage", "task_id", "elapsed_seconds"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} runtime rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
