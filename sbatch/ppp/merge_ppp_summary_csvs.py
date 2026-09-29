#!/usr/bin/env python3
"""Merge PPP summary_val.csv + summary_test.csv into a consolidated summary.csv.

Drops mean/sum rows from inputs, concatenates sample rows, then appends
recomputed mean and sum over the combined samples.

    python3 sbatch/ppp/merge_ppp_summary_csvs.py \\
      --val  path/to/summary_val.csv \\
      --test path/to/summary_test.csv \\
      --out  path/to/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


SKIP = frozenset({"mean", "sum"})


def read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as f:
        rdr = csv.reader(f, delimiter=";")
        rows = list(rdr)
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    header, body = rows[0], rows[1:]
    samples = [r for r in body if r and r[0] not in SKIP]
    return header, samples


def write_summary(path: Path, header: list[str], samples: list[list[str]]) -> None:
    ncols = len(header) - 1
    nums: list[list[float]] = []
    for row in samples:
        vals = []
        for i in range(1, ncols + 1):
            try:
                vals.append(float(row[i]) if i < len(row) and row[i] != "" else 0.0)
            except ValueError:
                vals.append(0.0)
        nums.append(vals)

    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter=";", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        w.writerow(header)
        for row in samples:
            # pad/truncate to header width
            out = [row[0]] + [(row[i] if i < len(row) else "") for i in range(1, ncols + 1)]
            w.writerow(out)
        if nums:
            n = len(nums)
            means = [sum(nums[r][c] for r in range(n)) / n for c in range(ncols)]
            sums = [sum(nums[r][c] for r in range(n)) for c in range(ncols)]
            w.writerow(["mean"] + means)
            w.writerow(["sum"] + sums)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val", required=True, type=Path, help="summary_val.csv")
    p.add_argument("--test", required=True, type=Path, help="summary_test.csv")
    p.add_argument("--out", required=True, type=Path, help="output summary.csv")
    args = p.parse_args()

    for path in (args.val, args.test):
        if not path.is_file():
            print(f"Error: missing {path}", file=sys.stderr)
            return 1

    h_val, s_val = read_rows(args.val)
    h_test, s_test = read_rows(args.test)
    if h_val != h_test:
        print(
            f"Error: header mismatch\n  val={h_val}\n  test={h_test}",
            file=sys.stderr,
        )
        return 1

    combined = s_val + s_test
    if not combined:
        print("Error: no sample rows to merge", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_summary(args.out, h_val, combined)
    print(
        f"merged {len(s_val)} val + {len(s_test)} test samples → {args.out} "
        f"({len(combined)} rows + mean/sum)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
