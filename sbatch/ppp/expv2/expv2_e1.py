#!/usr/bin/env python3
"""experiment_v2 EXP 1 driver (md_guides/experiments/experiment_v2.md, EXP 1.1-1.6).

Two independent lanes, each advanced by an idempotent ``tick``:
  lane A: 1.1 pretrain 1k_v1.1 n=1000 (all 796) -> 1.3 finetune completely -> 1.5 test completely
  lane B: 1.2 pretrain 1k_v1.1 n=500           -> 1.4 finetune completely -> 1.6 test completely

``expv2_e1_tick.sh`` runs ``tick`` every ~10 min on a CPU node and resubmits itself.

From the repo root:
  python3 sbatch/ppp/expv2/expv2_e1.py tick        # one step (safe by hand)
  python3 sbatch/ppp/expv2/expv2_e1.py status
  python3 sbatch/ppp/expv2/expv2_e1.py --dry-run tick

Setup (user, 2026-11-08 ~20:00; finish by 9 AM):
  * preset our-noenc-c1k_v1.1 + original_default.toml, FMAPs 15, compile + fp16 raw, L40S,
    account torch_pr_61_general (overlays in setup01/configs/exp/expv2/).
  * pretrain 28k its, peak lr 4e-4, 10% warmup + cosine; val (1k_v1.1 val) + ckpt every 2k.
  * best pretrain ckpt = lowest train-time val loss (val_metrics.csv).
  * finetune 6k its on completely from that ckpt: fresh Adam, peak 1e-4, 5% warmup + cosine
    (seed_finetune_ckpt.py); val (completely val) + ckpt every 1k.
  * best finetune ckpt = lowest val loss -> test on all 7 completely test vols
    (sparse pred_affs pipeline, per-volume evaluate).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import fcntl
import glob
import json
import math
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
METRICS = os.path.join(REPO, "metrics", "ppp")
STATE_DIR = os.path.join(METRICS, "expv2_e1")
STATE = os.path.join(STATE_DIR, "state.json")
STATUS_MD = os.path.join(REPO, "md_guides", "experiments", "experiment_v2_exp1_status.md")
LOGS = os.path.join(REPO, "sbatch", "ppp", "logs")
CFG_DIR = os.path.join(REPO, "PatchPerPix", "experiments", "flylight", "setups", "setup01",
                       "configs", "exp", "expv2")
SIF = "/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

PRESET = "our-noenc-c1k_v1.1"
BASE = "original_default.toml"
ACCOUNT = "torch_pr_61_general"
PRE_ITERS = 28000
FT_ITERS = 6000
FT_LR = 1e-4
FT_WARMUP_FRAC = 0.05
COMMON = ["expv2_e1_common.toml"]
SPARSE_CFG = "pred_sparse_affs.toml"
TEST_PCT = 7            # all 7 completely test volumes at once
EVAL_ENV = {"EVAL_MEM": "64g", "EVAL_PARTITION": "cs", "EVAL_TIME": "02:00:00"}
MAX_RETRIES = 1
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED"}

DRY = False


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def log(state, msg):
    line = f"{now()} {msg}"
    print(line)
    state.setdefault("events", []).append(line)


def run(cmd, env=None):
    print("+", " ".join(cmd))
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       env=None if env is None else {**os.environ, **env})
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({p.returncode}):\n{p.stdout}\n{p.stderr}")
    return p.stdout + p.stderr


def job_states(ids):
    ids = [str(i) for i in ids if i]
    out = {}
    if not ids:
        return out
    q = subprocess.run(["squeue", "-h", "-j", ",".join(ids), "-o", "%F|%T|%r"],
                       capture_output=True, text=True).stdout
    for line in q.splitlines():
        jid, st, reason = line.split("|", 2)
        out.setdefault(jid, (st, reason))
    missing = [i for i in ids if i not in out]
    if missing:
        s = subprocess.run(["sacct", "-n", "-X", "-P", "-j", ",".join(missing), "-o", "JobID,State"],
                           capture_output=True, text=True).stdout
        agg = {}
        for line in s.splitlines():
            jid, st = line.split("|")[:2]
            base, st = jid.split("_")[0], st.split()[0]
            prev = agg.get(base)
            # array jobs: any non-COMPLETED task wins
            if prev is None or (prev == "COMPLETED" and st != "COMPLETED"):
                agg[base] = st
        for i in missing:
            out[i] = (agg.get(i, "UNKNOWN"), "")
    return out


def ckpt_path(exp, it):
    return os.path.join(METRICS, exp, "train", f"train_net_checkpoint_{it}")


# ----------------------------------------------------------------------------- state

def initial_state():
    lanes = {}
    for lane, (pre, ft, te, n, extra) in {
        "A": ("1.1", "1.3", "1.5", 1000, []),
        "B": ("1.2", "1.4", "1.6", 500, ["expv2_e1_n500.toml"]),
    }.items():
        tag = f"n{n}"
        lanes[lane] = {
            "stage": "pre",
            "pre": {"key": pre, "exp": f"expv2-e{pre}-pre-{tag}", "status": "new", "jobs": [],
                    "retries": 0, "overlays": COMMON + ["expv2_e1_pre.toml"] + extra,
                    "max_it": PRE_ITERS, "label": f"pretrain 1k_v1.1 {tag} (n=1000 -> 796)" if n == 1000
                    else f"pretrain 1k_v1.1 {tag}"},
            "ft": {"key": ft, "exp": f"expv2-e{ft}-ft-{tag}-completely", "status": "wait", "jobs": [],
                   "retries": 0, "label": f"finetune completely from {pre}"},
            "test": {"key": te, "status": "wait", "jobs": [], "retries": 0,
                     "label": f"test completely, best {ft} ckpt"},
        }
    return {"lanes": lanes, "events": [], "done": False}


# ----------------------------------------------------------------------------- train

def submit_train(state, arm, time_limit):
    cmd = ["./sbatch/ppp/ppp_submit_train.sh"] + (["--dry-run"] if DRY else []) + [
        PRESET, "--base", BASE, "--id", arm["exp"]]
    for o in arm["overlays"]:
        cmd += ["--config", o]
    cmd += ["--account", ACCOUNT, "--time", time_limit]
    out = run(cmd)
    m = re.search(r"train=(\S+)", out)
    if DRY or not m:
        log(state, f"[dry] would submit train {arm['key']} {arm['exp']}")
        return
    arm["jobs"].append(m.group(1))
    arm["status"] = "training"
    log(state, f"EXP {arm['key']} ({arm['label']}): train job {m.group(1)} exp={arm['exp']}")


def nan_abort(arm):
    for j in arm["jobs"]:
        for p in glob.glob(os.path.join(LOGS, f"ppp_*_train-{j}.out")) + \
                 glob.glob(os.path.join(LOGS, f"ppp_*_train-{j}.err")):
            with open(p, errors="replace") as fh:
                if "Check lr, use_auto_mixed_precision" in fh.read():
                    return True
    return False


def advance_training(state, arm, time_limit):
    jid = arm["jobs"][-1]
    st, _ = job_states([jid]).get(jid, ("UNKNOWN", ""))
    if st in ACTIVE:
        return False
    if os.path.exists(ckpt_path(arm["exp"], arm["max_it"])):
        arm["status"] = "trained"
        log(state, f"EXP {arm['key']}: training done ({st}), ckpt {arm['max_it']}")
        return True
    if nan_abort(arm):
        arm["status"] = "failed"
        log(state, f"EXP {arm['key']}: training aborted on non-finite loss; lane stopped")
        return False
    if arm["retries"] < MAX_RETRIES:
        arm["retries"] += 1
        log(state, f"EXP {arm['key']}: train job {jid} ended {st} without final ckpt; resubmitting (resumes)")
        submit_train(state, arm, time_limit)
    else:
        arm["status"] = "failed"
        log(state, f"EXP {arm['key']}: training failed twice ({st}); lane stopped")
    return False


# ----------------------------------------------------------------------------- selection

def val_rows(exp, min_it=0):
    p = os.path.join(METRICS, exp, "train", "val_metrics.csv")
    rows = []
    if os.path.exists(p):
        with open(p) as fh:
            for r in csv.DictReader(fh):
                try:
                    it, loss = int(r["iteration"]), float(r["loss"])
                except (ValueError, KeyError):
                    continue
                if it > min_it and math.isfinite(loss) and os.path.exists(ckpt_path(exp, it)):
                    rows.append((it, loss))
    return rows


def select_best(state, arm, min_it=0):
    rows = val_rows(arm["exp"], min_it)
    if not rows:
        arm["best_ckpt"] = arm["max_it"]
        arm["best_reason"] = "no val rows with checkpoints; using last checkpoint"
    else:
        it, loss = min(rows, key=lambda r: (r[1], -r[0]))
        arm["best_ckpt"], arm["best_val_loss"] = it, loss
        arm["val_curve"] = rows
        arm["best_reason"] = f"lowest val loss {loss:.4f} of {len(rows)} checkpoints"
    log(state, f"EXP {arm['key']}: best checkpoint {arm['best_ckpt']} ({arm['best_reason']})")


# ----------------------------------------------------------------------------- finetune

def start_finetune(state, pre, ft):
    n = int(pre["best_ckpt"])
    ft["src_ckpt"], ft["start_it"], ft["max_it"] = n, n, n + FT_ITERS
    gen = f"expv2_e1_ft_{ft['exp']}.toml"
    with open(os.path.join(CFG_DIR, gen), "w") as fh:
        fh.write(f"# generated by sbatch/ppp/expv2/expv2_e1.py: EXP {ft['key']} finetunes "
                 f"{pre['exp']} @ {n} for {FT_ITERS} its.\n"
                 f"[training]\nmax_iterations = {n + FT_ITERS}\n\n"
                 f"[optimizer.lr_scheduler]\nstart_iter = {n}\n")
    ft["overlays"] = COMMON + ["expv2_e1_ft.toml", gen]
    if not DRY:
        dst = os.path.join(METRICS, ft["exp"], "train")
        inner = (f"source /ext3/env.sh; conda activate ppp; python sbatch/ppp/expv2/seed_finetune_ckpt.py "
                 f"--src {ckpt_path(pre['exp'], n)} --dst-dir {dst} --iteration {n} --lr {FT_LR} "
                 f"--warmup-frac {FT_WARMUP_FRAC} --ft-iters {FT_ITERS}")
        out = run(["singularity", "exec", "--overlay", "env/ppp.ext3:ro", SIF, "/bin/bash", "-c", inner])
        log(state, f"EXP {ft['key']}: " + out.strip().splitlines()[-1])
    submit_train(state, ft, "04:00:00")


# ----------------------------------------------------------------------------- test

def summary_csv(exp, ckpt):
    root = os.path.join(METRICS, exp, "test", "evaluated", str(ckpt))
    c = glob.glob(os.path.join(root, "**", "summary.csv"), recursive=True)
    return c[0] if c else None


def read_scores(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if r.get("sample", "").startswith("completely/"):
                rows.append(r)

    def mean(col):
        v = [float(r[col]) for r in rows if r.get(col) not in (None, "", "nan")]
        return sum(v) / len(v) if v else float("nan")

    return {"n": len(rows), "avF1": mean("confusion_matrix avFscore"),
            "avg_f1_cov": mean("general avg_f1_cov_score"), "C": mean("general avg_gt_skel_coverage"),
            "FM": mean("general FM"), "FS": mean("general FS"),
            "num_pred": mean("general Num Pred"), "num_gt": mean("general Num GT")}


def submit_test(state, ft, te):
    te["exp"], te["ckpt"] = ft["exp"], ft["best_ckpt"]
    cmd = ["./sbatch/ppp/ppp_submit_test.sh"] + (["--dry-run"] if DRY else []) + [
        "array", PRESET, "--id", ft["exp"], "--split", "test", "--ckpt", str(ft["best_ckpt"]),
        "--base", BASE]
    for o in ft["overlays"] + [SPARSE_CFG]:
        cmd += ["--config", o]
    cmd += ["--pct", str(TEST_PCT), "--account", ACCOUNT]
    out = run(cmd, env=EVAL_ENV)
    ids = re.findall(r"(?:array_job|evaluate)=(\d+)", out)
    ev = re.findall(r"evaluate=(\d+)", out)
    if DRY or not ev:
        log(state, f"[dry] would submit test {te['key']}")
        return
    te["jobs"], te["eval_job"], te["status"] = ids, ev[-1], "testing"
    log(state, f"EXP {te['key']}: test {ft['exp']} @ {ft['best_ckpt']} on completely/test: "
               f"jobs {','.join(ids)} (evaluate {ev[-1]})")


def advance_testing(state, ft, te):
    states = job_states(te["jobs"])
    st, reason = states.get(te["eval_job"], ("UNKNOWN", ""))
    if st in ACTIVE and "DependencyNeverSatisfied" not in reason:
        return
    path = summary_csv(te["exp"], te["ckpt"])
    if st == "COMPLETED" and path:
        te["scores"], te["summary_csv"], te["status"] = read_scores(path), os.path.relpath(path, REPO), "tested"
        s = te["scores"]
        log(state, f"EXP {te['key']}: tested avF1={s['avF1']:.4f} avg_f1_cov={s['avg_f1_cov']:.4f} "
                   f"C={s['C']:.4f} (n={s['n']})")
        return
    active = [j for j, (s, _) in states.items() if s in ACTIVE]
    if active and not DRY:
        subprocess.run(["scancel"] + active)
    if te["retries"] < MAX_RETRIES:
        te["retries"] += 1
        log(state, f"EXP {te['key']}: test pipeline ended {st} {reason}; retrying")
        submit_test(state, ft, te)
    else:
        te["status"] = "failed"
        log(state, f"EXP {te['key']}: test pipeline failed twice")


# ----------------------------------------------------------------------------- tick

def tick_lane(state, lane):
    pre, ft, te = lane["pre"], lane["ft"], lane["test"]
    if lane["stage"] == "pre":
        if pre["status"] == "new":
            submit_train(state, pre, "11:00:00")
        elif pre["status"] == "training" and advance_training(state, pre, "11:00:00"):
            select_best(state, pre)
            lane["stage"] = "ft"
            start_finetune(state, pre, ft)
        if pre["status"] == "failed":
            lane["stage"] = "failed"
    elif lane["stage"] == "ft":
        if ft["status"] == "training" and advance_training(state, ft, "04:00:00"):
            select_best(state, ft, min_it=ft["start_it"])
            lane["stage"] = "test"
            submit_test(state, ft, te)
        if ft["status"] == "failed":
            lane["stage"] = "failed"
    elif lane["stage"] == "test":
        if te["status"] == "testing":
            advance_testing(state, ft, te)
        if te["status"] == "tested":
            lane["stage"] = "done"
        elif te["status"] == "failed":
            lane["stage"] = "failed"


def tick(state):
    if state.get("done"):
        print("done")
        return
    for lane in state["lanes"].values():
        tick_lane(state, lane)
    if all(l["stage"] in ("done", "failed") for l in state["lanes"].values()):
        state["done"] = True
        log(state, "EXP 1 driver finished: " + ", ".join(
            f"lane {k}={l['stage']}" for k, l in state["lanes"].items()))


# ----------------------------------------------------------------------------- status md

def fmt(x, n=4):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{n}f}"


def write_status(state):
    L = ["# experiment_v2 EXP 1 status (auto-generated by `sbatch/ppp/expv2/expv2_e1.py`)", "",
         f"Last update: {now()}. Setup: docstring of `sbatch/ppp/expv2/expv2_e1.py`; overlays in "
         "`setup01/configs/exp/expv2/`.", "",
         "| Step | What | exp | status | jobs | best ckpt (val loss) | test avF1 | avg_f1_cov | C | FM | FS | Num Pred / GT |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, lane in state["lanes"].items():
        for part in ("pre", "ft", "test"):
            a = lane[part]
            s = a.get("scores") or {}
            best = f"{a['best_ckpt']} ({fmt(a.get('best_val_loss'))})" if "best_ckpt" in a else \
                (f"{a['ckpt']}" if "ckpt" in a else "—")
            npgt = f"{fmt(s.get('num_pred'), 0)} / {fmt(s.get('num_gt'), 1)}" if s else "—"
            L.append(f"| {a['key']} | {a['label']} | `{a.get('exp', '—')}` | {a['status']} | "
                     f"{','.join(a['jobs']) or '—'} | {best} | {fmt(s.get('avF1'))} | {fmt(s.get('avg_f1_cov'))} | "
                     f"{fmt(s.get('C'))} | {fmt(s.get('FM'), 1)} | {fmt(s.get('FS'), 1)} | {npgt} |")
    L += ["", "## Event log", ""] + [f"- {e}" for e in state.get("events", [])] + [""]
    with open(STATUS_MD, "w") as fh:
        fh.write("\n".join(L))


@contextlib.contextmanager
def locked_state():
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        state = json.load(open(STATE)) if os.path.exists(STATE) else initial_state()
        try:
            yield state
        finally:
            if not DRY:
                tmp = STATE + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(state, fh, indent=1)
                os.replace(tmp, STATE)
                write_status(state)


def main():
    global DRY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("cmd", choices=["tick", "status"])
    args = ap.parse_args()
    DRY = args.dry_run
    with locked_state() as state:
        if args.cmd == "tick":
            tick(state)
    if os.path.exists(STATUS_MD) and args.cmd == "status":
        print(open(STATUS_MD).read())


if __name__ == "__main__":
    main()
