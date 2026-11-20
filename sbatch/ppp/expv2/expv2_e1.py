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
  * pretrain 20k its / 8 h limit (2026-11-08 21:10: was 28k / 11 h; long queue waits), peak lr 4e-4,
    10% warmup + cosine; val (1k_v1.1 val) + ckpt every 2k.
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
PRE_ITERS = 20000
PRE_TIME = "08:00:00"
FT_ITERS = 6000
FT_TIME = "04:00:00"
FT_LR = 1e-4
FT_WARMUP_FRAC = 0.05
COMMON = ["expv2_e1_common.toml"]
SPARSE_CFG = "pred_sparse_affs.toml"
TEST_PCT = 7            # all 7 completely test volumes at once
EVAL_ENV = {"EVAL_MEM": "96g", "EVAL_PARTITION": "cs", "EVAL_TIME": "01:30:00"}
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
    submit_train(state, ft, FT_TIME)


# ----------------------------------------------------------------------------- test

def summary_csv(exp, ckpt):
    root = os.path.join(METRICS, exp, "test", "evaluated", str(ckpt))
    c = glob.glob(os.path.join(root, "**", "summary.csv"), recursive=True)
    return c[0] if c else None


# FISBe leaderboard metrics (md_guides/web/web_fisbe_metrics.md) -> summary.csv column.
# S = 0.5 * avF1 + 0.5 * C (= general.avg_f1_cov_score).
FISBE_COLS = {"S": "general avg_f1_cov_score", "avF1": "confusion_matrix avFscore19",
              "C": "general avg_gt_skel_coverage", "clDiceTP": "general avg_TP_05_cldice",
              "tp": "general TP_05_rel", "FS": "general FS", "FM": "general FM",
              "num_pred": "general Num Pred", "num_gt": "general Num GT"}


def score_rows(path):
    """Per-volume rows {sample, S, avF1, ...} of a summary.csv (mean/sum rows dropped)."""
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if not r.get("sample", "").startswith("completely/"):
                continue
            row = {"sample": r["sample"].split("/", 1)[1]}
            for k, col in FISBE_COLS.items():
                v = r.get(col)
                if v in (None, "", "nan") and k == "avF1":
                    v = r.get("confusion_matrix avFscore")
                row[k] = float(v) if v not in (None, "", "nan") else float("nan")
            out.append(row)
    return out


def read_scores(path):
    rows = score_rows(path)

    def mean(k):
        v = [r[k] for r in rows if math.isfinite(r[k])]
        return sum(v) / len(v) if v else float("nan")

    s = {"n": len(rows), **{k: mean(k) for k in FISBE_COLS}}
    s["avg_f1_cov"] = s["S"]  # old key name
    return s


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
        log(state, f"EXP {te['key']}: tested S={s['S']:.4f} avF1={s['avF1']:.4f} C={s['C']:.4f} "
                   f"clDiceTP={s['clDiceTP']:.4f} tp={s['tp']:.4f} (n={s['n']})")
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
            submit_train(state, pre, PRE_TIME)
        elif pre["status"] == "training" and advance_training(state, pre, PRE_TIME):
            select_best(state, pre)
            lane["stage"] = "ft"
            start_finetune(state, pre, ft)
        if pre["status"] == "failed":
            lane["stage"] = "failed"
    elif lane["stage"] == "ft":
        if ft["status"] == "training" and advance_training(state, ft, FT_TIME):
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


# Setup section: read from each run's merged metrics/ppp/<exp>/config.toml (what the jobs actually ran
# with), so it stays correct if an overlay changes. Settings equal across runs go in a "shared" table.

def load_toml(path):
    try:
        import tomllib
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except ModuleNotFoundError:  # Python < 3.11
        import toml
        return toml.load(path)


def run_config(exp):
    p = os.path.join(METRICS, exp or "", "config.toml")
    return load_toml(p) if exp and os.path.exists(p) else None


def cfg(c, dotted):
    for k in dotted.split("."):
        if not isinstance(c, dict) or k not in c:
            return None
        c = c[k]
    return c


def shp(v):
    return f"{v[0]}³" if len(v) == 3 and len(set(v)) == 1 else "×".join(map(str, v))


def lr_fmt(x):
    m, e = f"{float(x):.1e}".split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"


def hours(t):
    h, m, _ = map(int, t.split(":"))
    return f"{h} h" + (f" {m} min" if m else "")


def data_desc(c, split):
    ds = cfg(c, f"data.{split}_datasets") or []
    total = sum(len(glob.glob(os.path.join(cfg(c, "data.fisbe_root"), d, split, "*.zarr"))) for d in ds)
    cap = cfg(c, f"data.max_{split}_samples") if split != "test" else None
    used = min(total, cap) if cap else total
    return f"{', '.join(ds)} {split}: {used} vols" + (f" (first {used} of {total})" if used < total else "")


def sched_desc(c):
    s = cfg(c, "optimizer.lr_scheduler") or {}
    if s.get("lr_scheduler") != "WarmupCosine":
        return s.get("lr_scheduler") or "constant"
    # mirrors setup01/train.py (WarmupCosine branch)
    start, end = int(s.get("start_iter", 0)), int(cfg(c, "training.max_iterations"))
    total = int(s.get("total_iters", end - start))
    warm = int(s["warmup_iters"]) if "warmup_iters" in s else \
        int(round(float(s.get("warmup_frac", 0.05)) * total))
    floor = float(s.get("min_lr_ratio", 0.0)) * float(cfg(c, "optimizer.optimizer.lr"))
    return (f"WarmupCosine: linear warmup {warm} its ({warm / total:.0%}; it {start} → {start + warm}), "
            f"then cosine to {lr_fmt(floor) if floor else '0'} at it {start + total}")


def opt_desc(c):
    o = dict(cfg(c, "optimizer.optimizer") or {})
    name = o.pop("optimizer", "?")
    o.pop("lr", None)
    extra = ", ".join(f"{k} = {v}" for k, v in o.items())
    return f"{name} ({extra})" if extra else f"{name} (torch defaults, no weight decay)"


TRAIN_FIELDS = [
    ("exp", lambda c, x: f"`{x['arm']['exp']}`"),
    ("init", lambda c, x: "from scratch" if x["part"] == "pre" else
        f"EXP {x['lane']['pre']['key']} @ it {x['arm']['src_ckpt']}; fresh Adam + GradScaler "
        "(`seed_finetune_ckpt.py`)"),
    ("train data", lambda c, x: data_desc(c, "train")),
    ("val data", lambda c, x: data_desc(c, "val")),
    ("iterations", lambda c, x: (lambda s, e: f"{s} → {e} ({e - s} its)")(
        int(cfg(c, "optimizer.lr_scheduler.start_iter") or 0), int(cfg(c, "training.max_iterations")))),
    ("optimizer", lambda c, x: opt_desc(c)),
    ("peak lr", lambda c, x: lr_fmt(cfg(c, "optimizer.optimizer.lr"))),
    ("lr schedule", lambda c, x: sched_desc(c)),
    ("batch size", lambda c, x: str(cfg(c, "training.batch_size"))),
    ("checkpoint / val every", lambda c, x:
        f"{cfg(c, 'training.checkpoints')} / {cfg(c, 'training.val_log_step')} its"),
    ("network", lambda c, x: f"{cfg(c, 'model.network_style')}, train_code = "
        f"{str(cfg(c, 'model.train_code')).lower()}, {cfg(c, 'model.activation')}"),
    ("fmaps", lambda c, x: f"num_fmaps {cfg(c, 'model.num_fmaps')}, inc {cfg(c, 'model.fmap_inc_factors')}, "
        f"dec {cfg(c, 'model.fmap_dec_factors')}"),
    ("levels", lambda c, x: f"{len(cfg(c, 'model.downsample_factors'))} × downsample "
        f"{shp(cfg(c, 'model.downsample_factors')[0])}, kernel {cfg(c, 'model.kernel_size')}, "
        f"{cfg(c, 'model.num_repetitions')} convs/level, upsampling {cfg(c, 'model.upsampling')}"),
    ("train input", lambda c, x: f"{shp(cfg(c, 'model.train_input_shape_valid'))} "
        f"({cfg(c, 'model.train_padding')})"),
    ("patches", lambda c, x: f"{shp(cfg(c, 'model.patchshape'))}, {cfg(c, 'model.patch_activation')}, "
        f"max_num_inst {cfg(c, 'model.max_num_inst')}"),
    ("loss", lambda c, x: f"add_affinities = {cfg(c, 'training.add_affinities')}, "
        f"mask_bg_weight = {cfg(c, 'training.mask_bg_weight')}, "
        f"fast_affinities = {str(cfg(c, 'training.fast_affinities')).lower()}"),
    ("precision / speed", lambda c, x: f"AMP {str(cfg(c, 'training.use_auto_mixed_precision')).lower()}, "
        f"compile {str(cfg(c, 'training.compile')).lower()}, "
        f"raw_transport_fp16 {str(cfg(c, 'training.raw_transport_fp16')).lower()}"),
    ("augmentation", lambda c, x: f"elastic {str(cfg(c, 'training.augmentation.elastic')).lower()}, "
        f"intensity scale {cfg(c, 'training.augmentation.intensity.scale')} "
        f"shift {cfg(c, 'training.augmentation.intensity.shift')}, "
        f"permute p {cfg(c, 'training.probability_permute')}, hue p {cfg(c, 'training.probability_hue')}"),
    ("sampling", lambda c, x: f"p_fg {cfg(c, 'training.sampling.probability_fg')}, "
        f"p_random {cfg(c, 'training.sampling.probability_random')}, "
        f"min_masked {cfg(c, 'training.sampling.min_masked')}"),
    ("GPU / walltime", lambda c, x: f"1× L40S, {hours(PRE_TIME if x['part'] == 'pre' else FT_TIME)}"),
    ("overlays", lambda c, x: ", ".join(f"`{o}`" for o in x["arm"].get("overlays", [])) or "—"),
]

TEST_FIELDS = [
    ("model", lambda c, x: f"EXP {x['lane']['ft']['key']} @ it {x['arm'].get('ckpt', '—')} (lowest val loss)"),
    ("test data", lambda c, x: data_desc(c, "test")),
    ("prediction", lambda c, x: f"input {shp(cfg(c, 'model.test_input_shape_valid'))}, "
        f"batch {cfg(c, 'prediction.batch_size')}, sparse_affs "
        f"{str(cfg(c, 'prediction.sparse_affs')).lower()} (fg thresh {cfg(c, 'prediction.sparse_affs_fg_thresh')})"),
    ("vote_instances", lambda c, x: f"patch_threshold {cfg(c, 'vote_instances.patch_threshold')}, "
        f"fc_threshold {cfg(c, 'vote_instances.fc_threshold')}, "
        f"ignore_small_comps {cfg(c, 'vote_instances.ignore_small_comps')}, "
        f"mws {str(cfg(c, 'vote_instances.mws')).lower()}, blockwise chunks "
        f"{shp(cfg(c, 'vote_instances.chunksize'))}"),
    ("evaluation", lambda c, x: f"{cfg(c, 'evaluation.metric')}, {cfg(c, 'evaluation.localization_criterion')} "
        f"/ {cfg(c, 'evaluation.assignment_strategy')}, "
        f"remove_small_components {cfg(c, 'evaluation.remove_small_components')}"),
    ("overlays", lambda c, x: ", ".join(f"`{o}`" for o in
        x["lane"]["ft"].get("overlays", []) + [SPARSE_CFG])),
]


def setup_tables(cols, fields):
    """cols: [(title, config or None, ctx)]. Rows equal across all columns with a config -> shared."""
    def cell(fn, c, x):
        try:
            return str(fn(c or {}, x)).replace("|", "\\|")
        except Exception:
            return "—"
    known = [i for i, (_, c, _) in enumerate(cols) if c is not None]
    shared, per = [], []
    for label, fn in fields:
        vals = [cell(fn, c, x) for _, c, x in cols]
        if len(known) > 1 and len({vals[i] for i in known}) == 1:
            shared.append(f"| {label} | {vals[known[0]]} |")
        else:
            per.append(f"| {label} | " + " | ".join(vals) + " |")
    out = []
    if per:
        out += ["| | " + " | ".join(t for t, _, _ in cols) + " |", "|---" * (len(cols) + 1) + "|"] + per + [""]
    if shared:
        out += [f"Same for {', '.join(cols[i][0] for i in known)}:", "", "| Setting | Value |", "|---|---|"] + \
            shared + [""]
    return out


def setup_section(state):
    lanes = list(state["lanes"].values())
    train_cols = []
    for part, kind in (("pre", "pretrain"), ("ft", "finetune")):
        for lane in lanes:
            a = lane[part]
            ok = part == "pre" or "src_ckpt" in a
            train_cols.append((f"{a['key']} {kind}", run_config(a["exp"]) if ok else None,
                               {"arm": a, "lane": lane, "part": part}))
    L = ["## Setup", "",
         f"Preset `{PRESET}` + base `{BASE}`, account `{ACCOUNT}`. Values are read from each run's merged "
         "`metrics/ppp/<exp>/config.toml` (overlays in `setup01/configs/exp/expv2/`). Best checkpoint = "
         "lowest train-time val loss (`<exp>/train/val_metrics.csv`).", "",
         "### Training", ""] + setup_tables(train_cols, TRAIN_FIELDS)
    # The test jobs rewrite the finetune exp's config.toml with the test overlays, so only read it
    # for test settings once testing has started.
    test_cols = [(f"{lane['test']['key']} test", run_config(lane["ft"]["exp"]),
                  {"arm": lane["test"], "lane": lane, "part": "test"})
                 for lane in lanes if lane["test"]["status"] in ("testing", "tested", "failed")]
    if test_cols:
        L += ["### Test", ""] + setup_tables(test_cols, TEST_FIELDS)
    return L


def write_status(state):
    L = ["# experiment_v2 EXP 1 status (auto-generated by `sbatch/ppp/expv2/expv2_e1.py`)", "",
         f"Last update: {now()}. Full setup in [Setup](#setup) below; driver logic in the docstring of "
         "`sbatch/ppp/expv2/expv2_e1.py`.", "",
         "| Step | What | exp | status | jobs | best ckpt (val loss) | S | avF1 | C | clDiceTP | tp | FS | FM | "
         "nPred / nGT |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    per_volume = []
    for k, lane in state["lanes"].items():
        for part in ("pre", "ft", "test"):
            a = lane[part]
            s = a.get("scores") or {}
            csv_path = os.path.join(REPO, a["summary_csv"]) if a.get("summary_csv") else None
            if csv_path and os.path.exists(csv_path):  # re-read: picks up metrics added after scoring
                s = read_scores(csv_path)
                per_volume.append((a, score_rows(csv_path)))
            best = f"{a['best_ckpt']} ({fmt(a.get('best_val_loss'))})" if "best_ckpt" in a else \
                (f"{a['ckpt']}" if "ckpt" in a else "—")
            npgt = f"{fmt(s.get('num_pred'), 0)} / {fmt(s.get('num_gt'), 1)}" if s else "—"
            L.append(f"| {a['key']} | {a['label']} | `{a.get('exp', '—')}` | {a['status']} | "
                     f"{','.join(a['jobs']) or '—'} | {best} | {('**' + fmt(s['S']) + '**') if s else '—'} | {fmt(s.get('avF1'))} | "
                     f"{fmt(s.get('C'))} | {fmt(s.get('clDiceTP'))} | {fmt(s.get('tp'))} | {fmt(s.get('FS'), 2)} | "
                     f"{fmt(s.get('FM'), 2)} | {npgt} |")
    L += ["", "Test metrics are the FISBe ones, mean over the test volumes: **S** = 0.5·avF1 + 0.5·C; **avF1** = "
          "F1 averaged over clDice thresholds 0.1–0.9; **C** = mean GT skeleton coverage; **clDiceTP** = mean "
          "clDice of true positives (clDice ≥ 0.5); **tp** = fraction of GT matched at clDice ≥ 0.5; **FS** / "
          "**FM** = false splits / merges per volume; **nPred / nGT** = instances per volume."]
    if per_volume:
        L += ["", "## Per-volume test results", ""]
        for a, rows in per_volume:
            L += [f"**{a['key']}** (`{a.get('exp')}` @ {a.get('ckpt')}; `{a['summary_csv']}`)", "",
                  "| volume | S | avF1 | C | clDiceTP | tp | FS | FM | nPred | nGT |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            L += [f"| {r['sample']} | {fmt(r['S'])} | {fmt(r['avF1'])} | {fmt(r['C'])} | {fmt(r['clDiceTP'])} | "
                  f"{fmt(r['tp'])} | {fmt(r['FS'], 0)} | {fmt(r['FM'], 0)} | {fmt(r['num_pred'], 0)} | "
                  f"{fmt(r['num_gt'], 0)} |" for r in rows] + [""]
    try:
        L += [""] + setup_section(state)
    except Exception as e:  # never let the status page break a tick
        L += ["", "## Setup", "", f"(setup section failed: {type(e).__name__}: {e})", ""]
    L += ["## Event log", ""] + [f"- {e}" for e in state.get("events", [])] + [""]
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
