#!/usr/bin/env python3
"""experiment_v4 driver (md_guides/experiments/experiment_v4.md): simple_v1 synthetic scaling.

Five lanes, each advanced by an idempotent ``tick``:
  B0    EXP 4.0  train completely from scratch (50k its) -> select -> test simple_v1 + completely
  N1k   EXP 4.1  pretrain simple_v1 first 1k  (50k its) -> select -> test simple_v1
                 4.1F LoRA (encoder, r4) finetune completely 6k its -> select -> test simple_v1, then completely
  N5k / N10k / N50k  EXP 4.2 / 4.3 / 4.4: same with 5k / 10k / 50k samples

Selection (user, 2026-11-18): completely val only, highest mean(fg_dice, fg_precision, fg_recall); NaN counts as 0;
ties -> lower completely val loss, then the later iteration. Pretrain arms read val_metrics_completely.csv
(completely is their extra val set); baseline and finetunes read val_metrics.csv. The synthetic val set is
recorded at the selected iteration but never selects.

Tests of one exp at one checkpoint share metrics/ppp/<exp>/test/evaluated/<ckpt>/.../summary.csv, so a lane's
tests of the same exp run one after the other and each summary is copied to
metrics/ppp/expv4/summaries/<exp>@<ckpt>_<dataset>.csv when it completes.

From the repo root:
  python3 sbatch/ppp/expv4/expv4.py --dry-run tick   # print what would be submitted
  python3 sbatch/ppp/expv4/expv4.py tick             # one step (safe by hand)
  python3 sbatch/ppp/expv4/expv4.py status
  python3 sbatch/ppp/expv4/expv4.py hold --lane N50k      # keep a lane from starting (before or between ticks)
  python3 sbatch/ppp/expv4/expv4.py release --lane N50k   # let it start at the next tick
``expv4_tick.sh`` runs ``tick`` every ~10 min on a CPU node and resubmits itself (stop: scancel -n expv4_tick).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "sbatch", "ppp", "expv2"))
import expv2_e1 as E  # noqa: E402  (shared helpers: run, job_states, summary_csv, ckpt_path, ...)

STATE_DIR = os.path.join(E.METRICS, "expv4")
STATE = os.path.join(STATE_DIR, "state.json")
SUMMARY_DIR = os.path.join(STATE_DIR, "summaries")
STATUS_MD = os.path.join(REPO, "md_guides", "experiments", "expv4_status.md")
CFG_DIR = os.path.join(REPO, "PatchPerPix", "experiments", "flylight", "setups", "setup01", "configs", "exp", "expv4")
SIF = E.SIF
PRESET = E.PRESET
BASE = E.BASE
# 2026-11-19 (user): new submissions alternate over these (tandon_priority's fair-share was the lowest).
# Jobs submitted earlier stay on torch_pr_61_tandon_priority.
ACCOUNTS = ["torch_pr_61_tandon_advanced", "torch_pr_61_general"]
COMMON = ["expv2_e1_common.toml"]
PRE_ITERS = 50000
PRE_TIME = "24:00:00"
FT_ITERS = 6000
FT_LR = 5e-5
FT_WARMUP_FRAC = 0.10
FT_TIME = "04:00:00"
SPARSE_CFG = "pred_sparse_affs.toml"
TESTS = {   # dataset -> (test overlay, eval overlay, array concurrency)
    "simple_v1": ("expv4_test_simple_v1.toml", "eval_rm500.toml", 5),
    "completely": ("expv4_test_completely.toml", "eval_rm1500.toml", 7),
}
SELECT_METRICS = ("fg_dice", "fg_precision", "fg_recall")
EVAL_ENV = E.EVAL_ENV
MAX_RETRIES = 1     # failures without progress
MAX_CHAIN = 6       # timeouts with progress (resume from the last checkpoint)
ACTIVE = E.ACTIVE
ARMS = [("N1k", "4.1", "1k", 1000), ("N5k", "4.2", "5k", 5000), ("N10k", "4.3", "10k", 10000),
        ("N50k", "4.4", "50k", 50000)]

DRY = False


def log(state, msg):
    E.log(state, msg)


def latest_ckpt(exp):
    its = []
    for p in glob.glob(os.path.join(E.METRICS, exp, "train", "train_net_checkpoint_*")):
        m = re.search(r"_(\d+)$", p)
        if m:
            its.append(int(m.group(1)))
    return max(its) if its else -1


def reached_max(exp, max_it):
    p = os.path.join(E.METRICS, exp, "run.log")
    if not os.path.exists(p):
        return False
    with open(p, errors="replace") as fh:
        return f"Train process: iteration={max_it} " in fh.read()


# ----------------------------------------------------------------------------- state

def _arm(key, exp, label, **kw):
    return {"key": key, "exp": exp, "label": label, "status": "new", "jobs": [], "retries": 0, "chain": 0, **kw}


def _test(key, src, dataset):
    return {"key": key, "src": src, "dataset": dataset, "status": "wait", "jobs": [], "retries": 0,
            "label": f"test {dataset}" + (" (first 5)" if dataset == "simple_v1" else " (7)")}


def initial_state():
    lanes = {"B0": {
        "stage": "pre",
        "pre": _arm("4.0", "expv4-b0-completely", "completely from scratch, 50k its",
                    overlays=COMMON + ["expv4_pre.toml", "expv4_b0.toml"], max_it=PRE_ITERS, time=PRE_TIME,
                    val_csv="val_metrics.csv", synth_csv="val_metrics_simple_v1.csv"),
        "ft": None,
        "tests": [_test("4.0-Ts", "pre", "simple_v1"), _test("4.0-Tc", "pre", "completely")]}}
    for lane, key, tag, n in ARMS:
        lanes[lane] = {
            "stage": "pre",
            "pre": _arm(key, f"expv4-pre-n{tag}", f"pretrain simple_v1 first {n}, 50k its",
                        overlays=COMMON + ["expv4_pre.toml", "dsexp_simple_v1.toml", f"expv4_n{tag}.toml"],
                        max_it=PRE_ITERS, time=PRE_TIME,
                        val_csv="val_metrics_completely.csv", synth_csv="val_metrics.csv"),
            "ft": _arm(f"{key}F", f"expv4-ft-n{tag}", "LoRA r4 encoder finetune completely, 6k its",
                       status="wait", val_csv="val_metrics.csv", synth_csv="val_metrics_simple_v1.csv"),
            "tests": [_test(f"{key}-Ts", "pre", "simple_v1"), _test(f"{key}F-Ts", "ft", "simple_v1"),
                      _test(f"{key}F-Tc", "ft", "completely")]}
    return {"lanes": lanes, "events": [], "done": False}


# ----------------------------------------------------------------------------- train

def next_account(state):
    i = state.get("account_rr", 0)
    state["account_rr"] = i + 1
    return ACCOUNTS[i % len(ACCOUNTS)]


def submit_train(state, arm, time_limit):
    cmd = ["./sbatch/ppp/ppp_submit_train.sh"] + (["--dry-run"] if DRY else []) + [
        PRESET, "--base", BASE, "--id", arm["exp"]]
    for o in arm["overlays"]:
        cmd += ["--config", o]
    account = next_account(state)
    cmd += ["--account", account, "--time", time_limit]
    out = E.run(cmd)
    m = re.search(r"train=(\d+)", out)
    if DRY or not m:
        log(state, f"[dry] would submit train {arm['key']} {arm['exp']}: {' '.join(cmd)}")
        return
    arm["jobs"].append(m.group(1))
    arm["account"] = account
    arm["status"] = "training"
    arm["ckpt_at_submit"] = latest_ckpt(arm["exp"])
    log(state, f"EXP {arm['key']} ({arm['label']}): train job {m.group(1)} exp={arm['exp']}")


def advance_training(state, arm, time_limit):
    jid = arm["jobs"][-1]
    st, _ = E.job_states([jid]).get(jid, ("UNKNOWN", ""))
    if st in ACTIVE:
        return False
    last = latest_ckpt(arm["exp"])
    if last >= arm["max_it"] or reached_max(arm["exp"], arm["max_it"]):
        arm["status"] = "trained"
        log(state, f"EXP {arm['key']}: training done ({st}), iteration {arm['max_it']} reached, last ckpt {last}")
        return True
    if E.nan_abort(arm):
        arm["status"] = "failed"
        log(state, f"EXP {arm['key']}: training aborted on non-finite loss; lane stopped")
        return False
    if last > arm.get("ckpt_at_submit", -1) and arm["chain"] < MAX_CHAIN:
        arm["chain"] += 1
        log(state, f"EXP {arm['key']}: job {jid} ended {st} at ckpt {last}/{arm['max_it']}; "
                   f"resubmitting to resume (chain {arm['chain']}/{MAX_CHAIN})")
        submit_train(state, arm, time_limit)
    elif arm["retries"] < MAX_RETRIES:
        arm["retries"] += 1
        log(state, f"EXP {arm['key']}: job {jid} ended {st} without new checkpoint; retrying")
        submit_train(state, arm, time_limit)
    else:
        arm["status"] = "failed"
        log(state, f"EXP {arm['key']}: training failed ({st}, ckpt {last}); lane stopped")
    return False


# ----------------------------------------------------------------------------- selection

def _f(r, k):
    try:
        v = float(r.get(k, "nan"))
    except (TypeError, ValueError):
        v = float("nan")
    return v


def score_of(r):
    """mean(fg_dice, fg_precision, fg_recall) with NaN -> 0 (experiment_v4 selection score)."""
    vals = [_f(r, k) for k in SELECT_METRICS]
    return sum(v if math.isfinite(v) else 0.0 for v in vals) / len(vals)


def val_table(exp, csv_name, min_it=0, need_ckpt=True):
    """iteration -> {score, loss, fg_dice, fg_precision, fg_recall} (last row per iteration wins)."""
    p = os.path.join(E.METRICS, exp, "train", csv_name)
    out = {}
    if not os.path.exists(p):
        return out
    with open(p) as fh:
        for r in csv.DictReader(fh):
            try:
                it = int(r["iteration"])
            except (ValueError, KeyError):
                continue
            if it <= min_it or (need_ckpt and not os.path.exists(E.ckpt_path(exp, it))):
                continue
            out[it] = {"score": score_of(r), "loss": _f(r, "loss"), **{k: _f(r, k) for k in SELECT_METRICS}}
    return out


def select_best(state, arm, min_it=0):
    rows = val_table(arm["exp"], arm["val_csv"], min_it)
    if not rows:
        arm["best_ckpt"] = arm["max_it"]
        arm["best_reason"] = "no completely val rows with checkpoints; using the last iteration"
    else:
        def key(it):
            r = rows[it]
            loss = r["loss"] if math.isfinite(r["loss"]) else float("inf")
            return (r["score"], -loss, it)
        it = max(rows, key=key)
        r = rows[it]
        arm["best_ckpt"], arm["best"] = it, r
        arm["val_curve"] = [[i, rows[i]["score"], rows[i]["loss"]] for i in sorted(rows)]
        arm["best_reason"] = (f"highest completely val score {r['score']:.4f} (dice {E.fmt(r['fg_dice'])}, "
                              f"precision {E.fmt(r['fg_precision'])}, recall {E.fmt(r['fg_recall'])}; "
                              f"loss {E.fmt(r['loss'])}) of {len(rows)} checkpoints")
        synth = val_table(arm["exp"], arm["synth_csv"], min_it, need_ckpt=False)
        arm["synth_at_best"] = synth.get(it)
        arm["synth_curve"] = [[i, synth[i]["score"], synth[i]["loss"]] for i in sorted(synth)]
    log(state, f"EXP {arm['key']}: best checkpoint {arm['best_ckpt']} ({arm['best_reason']})")


# ----------------------------------------------------------------------------- finetune

def start_finetune(state, lane):
    pre, ft = lane["pre"], lane["ft"]
    n = int(pre["best_ckpt"])
    ft["src_ckpt"], ft["start_it"], ft["max_it"] = n, n, n + FT_ITERS
    gen = f"expv4_ft_{ft['exp']}.toml"
    if not DRY:
        with open(os.path.join(CFG_DIR, gen), "w") as fh:
            fh.write(f"# generated by sbatch/ppp/expv4/expv4.py: EXP {ft['key']} finetunes {pre['exp']} @ {n} "
                     f"for {FT_ITERS} its (peak lr {FT_LR} and LoRA in expv4_ft.toml).\n"
                     f"[training]\nmax_iterations = {n + FT_ITERS}\n\n"
                     f"[optimizer.lr_scheduler]\nstart_iter = {n}\n")
    ft["overlays"] = COMMON + ["expv4_ft.toml", gen]
    if not DRY:
        dst = os.path.join(E.METRICS, ft["exp"], "train")
        inner = (f"source /ext3/env.sh; conda activate ppp; python sbatch/ppp/expv2/seed_finetune_ckpt.py "
                 f"--src {E.ckpt_path(pre['exp'], n)} --dst-dir {dst} --iteration {n} --lr {FT_LR} "
                 f"--warmup-frac {FT_WARMUP_FRAC} --ft-iters {FT_ITERS}")
        out = E.run(["singularity", "exec", "--overlay", "env/ppp.ext3:ro", SIF, "/bin/bash", "-c", inner])
        log(state, f"EXP {ft['key']}: " + out.strip().splitlines()[-1])
    submit_train(state, ft, FT_TIME)


# ----------------------------------------------------------------------------- test

def score_rows(path, dataset):
    """Per-volume rows of a summary.csv for one dataset (E.score_rows only keeps completely/)."""
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if not r.get("sample", "").startswith(dataset + "/"):
                continue
            row = {"sample": r["sample"].split("/", 1)[1]}
            for k, col in E.FISBE_COLS.items():
                v = r.get(col)
                if v in (None, "", "nan") and k == "avF1":
                    v = r.get("confusion_matrix avFscore")
                row[k] = float(v) if v not in (None, "", "nan") else float("nan")
            out.append(row)
    return out


def read_scores(path, dataset):
    rows = score_rows(path, dataset)

    def mean(k):
        v = [r[k] for r in rows if math.isfinite(r[k])]
        return sum(v) / len(v) if v else float("nan")

    return {"n": len(rows), **{k: mean(k) for k in E.FISBE_COLS}}


def submit_test(state, lane, te, stage="all"):
    """stage "label": predictions already exist; run label graph -> label -> evaluate only."""
    arm = lane[te["src"]]
    exp, ckpt = arm["exp"], arm["best_ckpt"]
    test_cfg, eval_cfg, pct = TESTS[te["dataset"]]
    cmd = ["./sbatch/ppp/ppp_submit_test.sh"] + (["--dry-run"] if DRY else []) + [
        "array", PRESET, "--id", exp, "--split", "test", "--ckpt", str(ckpt), "--base", BASE, "--eval", eval_cfg,
        "--stage", stage]
    for o in arm["overlays"] + [SPARSE_CFG, test_cfg]:
        cmd += ["--config", o]
    account = next_account(state)
    cmd += ["--pct", str(pct), "--account", account]
    out = E.run(cmd, env=EVAL_ENV)
    ids = re.findall(r"(?:array_job|evaluate)=(\d+)", out)
    ev = re.findall(r"evaluate=(\d+)", out)
    if DRY or not ev:
        log(state, f"[dry] would submit test {te['key']}: {' '.join(cmd)}")
        return
    te.update(exp=exp, ckpt=ckpt, jobs=ids, eval_job=ev[-1], status="testing", account=account)
    log(state, f"EXP {te['key']}: test {exp} @ {ckpt} on {te['dataset']}/test ({account}, stage {stage}): "
               f"jobs {','.join(ids)} (evaluate {ev[-1]})")


def advance_testing(state, lane, te):
    states = E.job_states(te["jobs"])
    st, reason = states.get(te["eval_job"], ("UNKNOWN", ""))
    if st in ACTIVE and "DependencyNeverSatisfied" not in reason:
        return
    path = E.summary_csv(te["exp"], te["ckpt"])
    if st == "COMPLETED" and path:
        os.makedirs(SUMMARY_DIR, exist_ok=True)
        snap = os.path.join(SUMMARY_DIR, f"{te['exp']}@{te['ckpt']}_{te['dataset']}.csv")
        shutil.copy(path, snap)
        s = read_scores(snap, te["dataset"])
        if s["n"] == 0:
            te["status"] = "failed"
            log(state, f"EXP {te['key']}: summary {os.path.relpath(path, REPO)} has no {te['dataset']} rows")
            return
        te.update(scores=s, summary_csv=os.path.relpath(snap, REPO), status="tested")
        log(state, f"EXP {te['key']}: tested on {te['dataset']} S={s['S']:.4f} avF1={s['avF1']:.4f} "
                   f"C={s['C']:.4f} tp={s['tp']:.4f} (n={s['n']})")
        return
    active = [j for j, (s, _) in states.items() if s in ACTIVE]
    if active and not DRY:
        subprocess.run(["scancel"] + active)
    if te["retries"] < MAX_RETRIES:
        te["retries"] += 1
        log(state, f"EXP {te['key']}: test pipeline ended {st} {reason}; retrying")
        submit_test(state, lane, te)
    else:
        te["status"] = "failed"
        log(state, f"EXP {te['key']}: test pipeline failed twice")


def tick_tests(state, lane):
    """Start ready tests (source arm selected), at most one running test per exp; advance running ones."""
    for te in lane["tests"]:
        if te["status"] == "testing":
            advance_testing(state, lane, te)
    for te in lane["tests"]:
        arm = lane[te["src"]]
        if te["status"] != "wait" or "best_ckpt" not in arm:
            continue
        busy = any(o["status"] == "testing" and lane[o["src"]]["exp"] == arm["exp"] for o in lane["tests"])
        if not busy:
            submit_test(state, lane, te)


# ----------------------------------------------------------------------------- tick

def tick_lane(state, name, lane):
    pre, ft = lane["pre"], lane["ft"]
    if lane["stage"] == "hold":
        return
    if lane["stage"] == "pre":
        if pre["status"] == "new":
            submit_train(state, pre, pre["time"])
        elif pre["status"] == "training" and advance_training(state, pre, pre["time"]):
            select_best(state, pre)
            if ft is None:
                lane["stage"] = "test"
            else:
                lane["stage"] = "ft"
                start_finetune(state, lane)
        if pre["status"] == "failed":
            lane["stage"] = "failed"
    elif lane["stage"] == "ft":
        if ft["status"] == "training" and advance_training(state, ft, FT_TIME):
            select_best(state, ft, min_it=ft["start_it"])
            lane["stage"] = "test"
        if ft["status"] == "failed":
            lane["stage"] = "failed"
    if lane["stage"] in ("ft", "test"):
        tick_tests(state, lane)
    if lane["stage"] == "test" and all(t["status"] in ("tested", "failed") for t in lane["tests"]):
        lane["stage"] = "done" if all(t["status"] == "tested" for t in lane["tests"]) else "failed"
        log(state, f"lane {name}: {lane['stage']}")


def _stage_states(jobs):
    """{stage: [states]} for a test pipeline's jobs (stage = pred / labgraph / label / evalvol / eval)."""
    out = subprocess.run(["sacct", "-n", "-X", "-P", "-j", ",".join(jobs), "-o", "JobID,JobName,State"],
                         capture_output=True, text=True).stdout
    stages = {}
    for line in out.splitlines():
        _, name, st = line.split("|")[:3]
        stages.setdefault(name.split("_")[-2], []).append(st.split()[0])
    return stages


def move_tests(state):
    """Resubmit test pipelines whose GPU work has not started on the current account rotation:
    nothing started -> full pipeline; predictions all done and nothing after started -> --stage label.
    Anything partly through a stage is left alone. Retry counters are not touched."""
    for lane in state["lanes"].values():
        for te in lane["tests"]:
            if te["status"] != "testing" or te.get("account") in ACCOUNTS:
                continue
            st = _stage_states(te["jobs"])
            started = {k for k, v in st.items() if any(x != "PENDING" for x in v)}
            if not started:
                stage = "all"
            elif started == {"pred"} and all(x == "COMPLETED" for x in st.get("pred", [])):
                stage = "label"
            else:
                log(state, f"EXP {te['key']}: not moved (started stages {sorted(started)})")
                continue
            old = list(te["jobs"])
            if DRY:
                log(state, f"[dry] would cancel {','.join(old)} and resubmit {te['key']} from stage {stage}")
                submit_test(state, lane, te, stage)
                continue
            subprocess.run(["scancel"] + old)
            log(state, f"EXP {te['key']}: cancelled {','.join(old)} to move it off tandon_priority")
            submit_test(state, lane, te, stage)


def set_hold(state, name, hold):
    lane = state["lanes"][name]
    if hold:
        if lane["stage"] != "pre" or lane["pre"]["status"] != "new":
            raise SystemExit(f"lane {name} already started (stage {lane['stage']}, pretrain {lane['pre']['status']})")
        lane["stage"] = "hold"
    else:
        if lane["stage"] != "hold":
            raise SystemExit(f"lane {name} is not on hold ({lane['stage']})")
        lane["stage"] = "pre"
        state["done"] = False
    log(state, f"lane {name}: {'held' if hold else 'released'}")


def tick(state):
    if state.get("done"):
        print("done")
        return
    for name, lane in state["lanes"].items():
        tick_lane(state, name, lane)
    if all(l["stage"] in ("done", "failed") for l in state["lanes"].values()):
        state["done"] = True
        log(state, "experiment_v4 driver finished: " + ", ".join(
            f"{k}={l['stage']}" for k, l in state["lanes"].items()))


# ----------------------------------------------------------------------------- status md

def _best_cells(a):
    if not a or "best_ckpt" not in a:
        return ["—"] * 4
    b, sy = a.get("best") or {}, a.get("synth_at_best") or {}
    fmt = E.fmt
    comp = (f"{a['best_ckpt']}: **{fmt(b.get('score'))}** ({fmt(b.get('fg_dice'), 3)} / "
            f"{fmt(b.get('fg_precision'), 3)} / {fmt(b.get('fg_recall'), 3)})") if b else f"{a['best_ckpt']}"
    syn = (f"{fmt(sy.get('score'))} ({fmt(sy.get('fg_dice'), 3)} / {fmt(sy.get('fg_precision'), 3)} / "
           f"{fmt(sy.get('fg_recall'), 3)})") if sy else "—"
    return [comp, fmt(b.get("loss")), syn, fmt(sy.get("loss"))]


def write_status(state):
    fmt = E.fmt
    L = ["# experiment_v4 status (auto-generated by `sbatch/ppp/expv4/expv4.py`)", "",
         f"Last update: {E.now()}. Plan: [experiment_v4.md](experiment_v4.md). Score = mean(fg_dice, fg_precision, "
         "fg_recall) on completely val (selects); synth = the simple_v1 val set at the same iteration (tracked only).",
         "", "## Training", "",
         "| Step | What | exp | status | jobs | best ckpt: completely score (dice / prec / rec) | completely val loss | "
         "synth score (dice / prec / rec) | synth val loss |", "|---" * 9 + "|"]
    for lane in state["lanes"].values():
        for part in ("pre", "ft"):
            a = lane[part]
            if a is None:
                continue
            L.append(f"| {a['key']} | {a['label']} | `{a['exp']}` | {a['status']} | {','.join(a['jobs']) or '—'} | "
                     + " | ".join(_best_cells(a)) + " |")
    L += ["", "## Tests", "",
          "| Step | test set | exp @ ckpt | status | jobs | S | avF1 | C | clDiceTP | tp | FS | FM | nPred / nGT |",
          "|---" * 13 + "|"]
    for lane in state["lanes"].values():
        for t in lane["tests"]:
            s = t.get("scores") or {}
            npgt = f"{fmt(s.get('num_pred'), 0)} / {fmt(s.get('num_gt'), 1)}" if s else "—"
            where = f"`{t['exp']}` @ {t['ckpt']}" if t.get("exp") else "—"
            L.append(f"| {t['key']} | {t['dataset']} | {where} | {t['status']} | {','.join(t['jobs']) or '—'} | "
                     f"{('**' + fmt(s['S']) + '**') if s else '—'} | {fmt(s.get('avF1'))} | {fmt(s.get('C'))} | "
                     f"{fmt(s.get('clDiceTP'))} | {fmt(s.get('tp'))} | {fmt(s.get('FS'), 2)} | {fmt(s.get('FM'), 2)} "
                     f"| {npgt} |")
    L += ["", "## Validation curves (completely score / loss; synth score / loss)", ""]
    for lane in state["lanes"].values():
        for part in ("pre", "ft"):
            a = lane[part]
            if not a or not a.get("val_curve"):
                continue
            syn = {i: (sc, lo) for i, sc, lo in a.get("synth_curve", [])}
            pts = []
            for i, sc, lo in a["val_curve"]:
                ss, sl = syn.get(i, (None, None))
                pts.append(f"{i / 1000:g}k {fmt(sc, 3)}/{fmt(lo, 3)}; {fmt(ss, 3)}/{fmt(sl, 3)}")
            L += [f"- **{a['key']}** `{a['exp']}`: " + " · ".join(pts)]
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
    ap.add_argument("cmd", choices=["tick", "status", "hold", "release", "move"])
    ap.add_argument("--lane", choices=["B0"] + [a[0] for a in ARMS])
    args = ap.parse_args()
    if args.cmd in ("hold", "release") and not args.lane:
        raise SystemExit(f"{args.cmd} needs --lane")
    DRY = args.dry_run
    E.DRY = DRY
    if args.cmd == "status" and not os.path.exists(STATE):
        raise SystemExit("no state yet (run tick)")
    with locked_state() as state:
        if args.cmd == "tick":
            tick(state)
        elif args.cmd == "move":
            move_tests(state)
        elif args.cmd in ("hold", "release"):
            set_hold(state, args.lane, args.cmd == "hold")
    if args.cmd == "status" and os.path.exists(STATUS_MD):
        print(open(STATUS_MD).read())


if __name__ == "__main__":
    main()
