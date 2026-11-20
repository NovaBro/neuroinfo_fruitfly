#!/usr/bin/env python3
"""OP-EXP 1 -> 2 -> 3 driver (TODO.md "Optimization Experiments").

Idempotent state machine. Each ``tick`` looks at SLURM + files on disk and
submits whatever is next: training arms, then (<= 2 at a time) the 1k_v1.1
test pipeline on each arm's last checkpoint, then the phase decision, then the
next phase. ``opexp_tick.sh`` runs ``tick`` every ~10 min on a CPU node and
resubmits itself, so the chain does not depend on an open session.

From the repo root:
  python3 sbatch/ppp/opexp/opexp.py tick       # one step (safe to run by hand)
  python3 sbatch/ppp/opexp/opexp.py status     # rewrite + print the status md
  python3 sbatch/ppp/opexp/opexp.py --dry-run tick

Decisions (user, 2026-11-06):
  * L40S + compile + fp16 raw; n=500 train vols of 1k_v1.1; FMAPs 20 in OP-EXP 1.
  * Test = first 10 sorted 1k_v1.1 test volumes, last checkpoint, --pct 3,
    tandon_advanced, <= 2 test pipelines on disk at once.
  * Metric = mean confusion_matrix avFscore (avF1) over those volumes.
  * stable = no NaN/abort, no loss blow-up, loss decreased (see train_stability).
    (2026-11-07: the "not >5% worse than lr=1e-4" clause was dropped.)
  * 2026-11-07: OP-EXP 3 starts with OP-EXP 2 right after the 1.5 decision, at FMAPs=20.
  * best = stable arm with the highest avF1.
  * largest valid lr = highest lr that is stable and has avF1 >= 0.9 * best avF1.
  * OP-EXP 2.1 (FMAPs 20) reuses the best OP-EXP 1 arm.
  * Phases chain automatically.
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
import statistics
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
METRICS = os.path.join(REPO, "metrics", "ppp")
STATE_DIR = os.path.join(METRICS, "opexp")
STATE = os.path.join(STATE_DIR, "state.json")
STATUS_MD = os.path.join(REPO, "md_guides", "experiments", "opexp_status.md")
LOGS = os.path.join(REPO, "sbatch", "ppp", "logs")

PRESET = "our-noenc-c1k_v1.1"
BASE = "original_default.toml"
# 2026-11-07 20:45 (user): default switched from torch_pr_61_tandon_advanced, whose fair-share fell to 0.015
# (shared heavy usage) vs 0.178 on general. Applies to all new train/test submissions and retries.
ACCOUNT = "torch_pr_61_general"
TEST_CFG = "opexp_test_1k_v1.1.toml"
# 2026-11-07 (user): l40s_public queue full -> OP-EXP 2/3 test only the first 5 sorted volumes;
# every arm compared within a phase is scored on exactly those volumes (read_scores n=...).
TEST_CFG_N5 = "opexp_test_1k_v1.1_n5.toml"
N_TEST = {1: 10, 2: 5, 3: 5}
# 2026-11-08 (user): new test pipeline (commit ac1b4dd). Sparse pred_affs (~0.5 GB instead of
# ~60 GB per volume, labels/metrics identical in the A/B) remove the disk cap that limited us to 2
# pipelines; per-volume chaining + per-volume evaluate are the ppp_submit_test.sh default now.
SPARSE_CFG = "pred_sparse_affs.toml"
MAX_TEST_PIPELINES = 4
TEST_PCT = 5          # all volumes of a 5-volume test at once
MAX_RETRIES = 1
# evaluate: the merge job only reuses per-volume results; per-volume tasks get EVAL_VOL_MEM each
# (10 vols in one 112g job OOM-hung on 2026-11-07; cs has no per-user mem cap)
EVAL_ENV = {"EVAL_MEM": "96g", "EVAL_PARTITION": "cs", "EVAL_TIME": "02:00:00",
            "EVAL_VOL_MEM": "96g", "EVAL_VOL_TIME": "03:00:00"}

LRS = [("1e-4", 0.0001), ("4e-4", 0.0004), ("8e-4", 0.0008), ("1e-3", 0.001)]
LR_VALUE = dict(LRS)
VALID_FRAC = 0.9          # valid lr: avF1 >= 0.9 * best avF1
BLOWUP_FACTOR = 2.0       # final smoothed loss > 2x its running minimum -> blow-up
LEARN_FACTOR = 0.95       # final smoothed loss must be < 0.95x the early loss
WIN = 250                 # smoothing window (iterations)

DRY = False


# ----------------------------------------------------------------------------- utils

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
    """{id: (state, reason)} for job ids (array jobs: worst task state)."""
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
            base = jid.split("_")[0]
            st = st.split()[0]
            prev = agg.get(base)
            # any non-COMPLETED task wins
            if prev is None or (prev == "COMPLETED" and st != "COMPLETED"):
                agg[base] = st
        for i in missing:
            out[i] = (agg.get(i, "UNKNOWN"), "")
    return out


ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED"}


# ----------------------------------------------------------------------------- arms

def exp_dir(exp):
    return os.path.join(METRICS, exp)


def ckpt_path(exp, it):
    return os.path.join(exp_dir(exp), "train", f"train_net_checkpoint_{it}")


def make_arm(key, exp, overlays, max_it, label, **meta):
    return {"key": key, "exp": exp, "overlays": overlays, "max_it": max_it, "label": label,
            "status": "new", "train_jobs": [], "test_jobs": [], "train_retries": 0,
            "test_retries": 0, **meta}


def phase1_arms():
    return [make_arm(f"1.{i+1}", f"opexp1-lr{n}", ["opexp_common.toml", f"opexp_lr{n}.toml"], 10000,
                     f"lr={v}", lr=n, fmaps=20, sched=None)
            for i, (n, v) in enumerate(LRS)]


def fmaps_overlay(f):
    return [] if f == 20 else [f"opexp_fmaps{f}.toml"]


def phase2_arms(state):
    lr = state["decisions"]["best_lr"]
    best1 = state["decisions"]["best_arm_1"]
    reuse = dict(next(a for a in state["arms"] if a["key"] == best1))
    reuse.update({"key": "2.1", "label": f"fmaps=20 (reuse OP-EXP {best1})", "reused_from": best1})
    reuse["scores"] = read_scores(os.path.join(REPO, reuse["summary_csv"]), n=N_TEST[2])
    arms = [reuse]
    for key, f in (("2.2", 15), ("2.3", 10)):
        arms.append(make_arm(key, f"opexp2-f{f}-lr{lr}",
                             ["opexp_common.toml", f"opexp_lr{lr}.toml"] + fmaps_overlay(f), 10000,
                             f"fmaps={f}, lr={LR_VALUE[lr]}", lr=lr, fmaps=f, sched=None))
    return arms


def phase3_arms(state):
    d = state["decisions"]
    lr, vlr, f = d["best_lr"], d["valid_max_lr"], d["best_fmaps"]
    common = ["opexp_common.toml", "opexp_50k.toml"]
    arms = [make_arm("3.1", f"opexp3-base-f{f}-lr{lr}", common + [f"opexp_lr{lr}.toml"] + fmaps_overlay(f),
                     50000, f"no scheduler, lr={LR_VALUE[lr]}, fmaps={f}", lr=lr, fmaps=f, sched=None)]
    for key, w, frac in (("3.2", "05", "5%"), ("3.3", "01", "1%"), ("3.4", "10", "10%")):
        arms.append(make_arm(key, f"opexp3-wc{w}-f{f}-lr{vlr}",
                             common + [f"opexp_lr{vlr}.toml"] + fmaps_overlay(f) + [f"opexp_wc_w{w}.toml"],
                             50000, f"warmup {frac} + cosine, peak lr={LR_VALUE[vlr]}, fmaps={f}",
                             lr=vlr, fmaps=f, sched=f"wc{w}"))
    return arms


def phase_of(arm):
    return int(arm["key"].split(".")[0])


# ----------------------------------------------------------------------------- train

def submit_train(state, arm):
    t = "08:00:00" if arm["max_it"] <= 10000 else "36:00:00"
    cmd = ["./sbatch/ppp/ppp_submit_train.sh"] + (["--dry-run"] if DRY else []) + [
        PRESET, "--base", BASE, "--id", arm["exp"]]
    for o in arm["overlays"]:
        cmd += ["--config", o]
    cmd += ["--account", ACCOUNT, "--time", t]
    out = run(cmd)
    m = re.search(r"train=(\S+)", out)
    jid = m.group(1) if m else None
    if not jid or DRY:
        log(state, f"[dry] would submit train {arm['key']} {arm['exp']}")
        return
    arm["train_jobs"].append(jid)
    arm["status"] = "training"
    log(state, f"OP-EXP {arm['key']} ({arm['label']}): train job {jid} exp={arm['exp']}")


def train_logs(arm):
    paths = []
    for j in arm["train_jobs"]:
        paths += glob.glob(os.path.join(LOGS, f"ppp_*_train-{j}.out"))
        paths += glob.glob(os.path.join(LOGS, f"ppp_*_train-{j}.err"))
    return paths


LOSS_RE = re.compile(r"Train process: iteration=(\d+) loss=(\S+)")


def read_losses(arm):
    losses = {}
    for p in train_logs(arm):
        with open(p, errors="replace") as fh:
            for line in fh:
                m = LOSS_RE.search(line)
                if m:
                    try:
                        losses[int(m.group(1))] = float(m.group(2))
                    except ValueError:
                        losses[int(m.group(1))] = float("nan")
    return [losses[k] for k in sorted(losses)]


def log_has_nan_abort(arm):
    for p in train_logs(arm):
        with open(p, errors="replace") as fh:
            txt = fh.read()
        if re.search(r"Check lr, use_auto_mixed_precision|loss is nan|NaN loss", txt, re.I):
            return True
    return False


def train_stability(arm, ref_final=None):
    """(stable: bool, info: dict). ref_final: lr=1e-4 arm's final loss (phase 1)."""
    if arm.get("diverged"):
        return False, {"reason": "aborted (NaN / non-finite loss)"}
    ls = read_losses(arm)
    if len(ls) < 2 * WIN:
        return False, {"reason": f"too few loss lines ({len(ls)})"}
    if any(not math.isfinite(x) for x in ls[-WIN:]):
        return False, {"reason": "non-finite loss at end"}
    finite = [x if math.isfinite(x) else 1e9 for x in ls]
    meds = [statistics.median(finite[i:i + WIN]) for i in range(50, len(finite) - WIN + 1, WIN // 2)]
    early, final, best = meds[0], statistics.median(finite[-WIN:]), min(meds)
    info = {"early": early, "final": final, "min": best}
    if final > BLOWUP_FACTOR * best:
        info["reason"] = f"blow-up: final {final:.3f} > {BLOWUP_FACTOR}x min {best:.3f}"
        return False, info
    if final >= LEARN_FACTOR * early:
        info["reason"] = f"not learning: final {final:.3f} >= {LEARN_FACTOR}x early {early:.3f}"
        return False, info
    if ref_final is not None and final > 1.05 * ref_final:
        info["reason"] = f"worse than lr=1e-4: final {final:.3f} > 1.05x {ref_final:.3f}"
        return False, info
    info["reason"] = "ok"
    return True, info


def advance_training(state, arm):
    jid = arm["train_jobs"][-1]
    st, _ = job_states([jid]).get(jid, ("UNKNOWN", ""))
    if st in ACTIVE:
        return
    if os.path.exists(ckpt_path(arm["exp"], arm["max_it"])):
        arm["status"] = "trained"
        log(state, f"OP-EXP {arm['key']}: training done ({st}), ckpt {arm['max_it']}")
        return
    if log_has_nan_abort(arm):
        arm["status"] = "trained"   # nothing to test; marked unstable
        arm["diverged"] = True
        log(state, f"OP-EXP {arm['key']}: training aborted on non-finite loss -> diverged")
        return
    if arm["train_retries"] < MAX_RETRIES:
        arm["train_retries"] += 1
        log(state, f"OP-EXP {arm['key']}: train job {jid} ended {st} without final ckpt; resubmitting (resumes)")
        submit_train(state, arm)
    else:
        arm["status"] = "failed"
        log(state, f"OP-EXP {arm['key']}: train failed twice ({st}); arm dropped")


# ----------------------------------------------------------------------------- test

def summary_csv(arm):
    root = os.path.join(exp_dir(arm["exp"]), "test", "evaluated", str(arm["max_it"]))
    c = [p for p in glob.glob(os.path.join(root, "**", "summary.csv"), recursive=True)]
    return c[0] if c else None


def read_scores(path, n=None):
    """Mean metrics over the first n sorted 1k_v1.1 test volumes (all if n is None)."""
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if r.get("sample", "").startswith("1k_v1.1/"):
                rows.append(r)
    rows.sort(key=lambda r: r["sample"])
    if n is not None:
        rows = rows[:n]

    def mean(col):
        v = [float(r[col]) for r in rows if r.get(col) not in (None, "", "nan")]
        return sum(v) / len(v) if v else float("nan")

    return {"n": len(rows), "avF1": mean("confusion_matrix avFscore"),
            "avg_f1_cov": mean("general avg_f1_cov_score"),
            "num_pred": mean("general Num Pred"), "num_gt": mean("general Num GT"),
            "f1_05": mean("th_0_5 fscore")}


def submit_test(state, arm):
    cmd = ["./sbatch/ppp/ppp_submit_test.sh"] + (["--dry-run"] if DRY else []) + [
        "array", PRESET, "--id", arm["exp"], "--split", "test", "--ckpt", str(arm["max_it"]),
        "--base", BASE]
    for o in (arm["overlays"] + [TEST_CFG] + ([TEST_CFG_N5] if N_TEST[phase_of(arm)] == 5 else [])
              + [SPARSE_CFG]):
        cmd += ["--config", o]
    # per-arm override (2026-11-07: 2.3 on torch_pr_61_general to compare queue start vs tandon_advanced)
    cmd += ["--pct", str(TEST_PCT), "--account", arm.get("account", ACCOUNT)]
    out = run(cmd, env=EVAL_ENV)
    ids = re.findall(r"(?:array_job|evaluate)=(\d+)", out)
    ev = re.findall(r"evaluate=(\d+)", out)
    if DRY or not ev:
        log(state, f"[dry] would submit test {arm['key']}")
        return
    arm["test_jobs"] = ids
    arm["eval_job"] = ev[-1]
    arm["status"] = "testing"
    log(state, f"OP-EXP {arm['key']}: test pipeline jobs {','.join(ids)} (evaluate {ev[-1]})")


def advance_testing(state, arm):
    states = job_states(arm["test_jobs"])
    ev = arm["eval_job"]
    st, reason = states.get(ev, ("UNKNOWN", ""))
    if st in ACTIVE and "DependencyNeverSatisfied" not in reason:
        return
    path = summary_csv(arm)
    if st == "COMPLETED" and path:
        arm["scores"] = read_scores(path, n=N_TEST[phase_of(arm)])
        arm["summary_csv"] = os.path.relpath(path, REPO)
        arm["status"] = "tested"
        s = arm["scores"]
        log(state, f"OP-EXP {arm['key']}: tested avF1={s['avF1']:.4f} avg_f1_cov={s['avg_f1_cov']:.4f} "
                   f"(n={s['n']})")
        return
    # failed / stuck pipeline
    active = [j for j, (s, _) in states.items() if s in ACTIVE]
    if active and not DRY:
        subprocess.run(["scancel"] + active)
    if arm["test_retries"] < MAX_RETRIES:
        arm["test_retries"] += 1
        log(state, f"OP-EXP {arm['key']}: test pipeline ended {st} {reason}; retrying")
        arm["status"] = "trained"   # re-queued for a test slot
    else:
        arm["status"] = "test_failed"
        log(state, f"OP-EXP {arm['key']}: test pipeline failed twice; arm has no avF1")


# ----------------------------------------------------------------------------- decisions

def evaluate_phase(state, phase):
    arms = [a for a in state["arms"] if phase_of(a) == phase]
    rows = []
    for a in arms:
        # user 2026-11-07: no "worse than lr=1e-4" clause (it excluded a smooth 8e-4 run by 6.6%)
        stable, info = train_stability(a)
        a["stable"], a["stability"] = stable, info
        rows.append(a)
    scored = [a for a in rows if a["stable"] and a.get("scores") and math.isfinite(a["scores"]["avF1"])]
    return rows, scored


def pick_best(scored):
    return max(scored, key=lambda a: (a["scores"]["avF1"], a["scores"]["avg_f1_cov"]))


def decide(state, phase):
    rows, scored = evaluate_phase(state, phase)
    d = state["decisions"]
    if not scored:
        log(state, f"OP-EXP {phase}: no stable arm with a test score -> stopping; needs a human")
        state["done"] = True
        state["blocked"] = f"phase {phase}: no stable scored arm"
        return
    best = pick_best(scored)
    thr = VALID_FRAC * best["scores"]["avF1"]
    for a in rows:
        a["valid"] = bool(a["stable"] and a.get("scores") and a["scores"]["avF1"] >= thr)
    if phase == 1:
        d["best_arm_1"], d["best_lr"] = best["key"], best["lr"]
        valid = [a for a in rows if a["valid"]]
        vmax = max(valid, key=lambda a: LR_VALUE[a["lr"]])
        d["valid_max_lr"] = vmax["lr"]
        log(state, f"OP-EXP 1.5 decision: best lr={LR_VALUE[best['lr']]} ({best['key']}, "
                   f"avF1 {best['scores']['avF1']:.4f}); largest valid lr={LR_VALUE[vmax['lr']]} "
                   f"(threshold avF1 >= {thr:.4f})")
        state["phase"] = 2
        state["arms"] += phase2_arms(state)
        # user 2026-11-07: run OP-EXP 3 in parallel with OP-EXP 2, at FMAPs=20 (deadline)
        d["best_fmaps"] = 20
        d["phase3_fmaps"] = "20 (overlapped with OP-EXP 2; user decision 2026-11-07)"
        state["arms"] += phase3_arms(state)
        log(state, "OP-EXP 3 launched alongside OP-EXP 2 at FMAPs=20")
    elif phase == 2:
        d["best_arm_2"], d["best_fmaps_opexp2"] = best["key"], best["fmaps"]
        log(state, f"OP-EXP 2.4 decision: best fmaps={best['fmaps']} ({best['key']}, "
                   f"avF1 {best['scores']['avF1']:.4f})")
        state["phase"] = 3
    else:
        d["best_arm_3"] = best["key"]
        log(state, f"OP-EXP 3 result: best = {best['key']} {best['label']} (avF1 {best['scores']['avF1']:.4f})")


# ----------------------------------------------------------------------------- tick

def tick(state):
    if state.get("done"):
        print("done:", state.get("blocked") or state["decisions"])
        return
    phase = state["phase"]
    for a in state["arms"]:
        if a["status"] == "new":
            submit_train(state, a)
        elif a["status"] == "training":
            advance_training(state, a)
        elif a["status"] == "testing":
            advance_testing(state, a)
    # test slots (oldest phase first, then key order)
    busy = sum(a["status"] == "testing" for a in state["arms"])
    for a in state["arms"]:
        if busy >= MAX_TEST_PIPELINES:
            break
        if a["status"] == "trained" and not a.get("diverged"):
            submit_test(state, a)
            if a["status"] == "testing":
                busy += 1
    terminal = {"tested", "failed", "test_failed"}
    decided = state.setdefault("decided", [])
    for p in (1, 2, 3):
        cur = [a for a in state["arms"] if phase_of(a) == p]
        if p in decided or not cur:
            continue
        if all(a["status"] in terminal or (a["status"] == "trained" and a.get("diverged")) for a in cur):
            decided.append(p)
            decide(state, p)
            if state.get("done"):
                return
            if p == 1:
                tick(state)   # submit OP-EXP 2 + 3 right away
                return
    if 2 in decided and 3 in decided:
        state["done"] = True


# ----------------------------------------------------------------------------- status md

def fmt(x, n=4):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{n}f}"


def write_status(state):
    L = ["# OP-EXP status (auto-generated by `sbatch/ppp/opexp/opexp.py`)", "",
         f"Last update: {now()}. Rules and setup: docstring of `sbatch/ppp/opexp/opexp.py`; "
         "overlays in `setup01/configs/exp/opexp/`.", "",
         "Setup: `our-noenc-c1k_v1.1` + `original_default.toml`, n=500 train vols, bs18, L40S, compile + fp16 raw. "
         "Test: first 10 vols of 1k_v1.1/test, last checkpoint, vi_th_0_3 + eval_rm200.", "",
         f"Phase: **{state['phase']}**{' (done)' if state.get('done') else ''}"
         + (f" — BLOCKED: {state['blocked']}" if state.get("blocked") else ""), "",
         "## Decisions", ""]
    for k, v in state["decisions"].items():
        L.append(f"- `{k}` = {v}")
    L += ["", "## Arms", "",
          "| Arm | Setup | exp | status | train jobs | test eval | final loss | stable | avF1 | avg_f1_cov | Num Pred / GT |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in state["arms"]:
        s = a.get("scores") or {}
        st = a.get("stability") or {}
        stable = "—" if "stable" not in a else ("yes" if a["stable"] else f"no ({st.get('reason')})")
        np_gt = f"{fmt(s.get('num_pred'), 0)} / {fmt(s.get('num_gt'), 1)}" if s else "—"
        L.append(f"| {a['key']} | {a['label']} | `{a['exp']}` | {a['status']} | {','.join(a['train_jobs']) or '—'} "
                 f"| {a.get('eval_job', '—')} | {fmt(st.get('final'), 3)} | {stable} | {fmt(s.get('avF1'))} "
                 f"| {fmt(s.get('avg_f1_cov'))} | {np_gt} |")
    L += ["", "## Event log", ""] + [f"- {e}" for e in state.get("events", [])] + [""]
    os.makedirs(os.path.dirname(STATUS_MD), exist_ok=True)
    with open(STATUS_MD, "w") as fh:
        fh.write("\n".join(L))


# ----------------------------------------------------------------------------- main

@contextlib.contextmanager
def locked_state():
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(STATE):
            with open(STATE) as fh:
                state = json.load(fh)
        else:
            state = {"phase": 1, "arms": phase1_arms(), "decisions": {}, "events": []}
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
        else:
            for a in state["arms"]:
                if a["status"] in ("tested", "trained", "failed", "test_failed"):
                    a["stable"], a["stability"] = train_stability(a)
    if args.cmd == "status" or DRY:
        if os.path.exists(STATUS_MD):
            print(open(STATUS_MD).read())


if __name__ == "__main__":
    main()
