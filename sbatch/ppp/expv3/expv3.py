#!/usr/bin/env python3
"""experiment_v3 driver (md_guides/experiments/experiment_v3.md): S6 D2 W{64,32} U-Net vs the current net.

Three lanes, each advanced by an idempotent ``tick`` (pretrain -> select -> finetune -> select -> test):
  B   EXP 3.0  baseline: current net (FMAPs 15, 4 stages, valid). Pretrain = existing vince-data_n1000
               (Dataset EXP 1.1); its best *completely* val ckpt -> finetune completely -> test.
  W64 EXP 3.1 pretrain S6D2W64 on 1k_v1.1 -> 3.2 finetune completely -> 3.3 test completely
  W32 EXP 3.4 pretrain S6D2W32 on 1k_v1.1 -> 3.5 finetune completely -> 3.6 test completely

W64 / W32 start in stage ``hold`` until ``release`` sets crop / batch size / length from the P3 smoke:
  python3 sbatch/ppp/expv3/expv3.py release --lane W64 --iters 75000 --time 12:00:00 [--overlay expv3_smoke_bs3.toml]

From the repo root:
  python3 sbatch/ppp/expv3/expv3.py tick        # one step (safe by hand)
  python3 sbatch/ppp/expv3/expv3.py status
  python3 sbatch/ppp/expv3/expv3.py --dry-run tick
``expv3_tick.sh`` runs ``tick`` every ~10 min on a CPU node and resubmits itself (stop: scancel -n expv3_tick).

Setup (user, 2026-11-17):
  * preset our-noenc-c1k_v1.1 + original_default.toml + expv2_e1_common.toml (compile, fp16 raw), L40S,
    account torch_pr_61_general; overlays in setup01/configs/exp/expv3/.
  * pretrain: all 796 1k_v1.1 train vols, peak lr 4e-4, 10% warmup + cosine, ckpt + val every 2.5k; val on
    completely (selects: lowest val loss) + 1k_v1.1 (extra). Length from the P3 smoke; a job that times out is
    resubmitted and resumes from its last checkpoint (scheduler state is checkpointed).
  * finetune: completely train (18), 6k its, fresh Adam, peak 5e-5, 10% warmup + cosine, ckpt + val every 1k
    (dsexp_ft.toml + dsexp_ft_1k_v1.1.toml); best = lowest completely val loss.
  * test: 7 completely test vols, sparse pred_affs, vi_th_0_3, eval_rm1500; W lanes predict with
    expv3_predict.toml (224^3 tiles, test_same_margin 32).
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import glob
import json
import math
import os
import re
import subprocess
import sys
import csv

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "sbatch", "ppp", "expv2"))
import expv2_e1 as E  # noqa: E402  (shared helpers: run, job_states, scores, ...)

STATE_DIR = os.path.join(E.METRICS, "expv3")
STATE = os.path.join(STATE_DIR, "state.json")
STATUS_MD = os.path.join(REPO, "md_guides", "experiments", "expv3_status.md")
CFG_DIR = os.path.join(REPO, "PatchPerPix", "experiments", "flylight", "setups", "setup01", "configs", "exp", "expv3")
SIF = E.SIF
PRESET = E.PRESET
BASE = E.BASE
ACCOUNT = "torch_pr_61_general"
COMMON = ["expv2_e1_common.toml"]
FT_STATIC = ["dsexp_ft.toml", "dsexp_ft_1k_v1.1.toml"]
FT_ITERS = 6000
FT_LR = 5e-5
FT_WARMUP_FRAC = 0.10
FT_TIME = "06:00:00"
SPARSE_CFG = "pred_sparse_affs.toml"
PREDICT_CFG = "expv3_predict.toml"
EVAL_CFG = "eval_rm1500.toml"
TEST_PCT = 7
EVAL_ENV = E.EVAL_ENV
MAX_RETRIES = 1     # failures without progress
MAX_CHAIN = 6       # timeouts with progress (resume from the last checkpoint)
ACTIVE = E.ACTIVE
BASELINE_PRE = "vince-data_n1000"

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
    """True if the run log shows the final training iteration."""
    p = os.path.join(E.METRICS, exp, "run.log")
    if not os.path.exists(p):
        return False
    with open(p, errors="replace") as fh:
        return f"Train process: iteration={max_it} " in fh.read()


# ----------------------------------------------------------------------------- state

def initial_state():
    lanes = {
        "B": {"stage": "select", "model": COMMON[:], "ft_time": "03:30:00",
              "pre": {"key": "3.0a", "exp": BASELINE_PRE, "status": "trained", "jobs": [], "retries": 0,
                      "chain": 0, "val_csv": "val_metrics_completely.csv", "max_it": 35000,
                      "label": "pretrain (existing Dataset EXP 1.1, current net, 35k its)"},
              "ft": {"key": "3.0b", "exp": "expv3-e3.0-ft-baseline", "status": "wait", "jobs": [], "retries": 0,
                     "chain": 0, "label": "finetune completely, current net"},
              "test": {"key": "3.0c", "status": "wait", "jobs": [], "retries": 0,
                       "label": "test completely, rm1500"}},
    }
    for lane, (k_pre, k_ft, k_te, model) in {"W64": ("3.1", "3.2", "3.3", "expv3_s6d2w64.toml"),
                                             "W32": ("3.4", "3.5", "3.6", "expv3_s6d2w32.toml")}.items():
        tag = lane.lower()
        lanes[lane] = {
            "stage": "hold", "model": COMMON + [model], "ft_time": FT_TIME,
            "pre": {"key": k_pre, "exp": f"expv3-e{k_pre}-pre-s6d2{tag}", "status": "hold", "jobs": [],
                    "retries": 0, "chain": 0, "val_csv": "val_metrics.csv",
                    "label": f"pretrain 1k_v1.1 (796), S6D2{lane}"},
            "ft": {"key": k_ft, "exp": f"expv3-e{k_ft}-ft-s6d2{tag}", "status": "wait", "jobs": [], "retries": 0,
                   "chain": 0, "label": f"finetune completely, S6D2{lane}"},
            "test": {"key": k_te, "status": "wait", "jobs": [], "retries": 0,
                     "label": "test completely, rm1500, 224^3 tiles margin 32"},
        }
    return {"lanes": lanes, "events": [], "done": False}


# ----------------------------------------------------------------------------- train

def submit_train(state, arm, time_limit):
    cmd = ["./sbatch/ppp/ppp_submit_train.sh"] + (["--dry-run"] if DRY else []) + [
        PRESET, "--base", BASE, "--id", arm["exp"]]
    for o in arm["overlays"]:
        cmd += ["--config", o]
    cmd += ["--account", ACCOUNT, "--time", time_limit]
    out = E.run(cmd)
    m = re.search(r"train=(\d+)", out)
    if DRY or not m:
        log(state, f"[dry] would submit train {arm['key']} {arm['exp']}: {' '.join(cmd)}")
        return
    arm["jobs"].append(m.group(1))
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
        # no checkpoint is written at max_it when it is off the checkpoint grid (e.g. a finetune from a pretrain
        # ckpt at x500 with 1k checkpoints: 12500 + 6000 = 18500); the log's last iteration says it finished
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

def val_rows(exp, csv_name, min_it=0):
    p = os.path.join(E.METRICS, exp, "train", csv_name)
    rows = []
    if os.path.exists(p):
        with open(p) as fh:
            for r in csv.DictReader(fh):
                try:
                    it, loss = int(r["iteration"]), float(r["loss"])
                except (ValueError, KeyError):
                    continue
                if it > min_it and math.isfinite(loss) and os.path.exists(E.ckpt_path(exp, it)):
                    rows.append((it, loss))
    return rows


def select_best(state, arm, min_it=0):
    rows = val_rows(arm["exp"], arm.get("val_csv", "val_metrics.csv"), min_it)
    if not rows:
        arm["best_ckpt"] = arm["max_it"]
        arm["best_reason"] = "no val rows with checkpoints; using last checkpoint"
    else:
        it, loss = min(rows, key=lambda r: (r[1], -r[0]))
        arm["best_ckpt"], arm["best_val_loss"], arm["val_curve"] = it, loss, rows
        arm["best_reason"] = f"lowest completely val loss {loss:.4f} of {len(rows)} checkpoints"
    log(state, f"EXP {arm['key']}: best checkpoint {arm['best_ckpt']} ({arm['best_reason']})")


# ----------------------------------------------------------------------------- finetune

def start_finetune(state, lane):
    pre, ft = lane["pre"], lane["ft"]
    n = int(pre["best_ckpt"])
    ft["src_ckpt"], ft["start_it"], ft["max_it"] = n, n, n + FT_ITERS
    gen = f"expv3_ft_{ft['exp']}.toml"
    with open(os.path.join(CFG_DIR, gen), "w") as fh:
        fh.write(f"# generated by sbatch/ppp/expv3/expv3.py: EXP {ft['key']} finetunes {pre['exp']} @ {n} "
                 f"for {FT_ITERS} its (peak lr {FT_LR}, set in dsexp_ft.toml).\n"
                 f"[training]\nmax_iterations = {n + FT_ITERS}\n\n"
                 f"[optimizer.lr_scheduler]\nstart_iter = {n}\n")
    ft["overlays"] = lane["model"] + FT_STATIC + [gen]
    ft["val_csv"] = "val_metrics.csv"
    if not DRY:
        dst = os.path.join(E.METRICS, ft["exp"], "train")
        inner = (f"source /ext3/env.sh; conda activate ppp; python sbatch/ppp/expv2/seed_finetune_ckpt.py "
                 f"--src {E.ckpt_path(pre['exp'], n)} --dst-dir {dst} --iteration {n} --lr {FT_LR} "
                 f"--warmup-frac {FT_WARMUP_FRAC} --ft-iters {FT_ITERS}")
        out = E.run(["singularity", "exec", "--overlay", "env/ppp.ext3:ro", SIF, "/bin/bash", "-c", inner])
        log(state, f"EXP {ft['key']}: " + out.strip().splitlines()[-1])
    submit_train(state, ft, lane.get("ft_time", FT_TIME))


# ----------------------------------------------------------------------------- test

def submit_test(state, lane, te=None, ckpt=None):
    """Test the lane's finetune on completely/test. Default: lane["test"] at the selected checkpoint; extra
    tests pass their own record ``te`` and ``ckpt``."""
    ft = lane["ft"]
    te = lane["test"] if te is None else te
    ckpt = ft["best_ckpt"] if ckpt is None else ckpt
    te["exp"], te["ckpt"] = ft["exp"], ckpt
    extra = [SPARSE_CFG] + ([PREDICT_CFG] if any(m.startswith("expv3_s6") for m in lane["model"]) else [])
    cmd = ["./sbatch/ppp/ppp_submit_test.sh"] + (["--dry-run"] if DRY else []) + [
        "array", PRESET, "--id", ft["exp"], "--split", "test", "--ckpt", str(ckpt),
        "--base", BASE, "--eval", EVAL_CFG]
    for o in ft["overlays"] + extra:
        cmd += ["--config", o]
    cmd += ["--pct", str(TEST_PCT), "--account", ACCOUNT]
    out = E.run(cmd, env=EVAL_ENV)
    ids = re.findall(r"(?:array_job|evaluate)=(\d+)", out)
    ev = re.findall(r"evaluate=(\d+)", out)
    if DRY or not ev:
        log(state, f"[dry] would submit test {te['key']}: {' '.join(cmd)}")
        return
    te["jobs"], te["eval_job"], te["status"] = ids, ev[-1], "testing"
    log(state, f"EXP {te['key']}: test {ft['exp']} @ {ckpt} on completely/test: "
               f"jobs {','.join(ids)} (evaluate {ev[-1]})")


def advance_testing(state, lane, te=None):
    te = lane["test"] if te is None else te
    states = E.job_states(te["jobs"])
    st, reason = states.get(te["eval_job"], ("UNKNOWN", ""))
    if st in ACTIVE and "DependencyNeverSatisfied" not in reason:
        return
    path = E.summary_csv(te["exp"], te["ckpt"])
    if st == "COMPLETED" and path:
        te["scores"], te["summary_csv"], te["status"] = E.read_scores(path), os.path.relpath(path, REPO), "tested"
        s = te["scores"]
        log(state, f"EXP {te['key']}: tested S={s['S']:.4f} avF1={s['avF1']:.4f} C={s['C']:.4f} "
                   f"tp={s['tp']:.4f} (n={s['n']})")
        return
    active = [j for j, (s, _) in states.items() if s in ACTIVE]
    if active and not DRY:
        subprocess.run(["scancel"] + active)
    if te["retries"] < MAX_RETRIES:
        te["retries"] += 1
        log(state, f"EXP {te['key']}: test pipeline ended {st} {reason}; retrying")
        submit_test(state, lane, te, te["ckpt"])
    else:
        te["status"] = "failed"
        log(state, f"EXP {te['key']}: test pipeline failed twice")


# ----------------------------------------------------------------------------- tick

def tick_lane(state, name, lane):
    pre, ft, te = lane["pre"], lane["ft"], lane["test"]
    if lane["stage"] == "hold":
        return
    if lane["stage"] == "select":       # baseline: pretrain already done
        select_best(state, pre)
        lane["stage"] = "ft"
        start_finetune(state, lane)
    elif lane["stage"] == "pre":
        if pre["status"] == "new":
            submit_train(state, pre, pre["time"])
        elif pre["status"] == "training" and advance_training(state, pre, pre["time"]):
            select_best(state, pre)
            lane["stage"] = "ft"
            start_finetune(state, lane)
        if pre["status"] == "failed":
            lane["stage"] = "failed"
    elif lane["stage"] == "ft":
        if ft["status"] == "training" and advance_training(state, ft, lane.get("ft_time", FT_TIME)):
            select_best(state, ft, min_it=ft["start_it"])
            lane["stage"] = "test"
            submit_test(state, lane)
        if ft["status"] == "failed":
            lane["stage"] = "failed"
    elif lane["stage"] == "test":
        if te["status"] == "testing":
            advance_testing(state, lane)
        if te["status"] == "tested":
            lane["stage"] = "done"
        elif te["status"] == "failed":
            lane["stage"] = "failed"


P2_LANE, P2_CKPT = "W64", 5000


def maybe_p2(state):
    """P2 seam check (plan P2): once the W64 pretrain has checkpoint P2_CKPT, run p2_seam_check.sh once
    (2 predict jobs, margin 0 and 32, + a CPU compare job writing metrics/ppp/expv3/p2_seam_check.md)."""
    p2 = state.setdefault("p2", {"status": "wait"})
    if p2["status"] == "running":
        st, reason = E.job_states([p2["compare"]]).get(p2["compare"], ("UNKNOWN", ""))
        if st in ACTIVE and "DependencyNeverSatisfied" not in reason:
            return
        report = os.path.join(STATE_DIR, "p2_seam_check.md")
        p2["status"] = "done" if st == "COMPLETED" and os.path.exists(report) else "failed"
        log(state, f"P2 seam check {p2['status']} ({st} {reason}); report metrics/ppp/expv3/p2_seam_check.md")
        return
    if p2["status"] != "wait":
        return
    pre = state["lanes"][P2_LANE]["pre"]
    if pre.get("status") not in ("training", "trained") or not os.path.exists(E.ckpt_path(pre["exp"], P2_CKPT)):
        return
    if DRY:
        log(state, f"[dry] would start P2 seam check on {pre['exp']} @ {P2_CKPT}")
        return
    out = E.run(["bash", "sbatch/ppp/expv3/p2_seam_check.sh", pre["exp"], str(P2_CKPT)])
    m = re.search(r"p2 jobs=(\S+) compare=(\d+)", out)
    if not m:
        p2["status"] = "failed"
        log(state, f"P2 seam check: could not parse submit output: {out.strip()[-300:]}")
        return
    p2.update(status="running", ckpt=P2_CKPT, src=pre["exp"], jobs=m.group(1).split(","), compare=m.group(2))
    log(state, f"P2 seam check: {pre['exp']} @ {P2_CKPT}: predict jobs {m.group(1)} (margin 0/32), "
               f"compare job {m.group(2)}")


def tick(state):
    if state.get("done"):
        print("done")
        return
    try:
        maybe_p2(state)
    except Exception as e:  # never let the side check block the lanes
        state.setdefault("p2", {})["status"] = "failed"
        log(state, f"P2 seam check error: {type(e).__name__}: {e}")
    for name, lane in state["lanes"].items():
        tick_lane(state, name, lane)
    for x in state.get("extra", []):
        if x["status"] == "new":
            submit_test(state, state["lanes"][x["lane"]], x, x["ckpt"])
        elif x["status"] == "testing":
            advance_testing(state, state["lanes"][x["lane"]], x)
    if all(l["stage"] in ("done", "failed") for l in state["lanes"].values()) and \
            all(x["status"] in ("tested", "failed") for x in state.get("extra", [])):
        state["done"] = True
        log(state, "experiment_v3 driver finished: " + ", ".join(
            f"{k}={l['stage']}" for k, l in state["lanes"].items()))


def release(state, lane_name, iters, time_limit, overlays, ft_time=None):
    lane = state["lanes"][lane_name]
    if lane["stage"] != "hold":
        raise SystemExit(f"lane {lane_name} is {lane['stage']}, not hold")
    for o in overlays:
        if not os.path.exists(os.path.join(CFG_DIR, o)):
            raise SystemExit(f"overlay {o} not in {CFG_DIR}")
    lane["model"] = lane["model"] + list(overlays)
    if ft_time:
        lane["ft_time"] = ft_time
    pre = lane["pre"]
    gen = f"expv3_len_{pre['exp']}.toml"
    if not DRY:
        with open(os.path.join(CFG_DIR, gen), "w") as fh:
            fh.write(f"# generated by sbatch/ppp/expv3/expv3.py release: EXP {pre['key']} pretrain length "
                     f"(set from the P3 smoke).\n[training]\nmax_iterations = {iters}\n")
    pre.update(max_it=iters, time=time_limit, status="new",
               overlays=lane["model"] + ["expv3_pre.toml", gen])
    lane["stage"] = "pre"
    log(state, f"lane {lane_name} released: {iters} its, {time_limit} jobs, extra overlays {overlays or '-'}")


def resume_lane(state, lane_name):
    """Put a lane whose finetune was wrongly marked failed back to stage ft (the next tick re-checks it)."""
    lane = state["lanes"][lane_name]
    ft = lane["ft"]
    if not ft.get("jobs"):
        raise SystemExit(f"lane {lane_name} has no finetune job to re-check")
    lane["stage"], ft["status"] = "ft", "training"
    state["done"] = False
    log(state, f"lane {lane_name}: resumed at stage ft (re-check finetune completion)")


def add_extra_test(state, lane_name, ckpt, label):
    lane = state["lanes"][lane_name]
    key = f"{lane['test']['key']}x{ckpt}"
    if any(x["key"] == key for x in state.get("extra", [])):
        raise SystemExit(f"extra test {key} already exists")
    state.setdefault("extra", []).append({"key": key, "lane": lane_name, "ckpt": ckpt, "status": "new", "jobs": [],
                                          "retries": 0, "label": label})
    state["done"] = False
    log(state, f"extra test {key} queued: lane {lane_name} @ {ckpt} ({label})")


# ----------------------------------------------------------------------------- status md

def write_status(state):
    fmt = E.fmt
    L = ["# experiment_v3 status (auto-generated by `sbatch/ppp/expv3/expv3.py`)", "",
         f"Last update: {E.now()}. Plan: [experiment_v3.md](experiment_v3.md). Driver logic: docstring of "
         "`sbatch/ppp/expv3/expv3.py`. Reference (completely test, rm1500): expv2 EXP 1.3 S 0.305; OP-EXP 6.6 "
         "LoRA 0.291; Dataset EXP 0 (from scratch) 0.259.", "",
         "| Step | What | exp | status | jobs | best ckpt (completely val loss) | S | avF1 | C | clDiceTP | tp | "
         "FS | FM | nPred / nGT |", "|---" * 14 + "|"]
    per_volume = []
    rows = [lane[part] for lane in state["lanes"].values() for part in ("pre", "ft", "test")] + state.get("extra", [])
    for a in rows:
        s = a.get("scores") or {}
        csv_path = os.path.join(REPO, a["summary_csv"]) if a.get("summary_csv") else None
        if csv_path and os.path.exists(csv_path):
            s = E.read_scores(csv_path)
            per_volume.append((a, E.score_rows(csv_path)))
        best = f"{a['best_ckpt']} ({fmt(a.get('best_val_loss'))})" if "best_ckpt" in a else \
            (f"{a['ckpt']}" if "ckpt" in a else "—")
        npgt = f"{fmt(s.get('num_pred'), 0)} / {fmt(s.get('num_gt'), 1)}" if s else "—"
        L.append(f"| {a['key']} | {a['label']} | `{a.get('exp', '—')}` | {a['status']} | "
                 f"{','.join(a['jobs']) or '—'} | {best} | {('**' + fmt(s['S']) + '**') if s else '—'} | "
                 f"{fmt(s.get('avF1'))} | {fmt(s.get('C'))} | {fmt(s.get('clDiceTP'))} | {fmt(s.get('tp'))} | "
                 f"{fmt(s.get('FS'), 2)} | {fmt(s.get('FM'), 2)} | {npgt} |")
    p2 = state.get("p2") or {}
    L += ["", f"**P2 seam check** (W64 pretrain @ {P2_CKPT}, margin 0 vs 32): {p2.get('status', 'wait')}"
          + (f", predict jobs {','.join(p2.get('jobs', []))}, compare {p2.get('compare')}" if p2.get("jobs") else "")
          + ("; report `metrics/ppp/expv3/p2_seam_check.md`" if p2.get("status") == "done" else "")]
    L += ["", "## Lanes", ""]
    for name, lane in state["lanes"].items():
        pre = lane["pre"]
        L.append(f"- **{name}** stage `{lane['stage']}`; model overlays `{' + '.join(lane['model'])}`"
                 + (f"; pretrain {pre.get('max_it')} its, {pre.get('time')} jobs, chain {pre.get('chain', 0)}"
                    if name != "B" else f"; pretrain `{pre['exp']}` selected on `{pre['val_csv']}`"))
    for name, lane in state["lanes"].items():
        for part in ("pre", "ft"):
            a = lane[part]
            if a.get("val_curve"):
                L += ["", f"**{a['key']} val curve** (`{a['exp']}`, completely val loss): "
                      + ", ".join(f"{it // 1000 if it % 1000 == 0 else it / 1000}k {fmt(v)}" for it, v in a["val_curve"])]
    if per_volume:
        L += ["", "## Per-volume test results", ""]
        for a, rows in per_volume:
            L += [f"**{a['key']}** (`{a.get('exp')}` @ {a.get('ckpt')}; `{a['summary_csv']}`)", "",
                  "| volume | S | avF1 | C | clDiceTP | tp | FS | FM | nPred | nGT |", "|---" * 10 + "|"]
            L += [f"| {r['sample']} | {fmt(r['S'])} | {fmt(r['avF1'])} | {fmt(r['C'])} | {fmt(r['clDiceTP'])} | "
                  f"{fmt(r['tp'])} | {fmt(r['FS'], 0)} | {fmt(r['FM'], 0)} | {fmt(r['num_pred'], 0)} | "
                  f"{fmt(r['num_gt'], 0)} |" for r in rows] + [""]
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
    ap.add_argument("cmd", choices=["tick", "status", "release", "resume-lane", "extra-test"])
    ap.add_argument("--lane", choices=["B", "W64", "W32"])
    ap.add_argument("--ckpt", type=int, help="extra-test: checkpoint to test")
    ap.add_argument("--label", default="", help="extra-test: description")
    ap.add_argument("--iters", type=int)
    ap.add_argument("--time")
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--ft-time", help="finetune job time limit for the lane (default %s)" % FT_TIME)
    args = ap.parse_args()
    DRY = args.dry_run
    E.DRY = DRY
    with locked_state() as state:
        if args.cmd == "tick":
            tick(state)
        elif args.cmd == "release":
            if not (args.lane and args.iters and args.time):
                raise SystemExit("release needs --lane, --iters and --time")
            release(state, args.lane, args.iters, args.time, args.overlay, args.ft_time)
        elif args.cmd == "resume-lane":
            resume_lane(state, args.lane)
        elif args.cmd == "extra-test":
            if not (args.lane and args.ckpt):
                raise SystemExit("extra-test needs --lane and --ckpt")
            add_extra_test(state, args.lane, args.ckpt, args.label)
    if os.path.exists(STATUS_MD) and args.cmd == "status":
        print(open(STATUS_MD).read())


if __name__ == "__main__":
    main()
