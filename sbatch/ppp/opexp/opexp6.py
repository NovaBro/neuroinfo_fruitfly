#!/usr/bin/env python3
"""experiment_v2 OP-EXP 6 driver: LoRA fine-tuning hyperparameter search (md_guides/experiments/experiment_v2.md).

From expv2-e1.1-pre-n1000 @ 20000, fine-tune on completely for 6k its with LoRA adapters (setup01/lora.py) on the
UNet encoder + decoder convs (l_conv, r_conv; the two output heads fully trainable, everything else frozen),
WarmupCosine with 10% warmup, ckpt + val every 1k (completely val = selection, 1k_v1.1 val first 5 = extra).
Arms (user 2026-11-10: rank sweep from the plan at the planned 5e-5 *and* the LoRA-typical 5e-4):
  6.1 / 6.2 / 6.3  r = 16 / 32 / 64, alpha = r, peak lr 5e-5
  6.4 / 6.5 / 6.6  r = 16 / 32 / 64, alpha = r, peak lr 5e-4
Each arm's lowest-completely-val-loss checkpoint is tested on the 7 completely test volumes (sparse pred_affs
pipeline; predict folds the adapters into the weights) with remove_small_components 1500 (eval_rm1500.toml).
Gate: training jobs wait (afterok) on a GPU run of sbatch/ppp/tools/test_lora.py (LoRA had not run on a GPU).

Idempotent state machine; ``opexp6_tick.sh`` runs ``tick`` every ~10 min on a CPU node until done. Uses the
helpers of ``sbatch/ppp/expv2/expv2_e1.py`` (imported as E). From the repo root:
  python3 sbatch/ppp/opexp/opexp6.py tick | status | --dry-run tick
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "sbatch", "ppp", "expv2"))
import expv2_e1 as E  # noqa: E402

ACCOUNT = "torch_pr_61_tandon_priority"  # user 2026-11-10 (general queue ~420 jobs deep)
E.ACCOUNT = ACCOUNT

STATE_DIR = os.path.join(E.METRICS, "opexp6")
STATE = os.path.join(STATE_DIR, "state.json")
STATUS_MD = os.path.join(REPO, "md_guides", "experiments", "opexp6_status.md")
CFG_DIR = os.path.join(REPO, "PatchPerPix", "experiments", "flylight", "setups", "setup01", "configs", "exp", "opexp")

SRC_EXP, SRC_CKPT = "expv2-e1.1-pre-n1000", 20000
FT_ITERS, WARMUP_FRAC = 6000, 0.10
TARGETS = ["l_conv", "r_conv"]
ARMS = [("6.1", 16, 5e-5), ("6.2", 32, 5e-5), ("6.3", 64, 5e-5),
        ("6.4", 16, 5e-4), ("6.5", 32, 5e-4), ("6.6", 64, 5e-4)]
STATIC = E.COMMON + ["expv2_e1_ft.toml", "opexp4_ft.toml", "opexp6_ft.toml"]
TEST_EVAL = "eval_rm1500.toml"
SIF = E.SIF

DRY = False


def log(state, msg):
    E.log(state, msg)


def initial_state():
    arms = {}
    for key, r, lr in ARMS:
        tag = f"r{r}-lr{E.lr_fmt(lr)}"
        arms[key] = {
            "stage": "seed",
            "ft": {"key": key, "exp": f"opexp6-lora-{tag}", "rank": r, "lr": lr, "status": "new", "jobs": [],
                   "retries": 0, "label": f"LoRA r={r} alpha={r}, peak lr {E.lr_fmt(lr)}"},
            "test": {"key": key, "status": "wait", "jobs": [], "retries": 0,
                     "label": f"test completely, best {key} ckpt, rm1500"},
        }
    return {"gate": {"job": None, "status": "new"}, "arms": arms, "events": [], "done": False}


# ----------------------------------------------------------------------------- gate

def submit_gate(state):
    inner = ("source /ext3/env.sh; conda activate ppp; cd PatchPerPix/experiments; "
             "python -u -W ignore::FutureWarning ../../sbatch/ppp/tools/test_lora.py "
             f"--config ../../metrics/ppp/{SRC_EXP}/config.toml "
             f"--checkpoint ../../metrics/ppp/{SRC_EXP}/train/train_net_checkpoint_{SRC_CKPT}")
    cmd = ["sbatch", "--parsable", f"--account={ACCOUNT}", "--partition=l40s_public", "--gres=gpu:1",
           "--cpus-per-task=4", "--mem=48G", "--time=00:45:00", "--job-name=opexp6_test_lora",
           "--output=sbatch/ppp/logs/opexp/opexp6_test_lora-%j.out",
           f"--wrap=singularity exec --nv --overlay env/ppp.ext3:ro {SIF} /bin/bash -c '{inner}'"]
    if DRY:
        log(state, "[dry] would submit LoRA GPU test")
        return
    jid = E.run(cmd).strip().splitlines()[-1].split(";")[0]
    state["gate"] = {"job": jid, "status": "submitted"}
    log(state, f"OP-EXP 6: LoRA GPU test (sbatch/ppp/tools/test_lora.py) job {jid}; training waits afterok on it")


def gate_state(state):
    g = state["gate"]
    if g["status"] in ("passed", "failed") or not g.get("job"):
        return g["status"]
    st, _ = E.job_states([g["job"]]).get(g["job"], ("UNKNOWN", ""))
    if st == "COMPLETED":
        g["status"] = "passed"
        log(state, f"OP-EXP 6: LoRA GPU test {g['job']} passed")
    elif st not in E.ACTIVE:
        g["status"] = "failed"
        log(state, f"OP-EXP 6: LoRA GPU test {g['job']} ended {st}; arms stay queued behind it "
                   "(DependencyNeverSatisfied) -- check sbatch/ppp/logs/opexp/opexp6_test_lora-*.out")
    return g["status"]


# ----------------------------------------------------------------------------- arms

def start_arm(state, ft):
    n = SRC_CKPT
    ft["src_ckpt"], ft["start_it"], ft["max_it"] = n, n, n + FT_ITERS
    gen = f"opexp6_ft_{ft['exp']}.toml"
    with open(os.path.join(CFG_DIR, gen), "w") as fh:
        fh.write(f"# generated by sbatch/ppp/opexp/opexp6.py: OP-EXP {ft['key']} LoRA fine-tune of {SRC_EXP} @ {n}, "
                 f"{FT_ITERS} its, r={ft['rank']} alpha={ft['rank']}, peak lr {ft['lr']:g}.\n"
                 f"[training]\nmax_iterations = {n + FT_ITERS}\n\n"
                 f"[training.lora]\nenabled = true\nrank = {ft['rank']}\nalpha = {float(ft['rank'])}\n"
                 f"targets = {json.dumps(TARGETS)}\ntrain_heads = true\ntrain_bias = false\n\n"
                 f"[optimizer.optimizer]\nlr = {ft['lr']!r}\n\n"
                 f"[optimizer.lr_scheduler]\nstart_iter = {n}\n")
    ft["overlays"] = STATIC + [gen, TEST_EVAL]
    if not DRY:
        dst = os.path.join(E.METRICS, ft["exp"], "train")
        inner = (f"source /ext3/env.sh; conda activate ppp; python sbatch/ppp/expv2/seed_finetune_ckpt.py "
                 f"--src {E.ckpt_path(SRC_EXP, n)} --dst-dir {dst} --iteration {n} --lr {ft['lr']!r} "
                 f"--warmup-frac {WARMUP_FRAC} --ft-iters {FT_ITERS}")
        out = E.run(["singularity", "exec", "--overlay", "env/ppp.ext3:ro", SIF, "/bin/bash", "-c", inner])
        log(state, f"OP-EXP {ft['key']}: " + out.strip().splitlines()[-1])
    cmd = ["./sbatch/ppp/ppp_submit_train.sh"] + (["--dry-run"] if DRY else []) + [
        E.PRESET, "--base", E.BASE, "--id", ft["exp"]]
    for o in ft["overlays"]:
        cmd += ["--config", o]
    cmd += ["--account", ACCOUNT, "--time", E.FT_TIME]
    if state["gate"].get("job") and state["gate"]["status"] == "submitted":
        cmd += ["--dependency", f"afterok:{state['gate']['job']}"]
    out = E.run(cmd)
    m = re.search(r"train=(\S+)", out)
    if DRY or not m:
        log(state, f"[dry] would submit train {ft['key']} {ft['exp']}")
        return
    ft["jobs"].append(m.group(1))
    ft["status"] = "training"
    log(state, f"OP-EXP {ft['key']} ({ft['label']}): train job {m.group(1)} exp={ft['exp']}")


def tick_arm(state, arm):
    ft, te = arm["ft"], arm["test"]
    if arm["stage"] == "seed":
        start_arm(state, ft)
        if ft["status"] == "training":
            arm["stage"] = "train"
    elif arm["stage"] == "train":
        if ft["status"] == "training" and E.advance_training(state, ft, E.FT_TIME):
            E.select_best(state, ft, min_it=ft["start_it"])
            arm["stage"] = "test"
            E.submit_test(state, ft, te)
        if ft["status"] == "failed":
            arm["stage"] = "failed"
    elif arm["stage"] == "test":
        if te["status"] == "wait":
            E.submit_test(state, ft, te)
        elif te["status"] == "testing":
            E.advance_testing(state, ft, te)
        if te["status"] in ("tested", "failed"):
            arm["stage"] = "done" if te["status"] == "tested" else "failed"


def tick(state):
    if state["gate"]["status"] == "new":
        submit_gate(state)
    if gate_state(state) == "failed":
        return  # training jobs depend on the gate; nothing to do until a human looks
    for arm in state["arms"].values():
        try:
            tick_arm(state, arm)
        except Exception as e:
            log(state, f"OP-EXP {arm['ft']['key']}: tick error {type(e).__name__}: {str(e)[:300]}")
    if all(a["stage"] in ("done", "failed") for a in state["arms"].values()) and not state.get("done"):
        state["done"] = True
        log(state, "OP-EXP 6 driver finished")


# ----------------------------------------------------------------------------- status

def write_status(state):
    hdr = "S | avF1 | C | clDiceTP | tp | FS | FM | nPred / nGT"
    L = ["# experiment_v2 OP-EXP 6 status: LoRA fine-tuning (auto-generated by `sbatch/ppp/opexp/opexp6.py`)", "",
         f"Last update: {E.now()}. Plan: `md_guides/experiments/experiment_v2.md`. From `{SRC_EXP}` @ {SRC_CKPT}, "
         f"{FT_ITERS} its, 10% warmup + cosine, LoRA on {', '.join(TARGETS)} (heads fully trainable), "
         f"account `{ACCOUNT}`. Test: lowest completely-val-loss ckpt, 7 completely test vols, "
         "remove_small_components 1500.", "",
         f"LoRA GPU test (`sbatch/ppp/tools/test_lora.py`): job {state['gate'].get('job') or '—'}, "
         f"{state['gate']['status']}.", "",
         f"| Arm | rank (alpha) | peak lr | exp | status | jobs | best ckpt (completely val loss) | "
         f"1k_v1.1 val loss @ best | {hdr} |", "|---" * 16 + "|"]
    for key, arm in state["arms"].items():
        ft, te = arm["ft"], arm["test"]
        s = {}
        if te.get("summary_csv") and os.path.exists(os.path.join(REPO, te["summary_csv"])):
            s = E.read_scores(os.path.join(REPO, te["summary_csv"]))
        best = f"{ft['best_ckpt']} ({E.fmt(ft.get('best_val_loss'))})" if "best_ckpt" in ft else "—"
        syn = "—"
        p = os.path.join(E.METRICS, ft["exp"], "train", "val_metrics_1k_v1.1.csv")
        if "best_ckpt" in ft and os.path.exists(p):
            import csv
            rows = {int(r["iteration"]): r for r in csv.DictReader(open(p))}
            if ft["best_ckpt"] in rows:
                syn = E.fmt(float(rows[ft["best_ckpt"]]["loss"]))
        cells = (f"**{E.fmt(s['S'])}** | {E.fmt(s['avF1'])} | {E.fmt(s['C'])} | {E.fmt(s['clDiceTP'])} | "
                 f"{E.fmt(s['tp'])} | {E.fmt(s['FS'], 2)} | {E.fmt(s['FM'], 2)} | "
                 f"{E.fmt(s['num_pred'], 0)} / {E.fmt(s['num_gt'], 1)}") if s else " | ".join(["—"] * 8)
        status = te["status"] if arm["stage"] in ("test", "done") else ft["status"]
        L.append(f"| {key} | {ft['rank']} ({ft['rank']}) | {E.lr_fmt(ft['lr'])} | `{ft['exp']}` | {status} | "
                 f"{','.join(ft['jobs'] + te['jobs']) or '—'} | {best} | {syn} | {cells} |")
    L += ["", "Test metrics are the FISBe ones (S = 0.5·avF1 + 0.5·C), mean over the 7 completely test volumes.", "",
          "## Event log", ""] + [f"- {e}" for e in state["events"]] + [""]
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
    DRY = E.DRY = args.dry_run
    with locked_state() as state:
        if args.cmd == "tick":
            tick(state)
        if DRY:
            print("\n".join(state["events"][-14:]))
    if args.cmd == "status" and os.path.exists(STATUS_MD):
        print(open(STATUS_MD).read())


if __name__ == "__main__":
    main()
