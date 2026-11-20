#!/usr/bin/env python3
"""experiment_v2 OP-EXP 7 driver: LoRA fine-tuning on completely + a 1k_v1.1 subset (md_guides/experiments/experiment_v2.md).

Same setup as OP-EXP 6 (sbatch/ppp/opexp/opexp6.py): from expv2-e1.1-pre-n1000 @ 20000, 6k its, WarmupCosine 10% warmup,
LoRA on l_conv + r_conv (alpha = r), output heads fully trainable, ckpt + val every 1k (completely val = selection,
1k_v1.1 val first 5 = extra), test the best ckpt on the 7 completely test volumes at remove_small_components 1500,
account torch_pr_61_tandon_priority. New: the training set is the 18 completely train volumes plus the first N sorted
1k_v1.1 train volumes ([data] max_train_samples_per_dataset), drawn uniformly per volume (user 2026-11-11).
Arms (plan):
  7.1 r=2 lr 5e-4 N=18 | 7.2 r=4 lr 5e-4 N=18 | 7.3 r=2 lr 5e-4 N=36 | 7.4 r=4 lr 5e-4 N=36
  7.5 r=2 lr 8e-4 N=36 | 7.6 r=2 lr 8e-4 N=72
The LoRA GPU unit test already passed for OP-EXP 6 (job 22612704), so no gate job here.

Idempotent state machine; ``opexp7_tick.sh`` runs ``tick`` every ~10 min on a CPU node until done. From the repo root:
  python3 sbatch/ppp/opexp/opexp7.py tick | status | --dry-run tick
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

STATE_DIR = os.path.join(E.METRICS, "opexp7")
STATE = os.path.join(STATE_DIR, "state.json")
STATUS_MD = os.path.join(REPO, "md_guides", "experiments", "opexp7_status.md")
CFG_DIR = os.path.join(REPO, "PatchPerPix", "experiments", "flylight", "setups", "setup01", "configs", "exp", "opexp")

SRC_EXP, SRC_CKPT = "expv2-e1.1-pre-n1000", 20000
FT_ITERS, WARMUP_FRAC = 6000, 0.10
TARGETS = ["l_conv", "r_conv"]
SYN_DS = "1k_v1.1"
ARMS = [("7.1", 2, 5e-4, 18), ("7.2", 4, 5e-4, 18), ("7.3", 2, 5e-4, 36),
        ("7.4", 4, 5e-4, 36), ("7.5", 2, 8e-4, 36), ("7.6", 2, 8e-4, 72)]
STATIC = E.COMMON + ["expv2_e1_ft.toml", "opexp4_ft.toml", "opexp6_ft.toml", "opexp7_ft.toml"]
LORA_TEST_JOB = "22612704"  # OP-EXP 6 gate, passed 2026-11-10
# 2026-11-11 (user): the arms sat 6 h behind ~640 higher-priority l40s_public jobs. OP-EXP 6 arms took 2.0-2.5 h, so
# 3 h (shorter jobs backfill more easily). Users cannot scontrol-update Partition/TimeLimit here, so arms are resubmitted.
# h100 / a100 were tried: the job_submit plugin rejects every GPU job there for our accounts ("partition 'h100' is not
# valid for this job ... GPU job setup is not valid"), so l40s_public only.
TRAIN_PARTITION = "l40s_public"
TRAIN_TIME = "03:00:00"
TEST_EVAL = "eval_rm1500.toml"
SIF = E.SIF

DRY = False


def log(state, msg):
    E.log(state, msg)


def initial_state():
    arms = {}
    for key, r, lr, n_syn in ARMS:
        tag = f"r{r}-lr{E.lr_fmt(lr)}-syn{n_syn}"
        arms[key] = {
            "stage": "seed",
            "ft": {"key": key, "exp": f"opexp7-lora-{tag}", "rank": r, "lr": lr, "n_syn": n_syn, "status": "new",
                   "jobs": [], "retries": 0,
                   "label": f"LoRA r={r} alpha={r}, peak lr {E.lr_fmt(lr)}, completely 18 + {SYN_DS} {n_syn}"},
            "test": {"key": key, "status": "wait", "jobs": [], "retries": 0,
                     "label": f"test completely, best {key} ckpt, rm1500"},
        }
    return {"gate": {"job": LORA_TEST_JOB, "status": "passed"}, "arms": arms, "events": [], "done": False}


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
    gen = f"opexp7_ft_{ft['exp']}.toml"
    with open(os.path.join(CFG_DIR, gen), "w") as fh:
        fh.write(f"# generated by sbatch/ppp/opexp/opexp7.py: OP-EXP {ft['key']} LoRA fine-tune of {SRC_EXP} @ {n}, "
                 f"{FT_ITERS} its, r={ft['rank']} alpha={ft['rank']}, peak lr {ft['lr']:g}, "
                 f"completely 18 + {SYN_DS} first {ft['n_syn']}.\n"
                 # regular table, not inline: toml loads inline tables as an unpicklable DynamicInlineTableDict,
                 # which crashed prediction's multiprocessing (2026-11-12)
                 f"[data.max_train_samples_per_dataset]\n\"{SYN_DS}\" = {ft['n_syn']}\n\n"
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
    submit_train(state, ft)


def submit_train(state, ft, time_limit=None):
    """Submit / resubmit an arm's (already seeded) fine-tune on TRAIN_PARTITION for TRAIN_TIME. Also installed as
    E.submit_train, so E.advance_training's resubmissions use the same partition and limit."""
    cmd = ["./sbatch/ppp/ppp_submit_train.sh"] + (["--dry-run"] if DRY else []) + [
        E.PRESET, "--base", E.BASE, "--id", ft["exp"]]
    for o in ft["overlays"]:
        cmd += ["--config", o]
    cmd += ["--account", ft.get("account") or ACCOUNT, "--time", TRAIN_TIME, "--partition", TRAIN_PARTITION]
    if state["gate"].get("job") and state["gate"]["status"] == "submitted":
        cmd += ["--dependency", f"afterok:{state['gate']['job']}"]
    out = E.run(cmd)
    m = re.search(r"train=(\S+)", out)
    if DRY or not m:
        log(state, f"[dry] would submit train {ft['key']} {ft['exp']}")
        return
    ft["jobs"].append(m.group(1))
    ft["status"] = "training"
    log(state, f"OP-EXP {ft['key']} ({ft['label']}): train job {m.group(1)} exp={ft['exp']} "
               f"({TRAIN_PARTITION}, {TRAIN_TIME}, {ft.get('account') or ACCOUNT})")


E.submit_train = submit_train


def resubmit_pending_train(state, arms=None, account=None):
    """Cancel arms whose training job is still PENDING and resubmit them with TRAIN_PARTITION / TRAIN_TIME
    (the seeded checkpoint stays in place; nothing has trained yet). arms: only these keys; account: move them to
    this account (stored as ft["account"], also used for their retries and test chain)."""
    for key, arm in state["arms"].items():
        ft = arm["ft"]
        if arms and key not in arms:
            continue
        if arm["stage"] != "train" or ft["status"] != "training":
            continue
        if account:
            ft["account"] = account
        if not ft["jobs"]:  # cancelled, resubmission failed earlier
            submit_train(state, ft)
            continue
        jid = ft["jobs"][-1]
        st, _ = E.job_states([jid]).get(jid, ("UNKNOWN", ""))
        if st != "PENDING":
            continue
        if not DRY:
            subprocess.run(["scancel", jid])
        ft.setdefault("cancelled_jobs", []).append(jid)
        ft["jobs"] = ft["jobs"][:-1]
        log(state, f"OP-EXP {key}: cancelled pending train job {jid} to resubmit on {TRAIN_PARTITION}, {TRAIN_TIME}, "
                   f"{ft.get('account') or ACCOUNT}")
        submit_train(state, ft)


def tick_arm(state, arm):
    # per-arm account (user 2026-11-11: arms spread over general / tandon_advanced / tandon_priority) for the
    # E.submit_* calls below (retries, test chain); restored afterwards
    E.ACCOUNT = arm["ft"].get("account") or ACCOUNT
    try:
        _tick_arm(state, arm)
    finally:
        E.ACCOUNT = ACCOUNT


def _tick_arm(state, arm):
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
    if gate_state(state) == "failed":
        return  # training jobs depend on the gate; nothing to do until a human looks
    for arm in state["arms"].values():
        try:
            tick_arm(state, arm)
        except Exception as e:
            log(state, f"OP-EXP {arm['ft']['key']}: tick error {type(e).__name__}: {str(e)[:300]}")
    if all(a["stage"] in ("done", "failed") for a in state["arms"].values()) and not state.get("done"):
        state["done"] = True
        log(state, "OP-EXP 7 driver finished")


# ----------------------------------------------------------------------------- status

def write_status(state):
    hdr = "S | avF1 | C | clDiceTP | tp | FS | FM | nPred / nGT"
    L = ["# experiment_v2 OP-EXP 7 status: LoRA fine-tuning on completely + 1k_v1.1 subset "
         "(auto-generated by `sbatch/ppp/opexp/opexp7.py`)", "",
         f"Last update: {E.now()}. Plan: `md_guides/experiments/experiment_v2.md`. From `{SRC_EXP}` @ {SRC_CKPT}, "
         f"{FT_ITERS} its, 10% warmup + cosine, LoRA on {', '.join(TARGETS)} (heads fully trainable), training set = "
         f"completely train (18) + first N sorted {SYN_DS} train volumes, drawn uniformly per volume, "
         f"account `{ACCOUNT}`. Test: lowest completely-val-loss ckpt, 7 completely test vols, "
         "remove_small_components 1500.", "",
         f"LoRA GPU test (`sbatch/ppp/tools/test_lora.py`): job {state['gate'].get('job') or '—'} (OP-EXP 6), "
         f"{state['gate']['status']}. Reference (rm1500): full finetune EXP 1.3 S 0.305; OP-EXP 6 LoRA arms S 0.266-0.291.", "",
         f"| Arm | rank (alpha) | peak lr | {SYN_DS} N | exp | status | jobs | best ckpt (completely val loss) | "
         f"1k_v1.1 val loss @ best | {hdr} |", "|---" * 17 + "|"]
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
        L.append(f"| {key} | {ft['rank']} ({ft['rank']}) | {E.lr_fmt(ft['lr'])} | {ft['n_syn']} | `{ft['exp']}` | {status} | "
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
    ap.add_argument("cmd", choices=["tick", "status", "resubmit-pending-train"])
    ap.add_argument("--arms", help="resubmit-pending-train: comma-separated arm keys, e.g. 7.3,7.4")
    ap.add_argument("--account", help="resubmit-pending-train: move those arms to this account")
    args = ap.parse_args()
    DRY = E.DRY = args.dry_run
    with locked_state() as state:
        if args.cmd == "resubmit-pending-train":
            resubmit_pending_train(state, arms=args.arms.split(",") if args.arms else None, account=args.account)
        if args.cmd == "tick":
            tick(state)
        if DRY:
            print("\n".join(state["events"][-14:]))
    if args.cmd == "status" and os.path.exists(STATUS_MD):
        print(open(STATUS_MD).read())


if __name__ == "__main__":
    main()
