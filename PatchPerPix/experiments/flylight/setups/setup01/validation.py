"""Periodic validation during PPP training: val loss + simple voxel metrics.

Replaces gunpowder's interleaved ``TrainValProvider`` validation (one random
batch every ``val_log_step`` iterations, logged only through TensorBoard,
which the ppp overlay does not have, so its loss ended up as an ordinary
``Train process`` line inside the train-loss curve).

Every ``val_log_step`` train iterations ``train.py`` calls
:meth:`Validator.run`, which evaluates a **fixed** set of val crops
(``val_num_batches`` x ``batch_size``, gunpowder request seeds
``val_seed + i``) with the current weights (eval mode, no grad, val padding)
and writes:

* one row to ``<output_folder>/val_metrics.csv`` (``<exp>/train/val_metrics.csv``)
* one ``Validation: iteration=N loss=... fg_dice=...`` line to the log/run.log
* TensorBoard scalars under ``<exp>/validate`` if ``torch.utils.tensorboard``
  imports (it does not in the current overlay)

Metrics are cheap voxel-level ones computed with plain torch ops (no
torchmetrics: its ``unique`` hit CUB size limits on dense patch tensors).
Instance metrics (avF1, C, FM, FS, ...) stay in ``-d validate_checkpoints``.

* loss, loss_patch, loss_fg: the training ``LossWrapper`` terms, averaged over batches
* fg_*: foreground (numinst argmax > 0, or fg sigmoid > 0.5) vs gt > 0
* numinst_acc_fg: numinst class accuracy over voxels where gt > 0 or pred > 0
* overlap_precision / overlap_recall: voxels with >= 2 instances (overlapping_inst)
* aff_*: patch affinities (sigmoid > 0.5 vs gt) over gt-foreground voxels
  (noenc) or over the sampled patches (train_code)
"""
import csv
import logging
import math
import os
import random
import time

import numpy as np
import torch

import gunpowder as gp

from . import torch_loss

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "iteration", "n_samples",
    "loss", "loss_patch", "loss_fg",
    "fg_dice", "fg_iou", "fg_precision", "fg_recall",
    "numinst_acc_fg", "overlap_precision", "overlap_recall",
    "aff_acc", "aff_precision", "aff_recall", "aff_f1",
    "val_time_s",
]

COUNT_KEYS = [
    "fg_tp", "fg_fp", "fg_fn", "fg_tn",
    "ni_union", "ni_correct",
    "ov_tp", "ov_pred", "ov_gt",
    "aff_tp", "aff_fp", "aff_fn", "aff_tn",
]


# --------------------------------------------------------------------------
# metric helpers (pure torch; unit-testable on CPU)
# --------------------------------------------------------------------------

def _drop_singleton_channel(x, ref_dim):
    """(B, 1, *S) -> (B, *S) when x has one dim more than the reference."""
    if x.dim() == ref_dim + 1 and x.size(1) == 1:
        return x.squeeze(1)
    return x


def _mask_to_bool(mask, ref_dim):
    """Mirror the loss masks: MaskedBCE uses channel 0 if C == 1 else 1."""
    if mask is None:
        return None
    if mask.dim() == ref_dim + 1:
        mask = mask[:, 0 if mask.size(1) == 1 else 1]
    return mask > 0


def _confusion(pred, gt):
    """Counts of a boolean prediction vs boolean target."""
    tp = int(torch.count_nonzero(pred & gt))
    p = int(torch.count_nonzero(pred))
    g = int(torch.count_nonzero(gt))
    n = int(pred.numel())
    return tp, p - tp, g - tp, n - p - g + tp


@torch.no_grad()
def batch_counts(pred_logits_affs, pred_logits_fg, gt_affs_samples,
                 gt_fgbg_loss, loss_mask=None, *,
                 overlapping_inst=True, train_code=False):
    """Confusion counts for one batch, from the LossWrapper inputs.

    Args mirror ``torch_loss.LossWrapper.forward``:
        pred_logits_affs: (B, P, *S) patch logits (noenc) or (N, 1, *patch)
            sampled patch logits (train_code).
        pred_logits_fg: (B, K+1, *S) numinst logits (overlapping_inst) or
            (B, 1, *S) fg logits.
        gt_affs_samples: same shape as pred_logits_affs, values in {0, 1}.
        gt_fgbg_loss: (B, *S) numinst classes (clamped) or (B[, 1], *S) fg.
        loss_mask: optional (B[, C], *S) mask, > 0 = counted.
    """
    counts = dict.fromkeys(COUNT_KEYS, 0)

    if overlapping_inst:
        pred_cls = torch.argmax(pred_logits_fg, dim=1)
        gt_cls = _drop_singleton_channel(gt_fgbg_loss, pred_cls.dim()).long()
    else:
        pred_cls = _drop_singleton_channel(pred_logits_fg > 0, pred_logits_fg.dim() - 1)
        gt_cls = _drop_singleton_channel(gt_fgbg_loss, pred_cls.dim()) > 0.5
        pred_cls = pred_cls.long()
        gt_cls = gt_cls.long()
    mask = _mask_to_bool(loss_mask, pred_cls.dim())

    pred_fg = pred_cls > 0
    gt_fg = gt_cls > 0
    if mask is not None:
        sel_pred, sel_gt = pred_fg[mask], gt_fg[mask]
    else:
        sel_pred, sel_gt = pred_fg, gt_fg
    (counts["fg_tp"], counts["fg_fp"],
     counts["fg_fn"], counts["fg_tn"]) = _confusion(sel_pred, sel_gt)

    if overlapping_inst:
        union = pred_fg | gt_fg
        if mask is not None:
            union = union & mask
        counts["ni_union"] = int(torch.count_nonzero(union))
        counts["ni_correct"] = int(torch.count_nonzero(union & (pred_cls == gt_cls)))
        ov_pred = pred_cls >= 2
        ov_gt = gt_cls >= 2
        if mask is not None:
            ov_pred = ov_pred & mask
            ov_gt = ov_gt & mask
        counts["ov_tp"] = int(torch.count_nonzero(ov_pred & ov_gt))
        counts["ov_pred"] = int(torch.count_nonzero(ov_pred))
        counts["ov_gt"] = int(torch.count_nonzero(ov_gt))

    if pred_logits_affs is not None and pred_logits_affs.numel() > 0:
        if train_code:
            # decoder path: every row is a patch sampled at a gt-fg voxel
            tp, fp, fn, tn = _confusion(pred_logits_affs > 0, gt_affs_samples > 0.5)
            counts["aff_tp"] += tp
            counts["aff_fp"] += fp
            counts["aff_fn"] += fn
            counts["aff_tn"] += tn
        else:
            # dense path: per sample, so the (P, *S) selection stays small
            for b in range(pred_logits_affs.size(0)):
                sel = gt_fg[b]
                if mask is not None:
                    sel = sel & mask[b]
                if not bool(sel.any()):
                    continue
                p = pred_logits_affs[b][:, sel] > 0
                g = gt_affs_samples[b][:, sel] > 0.5
                tp, fp, fn, tn = _confusion(p, g)
                counts["aff_tp"] += tp
                counts["aff_fp"] += fp
                counts["aff_fn"] += fn
                counts["aff_tn"] += tn
    return counts


def _ratio(num, den):
    return num / den if den > 0 else float("nan")


def finalize_metrics(counts, overlapping_inst=True):
    """Turn accumulated counts into the metric columns (NaN if undefined)."""
    c = counts
    m = {
        "fg_dice": _ratio(2 * c["fg_tp"], 2 * c["fg_tp"] + c["fg_fp"] + c["fg_fn"]),
        "fg_iou": _ratio(c["fg_tp"], c["fg_tp"] + c["fg_fp"] + c["fg_fn"]),
        "fg_precision": _ratio(c["fg_tp"], c["fg_tp"] + c["fg_fp"]),
        "fg_recall": _ratio(c["fg_tp"], c["fg_tp"] + c["fg_fn"]),
        "aff_acc": _ratio(c["aff_tp"] + c["aff_tn"],
                          c["aff_tp"] + c["aff_tn"] + c["aff_fp"] + c["aff_fn"]),
        "aff_precision": _ratio(c["aff_tp"], c["aff_tp"] + c["aff_fp"]),
        "aff_recall": _ratio(c["aff_tp"], c["aff_tp"] + c["aff_fn"]),
        "aff_f1": _ratio(2 * c["aff_tp"], 2 * c["aff_tp"] + c["aff_fp"] + c["aff_fn"]),
    }
    if overlapping_inst:
        m["numinst_acc_fg"] = _ratio(c["ni_correct"], c["ni_union"])
        m["overlap_precision"] = _ratio(c["ov_tp"], c["ov_pred"])
        m["overlap_recall"] = _ratio(c["ov_tp"], c["ov_gt"])
    else:
        m["numinst_acc_fg"] = float("nan")
        m["overlap_precision"] = float("nan")
        m["overlap_recall"] = float("nan")
    return m


def _stack(arrays):
    if isinstance(arrays[0], torch.Tensor):
        return torch.stack(arrays)
    return np.stack(arrays)


# --------------------------------------------------------------------------
# Validator
# --------------------------------------------------------------------------

class Validator:
    """Fixed-subset validation run from the training loop.

    Args:
        model: the training ``UnetModelWrapper`` (the same object the
            ``TorchTrainExt`` node trains and restores checkpoints into).
        config: flat training config dict (as given to ``train_until``).
        val_pipeline: built gunpowder pipeline yielding *single* (unstacked)
            val samples; it must not contain ``PreCache``/``Stack`` (both
            replace the request's random seed).
        val_request: per-sample ``gp.BatchRequest``.
        inputs / outputs / loss_inputs: the dicts passed to ``TorchTrainExt``.
        upstream_lock: optional lock shared with the train pipeline's
            ``ThreadedPrefetch``: held while sampling so the prefetch thread
            cannot reseed ``random`` / ``np.random`` mid-request.
    """

    def __init__(self, model, config, val_pipeline, val_request, inputs,
                 outputs, loss_inputs, upstream_lock=None, name=None):
        # name: extra val set ([data] extra_val_datasets) -> val_metrics_<name>.csv
        # and "Validation[<name>]:" log lines; None = the main val set.
        self.name = name
        self.model = model
        self.config = config
        self.pipeline = val_pipeline
        self.request = val_request
        self.inputs = inputs
        self.outputs = outputs
        self.loss_inputs = loss_inputs
        self.lock = upstream_lock

        self.device = next(model.parameters()).device
        self.num_batches = int(config.get("val_num_batches", 4))
        self.batch_size = int(config["batch_size"])
        self.seed = int(config.get("val_seed", 0))
        self.cache = bool(config.get("val_cache_batches", True))
        self.amp = bool(config.get("use_auto_mixed_precision", False))
        self.overlapping_inst = bool(config.get("overlapping_inst"))
        self.train_code = bool(config.get("train_code"))
        self.train_padding = config.get("train_padding")
        self.val_padding = config.get("val_padding")
        self.switch_padding = (self.val_padding is not None and
                               self.val_padding != self.train_padding)
        if config.get("use_swa"):
            logger.warning("validation: use_swa is set but val uses the raw "
                           "(non-averaged) model")

        # separate instance so the train loss's step counter is untouched
        self.loss = torch_loss.LossWrapper(config).to(self.device)

        suffix = f"_{name}" if name else ""
        self.csv_path = os.path.join(config["output_folder"],
                                     f"val_metrics{suffix}.csv")
        self.log_prefix = f"Validation[{name}]: " if name else "Validation: "
        self._cached = None
        self.summary_writer = None
        try:
            from torch.utils import tensorboard
            self.summary_writer = tensorboard.SummaryWriter(
                os.path.join(config["output_folder"], "..", f"validate{suffix}"))
        except ImportError:
            pass

        # model input / loss input array keys pulled from the batch
        out_keys = set(outputs.values())
        self.batch_keys = []
        for key in list(inputs.values()) + list(loss_inputs.values()):
            if isinstance(key, gp.ArrayKey) and key not in out_keys \
                    and key not in self.batch_keys:
                self.batch_keys.append(key)

        logger.info(
            "validation: %d batches x %d samples every %s its (seed %d, cache %s) -> %s",
            self.num_batches, self.batch_size, config.get("val_log_step"),
            self.seed, self.cache, self.csv_path)

    # -- data -------------------------------------------------------------
    def _request_sample(self, seed):
        req = self.request.copy()
        req._random_seed = seed
        return self.pipeline.request_batch(req)

    def _load_batches(self):
        """Request the fixed val samples (deterministic given the seeds)."""
        py_state, np_state = random.getstate(), np.random.get_state()
        if self.lock is not None:
            self.lock.acquire()
        try:
            batches = []
            for k in range(self.num_batches):
                samples = [
                    self._request_sample(self.seed + k * self.batch_size + j)
                    for j in range(self.batch_size)]
                batches.append({
                    key: _stack([s[key].data for s in samples])
                    for key in self.batch_keys})
        finally:
            if self.lock is not None:
                self.lock.release()
            random.setstate(py_state)
            np.random.set_state(np_state)
        # identical across loads/runs <=> same val crops (val_cache_batches=false
        # reloads every val point, so this doubles as a determinism check)
        checksum = sum(float(b[k].sum(dtype=np.float64))
                       for b in batches for k in self.batch_keys
                       if isinstance(b[k], np.ndarray))
        logger.info("validation: loaded %d batches, checksum %.6e",
                    len(batches), checksum)
        return batches

    def _batches(self):
        if not self.cache:
            return self._load_batches()
        if self._cached is None:
            t0 = time.time()
            self._cached = self._load_batches()
            nbytes = sum(v.nbytes for b in self._cached for v in b.values()
                         if isinstance(v, np.ndarray))
            logger.info("validation: cached %d val batches (%.2f GB) in %.1fs",
                        len(self._cached), nbytes / 1e9, time.time() - t0)
        return self._cached

    # -- forward ----------------------------------------------------------
    def _forward(self, data):
        dev = {k: torch.as_tensor(v, device=self.device) for k, v in data.items()}
        model_kwargs = {name: dev[key] for name, key in self.inputs.items()
                        if key in dev and name not in self.loss_inputs}
        with torch.cuda.amp.autocast(enabled=self.amp):
            model_out = self.model(**model_kwargs)
        if isinstance(model_out, torch.Tensor):
            model_out = (model_out,)
        by_key = {key: model_out[idx] for idx, key in self.outputs.items()
                  if isinstance(idx, int) and idx < len(model_out)}
        loss_kwargs = {}
        for name, key in self.loss_inputs.items():
            if key in by_key:
                loss_kwargs[name] = by_key[key]
            elif key in dev:
                loss_kwargs[name] = dev[key]
        with torch.cuda.amp.autocast(enabled=self.amp):
            loss_out = self.loss(**loss_kwargs)
        return loss_kwargs, loss_out

    @torch.no_grad()
    def run(self, iteration):
        """Evaluate the fixed val set with the current weights; log + CSV."""
        t0 = time.time()
        batches = self._batches()
        loss_sums = {"loss": 0.0, "loss_patch": 0.0, "loss_fg": 0.0}
        counts = dict.fromkeys(COUNT_KEYS, 0)
        n_samples = 0

        was_training = self.model.training
        self.model.eval()
        if self.switch_padding:
            self.model.set_padding(self.val_padding)
        try:
            for data in batches:
                loss_kwargs, loss_out = self._forward(data)
                summ = loss_out["summaries"]
                loss_sums["loss"] += float(loss_out["loss"])
                loss_sums["loss_patch"] += float(summ["loss_patch"][0])
                loss_sums["loss_fg"] += float(summ["loss_fg"][0])
                c = batch_counts(
                    loss_kwargs.get("pred_logits_affs"),
                    loss_kwargs["pred_logits_fg"],
                    loss_kwargs.get("gt_affs_samples"),
                    loss_kwargs["gt_fgbg_loss"],
                    loss_kwargs.get("loss_mask"),
                    overlapping_inst=self.overlapping_inst,
                    train_code=self.train_code)
                for k, v in c.items():
                    counts[k] += v
                n_samples += self.batch_size
                del loss_kwargs, loss_out
        finally:
            if self.switch_padding:
                self.model.set_padding(self.train_padding)
            if was_training:
                self.model.train()

        nb = max(1, len(batches))
        row = {"iteration": int(iteration), "n_samples": n_samples}
        row.update({k: v / nb for k, v in loss_sums.items()})
        row.update(finalize_metrics(counts, self.overlapping_inst))
        row["val_time_s"] = time.time() - t0
        self._write(row)
        return row

    # -- output -----------------------------------------------------------
    def _write(self, row):
        logger.info(self.log_prefix + " ".join(
            f"{k}={row[k]}" if k in ("iteration", "n_samples")
            else f"{k}={row[k]:.6g}" for k in CSV_COLUMNS))

        header = CSV_COLUMNS
        new_file = not os.path.exists(self.csv_path) or \
            os.path.getsize(self.csv_path) == 0
        if not new_file:
            with open(self.csv_path, newline="") as f:
                existing = next(csv.reader(f), None)
            if existing and existing != CSV_COLUMNS:
                logger.warning("validation: %s has a different header; "
                               "appending with its columns", self.csv_path)
                header = existing
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v)
                        for k, v in row.items()})

        if self.summary_writer is not None:
            for k, v in row.items():
                if k in ("iteration", "n_samples"):
                    continue
                if not (isinstance(v, float) and math.isnan(v)):
                    self.summary_writer.add_scalar(k, v, row["iteration"])
            self.summary_writer.flush()
