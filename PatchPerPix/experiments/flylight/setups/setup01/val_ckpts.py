"""Train-time validation (validation.py) on saved checkpoints of a finished run.

Runs the same fixed-subset :class:`Validator` that ``train.py`` runs every
``val_log_step`` iterations, but offline: for each ``train_net_checkpoint_N``
of an experiment it loads the weights and evaluates the val split of one or
more datasets. Each dataset gets its own CSV, named like a ``[data]
extra_val_datasets`` set (``ppp_train_track.py`` reads both):

    <out-dir>/val_metrics_<dataset>.csv     (out-dir defaults to <exp>/train)

Use it to add a val set a run did not log while training, e.g. completely val
for a 1k_v1.1 pretrain and 1k_v1.1 val for its completely finetune.

The val crops are the ones train.py would draw for that dataset (same file list
as run_ppp ``extra_val_datasets``: ``<fisbe_root>/<dataset>/val``, first
``max_extra_val_samples`` or ``max_val_samples`` files; request seeds
``val_seed + i``), so an offline CSV is directly comparable to another run's
train-time CSV for that dataset (the "validation: loaded N batches, checksum"
log line matches). The model runs eager (no ``training.compile``), so rows can
differ from train-time ones in the last digits.

Checkpoints whose iteration is already in the CSV are skipped, so a re-run only
fills in missing ones.

Run from ``PatchPerPix/experiments`` inside the ppp container, with the
datasets' SquashFS images mounted (``sbatch/ppp/val_ckpts_sbatch.sh`` does this):

    python -u -m flylight.setups.setup01.val_ckpts \\
        --exp ../../metrics/ppp/expv2-e1.1-pre-n1000 --datasets completely
"""
import argparse
import csv
import logging
import os
import re
import sys
import time
from copy import deepcopy

import numpy as np
import toml
import torch

import gunpowder as gp

from . import torch_model
from .lora import merge_lora_state_dict
from .train import get_sources
from .validation import Validator

logger = logging.getLogger(__name__)


def list_checkpoints(train_folder, name="train_net"):
    """Sorted iterations of ``<train_folder>/<name>_checkpoint_<N>``."""
    pat = re.compile(r"^%s_checkpoint_(\d+)$" % re.escape(name))
    its = []
    for fn in os.listdir(train_folder):
        m = pat.match(fn)
        if m:
            its.append(int(m.group(1)))
    return sorted(its)


def done_iterations(csv_path):
    """Iterations already present in a val_metrics CSV."""
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {int(r["iteration"]) for r in csv.DictReader(f)
                if r.get("iteration")}


def flat_config(config, output_folder):
    """The keyword dict run_ppp.train() passes to train_until."""
    return dict(
        name=config['model']['train_net_name'],
        max_iteration=config['training']['max_iterations'],
        output_folder=output_folder,
        data_files=[],
        val_files=[],
        extra_val_files={},
        **config['data'],
        **config['model'],
        **config['training'],
        **config['optimizer'],
        **config.get('preprocessing', {}))


def val_files_for(config, dataset):
    """Val files train.py would use for ``dataset`` as an extra val set."""
    import run_ppp  # experiments/ (cwd); same helper run_ppp.train() uses
    cfg = deepcopy(config)
    cfg['data']['val_datasets'] = [dataset]
    if cfg['data'].get('max_extra_val_samples') is not None:
        cfg['data']['max_val_samples'] = cfg['data']['max_extra_val_samples']
    files = run_ppp.get_list_train_files(cfg, val=True)
    if not files:
        raise ValueError("no val files for dataset %s" % dataset)
    return files


def build_model(config, device):
    """Model as train_until builds it (eager), padding left at train_padding.

    Returns (model, input_shape_val, output_shape_val)."""
    model = torch_model.UnetModelWrapper(config, device)
    model.init_layers()
    model = model.to(device)
    model = model.use_channels_last()
    if config.get("network_style", "unet").lower() == "unet":
        model.set_padding(config['val_padding'])
        test_input_shape = (config["test_input_shape_valid"]
                            if config['val_padding'] == 'valid'
                            else config["test_input_shape_same"])
        input_shape_val, output_shape_val = model.inout_shapes(
            test_input_shape, "test_net", training=False)
        model.set_padding(config['train_padding'])
    else:
        input_shape_val = config["test_input_shape_same"]
        output_shape_val = config["test_input_shape_same"]
    model.eval()
    return model, input_shape_val, output_shape_val


def build_val(config, input_shape_val, output_shape_val):
    """Val request, a val-pipeline factory and the TorchTrainExt key dicts,
    mirroring train_until (validation-only parts)."""
    add_affinities = config.get("add_affinities", "cpu")
    assert add_affinities in ["cpu", "torch", "loss"], add_affinities
    overlapping_inst = config.get('overlapping_inst')
    train_code = config.get("train_code")
    add_partly = config.get('add_partly', False)
    sample_fg_points = config.get("sample_fg_points", False)

    raw = gp.ArrayKey('RAW')
    gt_instances = gp.ArrayKey('GT_INSTANCES')
    gt_labels = gp.ArrayKey('GT_LABELS')
    gt_affs = gp.ArrayKey('GT_AFFS') if add_affinities in ["cpu", "torch"] else None
    if sample_fg_points:
        gt_fg_points = gp.GraphKey('GT_FG_POINTS')
    else:
        gt_sample_mask = gp.ArrayKey('GT_SAMPLE_MASK')
    gt_close_to_overlap = gp.GraphKey('GT_CLOSE_TO_OVERLAP')
    gt_affs_samples = gp.ArrayKey('GT_AFFS_SAMPLES')
    pred_affs = gp.ArrayKey('PRED_AFFS')
    if overlapping_inst:
        pred_numinst = gp.ArrayKey('PRED_NUMINST')
        gt_numinst = gp.ArrayKey('GT_NUMINST')
    else:
        pred_fgbg = gp.ArrayKey('PRED_FGBG')
    gt_fgbg = gp.ArrayKey('GT_FGBG')
    gt_fgbg_loss = gp.ArrayKey('GT_FGBG_LOSS')
    if add_partly:
        gt_fgbg_extra = gp.ArrayKey('GT_FGBG_EXTRA')
        loss_mask = gp.ArrayKey('LOSS_MASK')
    if train_code:
        pred_code = gp.ArrayKey('PRED_CODE')

    voxel_size = gp.Coordinate(config["voxel_size"])
    is_3d = len(voxel_size) == 3

    val_input_size = gp.Coordinate(input_shape_val) * voxel_size
    val_output_size = gp.Coordinate(output_shape_val) * voxel_size
    val_request = gp.BatchRequest()
    val_request.add(raw, val_input_size)
    if add_affinities in ["cpu", "torch"]:
        val_request.add(gt_affs, val_output_size)
    if overlapping_inst:
        if add_affinities == "loss":
            val_request.add(gt_instances, gp.Coordinate(
                (o+p-1 for o, p in zip(
                    val_output_size, config['patchshape'][-len(voxel_size):]))))
        else:
            val_request.add(gt_instances, val_output_size)
        val_request.add(gt_numinst, val_output_size)
    else:
        val_request.add(gt_labels, val_output_size)
        val_request.add(gt_fgbg, val_output_size)
    if add_partly:
        val_request.add(loss_mask, val_output_size)

    arrays = {'raw': raw}
    graphs = {'gt_close_to_overlap': gt_close_to_overlap}
    if overlapping_inst:
        arrays['gt_numinst'] = gt_numinst
        arrays['gt_instances'] = gt_instances
    else:
        arrays['gt_fgbg'] = gt_fgbg
        arrays['gt_labels'] = gt_labels
    if add_partly:
        arrays['gt_fgbg_extra'] = gt_fgbg_extra
        arrays['loss_mask'] = loss_mask
    if sample_fg_points:
        graphs['gt_fg_points'] = gt_fg_points
    else:
        arrays['gt_sample_mask'] = gt_sample_mask

    neighborhood = []
    psH = np.array(config['patchshape'])//2
    for k in range(-psH[0], psH[0]+1, config['patchstride'][0]):
        for i in range(-psH[1], psH[1]+1, config['patchstride'][1]):
            for j in range(-psH[2], psH[2]+1, config['patchstride'][2]):
                if is_3d:
                    neighborhood.append([k, i, j])
                else:
                    neighborhood.append([i, j])

    augment = config.get("augmentation", {})
    overlay_on = (augment.get("overlay") is not None and
                  augment.get("overlay") is not False)

    def make_pipeline(files):
        """train.py _make_val_pipeline over get_sources(files, val=True)."""
        val_sources, gt_max = get_sources(
            config, arrays, voxel_size, files, val=True, graphs=graphs)
        return (
            val_sources +
            (gp.IntensityScaleShift(raw, 2, -1)
                if config.get("shift_intensity", False) else gp.NoOp()) +
            (gp.Reject(
                gt_numinst if overlapping_inst else gt_fgbg,
                min_masked=config["reject_min_masked"],
                reject_probability=1,
                mask_filter=1 if overlapping_inst else None
                ) if config.get("reject_min_masked", 0) > 0 else gp.NoOp()) +
            (gp.ZeroPadChannels(
                gt_instances, gt_max if not overlay_on else gt_max*2
            ) if overlapping_inst else gp.NoOp()) +
            (gp.AddAffinities(
                neighborhood,
                gt_instances if overlapping_inst else gt_labels,
                gt_affs,
                multiple_labels=overlapping_inst,
                torch=add_affinities == "torch",
                dtype=np.float32)
             if add_affinities in ["cpu", "torch"] else gp.NoOp())
        )

    inputs = {'raw': raw}
    if add_affinities in ["cpu", "torch"]:
        inputs['gt_affs'] = gt_affs
    else:
        inputs['gt_labels'] = gt_instances if overlapping_inst else gt_labels
    outputs = {0: pred_affs, 2: gt_affs_samples, 3: gt_fgbg_loss}
    loss_inputs = {'pred_logits_affs': pred_affs,
                   'gt_affs_samples': gt_affs_samples}
    if train_code:
        outputs[4] = pred_code
    if overlapping_inst:
        inputs['gt_fgbg_numinst'] = gt_numinst
        loss_inputs['gt_fgbg_loss'] = gt_fgbg_loss
        outputs[1] = pred_numinst
        loss_inputs['pred_logits_fg'] = pred_numinst
    else:
        inputs['gt_fgbg_numinst'] = gt_fgbg
        loss_inputs['gt_fgbg_loss'] = gt_fgbg_loss
        outputs[1] = pred_fgbg
        loss_inputs['pred_logits_fg'] = pred_fgbg
    if add_partly:
        loss_inputs['loss_mask'] = loss_mask

    return val_request, make_pipeline, inputs, outputs, loss_inputs


def load_weights(model, path, device, use_swa=False):
    ckpt = torch.load(path, map_location=device)
    key = "swa_model_state_dict" if use_swa else "model_state_dict"
    model.load_state_dict(merge_lora_state_dict(ckpt[key]))
    del ckpt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--exp", required=True,
                   help="experiment dir (with config.toml and train/)")
    p.add_argument("--datasets", nargs="+", required=True,
                   help="fisbe/<dataset> names whose val split to evaluate")
    p.add_argument("--checkpoints", nargs="+", type=int, default=None,
                   help="iterations (default: every train_net_checkpoint_N)")
    p.add_argument("--out-dir", default=None,
                   help="CSV folder (default <exp>/train)")
    p.add_argument("--config", default=None,
                   help="config TOML (default <exp>/config.toml)")
    args = p.parse_args(argv)

    exp = os.path.normpath(args.exp)
    train_folder = os.path.join(exp, "train")
    out_dir = args.out_dir or train_folder
    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(os.path.join(out_dir, "val_ckpts.log"))],
        force=True)

    config_toml = toml.load(args.config or os.path.join(exp, "config.toml"))
    config = flat_config(config_toml, out_dir)
    if config.get("use_swa"):
        logger.warning("use_swa set: evaluating swa_model_state_dict")

    available = list_checkpoints(train_folder, config["name"])
    ckpts = available if args.checkpoints is None else args.checkpoints
    missing = [c for c in ckpts if c not in available]
    if missing:
        raise FileNotFoundError("no checkpoint(s) %s in %s" % (missing, train_folder))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("GPU: %s", torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available() else "none")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, in_val, out_val = build_model(config, device)
    val_request, make_pipeline, inputs, outputs, loss_inputs = build_val(
        config, in_val, out_val)

    # one Validator per dataset; skip checkpoints already in its CSV
    todo = {}
    pipelines = {}
    for ds in args.datasets:
        files = val_files_for(config_toml, ds)
        logger.info("val set %s: %d files (%s ...)", ds, len(files), files[0])
        pipelines[ds] = make_pipeline(files)
        csv_path = os.path.join(out_dir, "val_metrics_%s.csv" % ds)
        done = done_iterations(csv_path)
        todo[ds] = [c for c in ckpts if c not in done]
        logger.info("val set %s -> %s: %d checkpoints to do, %d already done",
                    ds, csv_path, len(todo[ds]), len(ckpts) - len(todo[ds]))

    for ds in args.datasets:
        if not todo[ds]:
            continue
        with gp.build(pipelines[ds]):
            validator = Validator(model, config, pipelines[ds], val_request,
                                  inputs=inputs, outputs=outputs,
                                  loss_inputs=loss_inputs, name=ds)
            for it in todo[ds]:
                t0 = time.time()
                load_weights(model, os.path.join(
                    train_folder, "%s_checkpoint_%d" % (config["name"], it)),
                    device, use_swa=config.get("use_swa", False))
                model.eval()
                row = validator.run(it)
                logger.info("%s @ %d: loss %.6g fg_dice %.4f aff_f1 %.4f (%.1fs)",
                            ds, it, row["loss"], row["fg_dice"], row["aff_f1"],
                            time.time() - t0)
        del validator
    logger.info("done")


if __name__ == "__main__":
    main()
