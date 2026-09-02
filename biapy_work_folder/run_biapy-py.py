import os
import copy
import random
import shutil
import tempfile
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from tqdm import tqdm
from biapy import BiaPy

CONFIG_DIR = Path("biapy_work_folder/configs")
RESULT_DIR = Path('metrics/biapy')
TRAIN_PARTITION_SIZE = 18

# Change to load weight safety!
import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load


def _biapy_write_fullvolume_instance_gt(filepath, cfg, tag, dtype_str):
    """Read full label volume, build instance channels, write label_F Zarr v2."""
    import os

    import numpy as np
    import zarr
    from biapy.data.data_manipulation import read_img_as_ndarray
    from biapy.data.pre_processing import labels_into_channels

    img = read_img_as_ndarray(filepath, is_3d=True)
    if img.ndim == 3:
        img = np.expand_dims(img, -1)

    class_channel = None
    if cfg.DATA.N_CLASSES > 2:
        if img.shape[-1] != 2:
            raise ValueError(
                "In instance segmentation, when 'DATA.N_CLASSES' are more than 2 labels need to have two channels, "
                "e.g. (256,256,2), containing the instance segmentation map (first channel) and classification map (second channel)."
            )
        class_channel = np.expand_dims(img[..., 1].copy(), -1)
    elif img.shape[-1] != 1:
        raise ValueError(
            "Expected instance segmentation GT images to have a single channel containing the instance labels, "
            "but got image with shape {} ({} channels). Check the image file: {}".format(
                img.shape, img.shape[-1], filepath
            )
        )

    img = labels_into_channels(
        img,
        mode=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS,
        channel_extra_opts=cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS_EXTRA_OPTS[0],
        save_dir=getattr(cfg.PATHS, tag + "_INSTANCE_CHANNELS_CHECK"),
    )

    if cfg.DATA.N_CLASSES > 2:
        img = np.concatenate([img, class_channel], axis=-1)

    out_dir = getattr(cfg.DATA, tag).INSTANCE_CHANNELS_MASK_DIR
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, os.path.basename(filepath))
    out = np.asarray(img, dtype=dtype_str)
    root = zarr.open(fname, mode="w", shape=out.shape, dtype=dtype_str, zarr_format=2)
    root[:] = out


def _patch_biapy_zarr_create_instance_channels():
    """Patch BiaPy 3.7.0 create_instance_channels for Zarr/H5 GT export.

    Applies two fixes in one exec() from pristine source:
    - Use float32 when continuous channels (Dc/Dn/...) are requested.
    - Full-volume read/compute/write for label_F Zarr v2 (replaces patch loop).
    """
    import inspect
    import re
    import sys

    import biapy.data.pre_processing as pp

    orig = pp.create_instance_channels
    src = inspect.getsource(orig)
    if "_BIAPY_FULLVOLUME_ZARR_GT" in src:
        return

    dtype_pattern = re.compile(
        r'if "D" in cfg\.PROBLEM\.INSTANCE_SEG\.DATA_CHANNELS:\s*\n'
        r'\s*dtype_str = "float32"\s*\n'
        r'\s*raise ValueError\("Currently distance creation using Zarr by chunks is not implemented\."\)\s*\n'
        r'\s*else:\s*\n'
        r'\s*dtype_str = "uint8"',
        re.M,
    )
    dtype_replacement = (
        "_FLOAT_INSTANCE_CHANNELS = frozenset({\n"
        '            "D", "Dc", "Dn", "Db", "R", "H", "V", "Gv", "Gh", "Gz", "Dv2",\n'
        "        })\n"
        "        if any(ch in cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS for ch in _FLOAT_INSTANCE_CHANNELS):\n"
        '            dtype_str = "float32"\n'
        "        else:\n"
        '            dtype_str = "uint8"'
    )
    patched_src, dtype_n = dtype_pattern.subn(dtype_replacement, src, count=1)
    if dtype_n != 1:
        raise RuntimeError(
            "Could not patch Zarr channel dtype block in create_instance_channels"
        )

    fullvolume_replacement = '''        else:  # regular instances, not synapses
            # _BIAPY_FULLVOLUME_ZARR_GT
            rank = get_rank()
            world_size = get_world_size()
            unique_files = []
            seen = set()
            for i in range(len(Y)):
                fp = Y[i]["filepath"]
                if fp not in seen:
                    seen.add(fp)
                    unique_files.append(fp)
            it = [fp for j, fp in enumerate(unique_files) if j % world_size == rank]
            compute_diam = any(ch in cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS for ch in ("Gv", "Gh", "Gz"))
            file_label_counts = {}
            for filepath in tqdm(it, disable=not is_main_process()):
                if compute_diam:
                    img_diam = read_img_as_ndarray(filepath, is_3d=True)
                    if img_diam.ndim == 3:
                        img_diam = np.expand_dims(img_diam, -1)
                    fbase = os.path.basename(filepath)
                    lbls, cnts = np.unique(img_diam[..., 0].astype(np.int64), return_counts=True)
                    lc = file_label_counts.setdefault(fbase, {})
                    for lb, cnt in zip(lbls.tolist(), cnts.tolist()):
                        if lb != 0:
                            lc[lb] = lc.get(lb, 0) + int(cnt)
                    del img_diam
                _biapy_write_fullvolume_instance_gt(filepath, cfg, tag, dtype_str)
            if compute_diam and file_label_counts:
                is_3d = cfg.PROBLEM.NDIM == "3D"
                diam_stats = {
                    fbase: {
                        "diameter": cellpose_diameter_from_areas(list(lc.values()), is_3d),
                        "n_objects": len(lc),
                    }
                    for fbase, lc in file_label_counts.items()
                }
                save_cellpose_diameter_stats(diam_stats, getattr(cfg.DATA, tag).INSTANCE_CHANNELS_MASK_DIR, data_type)
'''

    loop_pattern = re.compile(
        r"        else:  # regular instances, not synapses\n"
        r"            mask = None.*?                save_cellpose_diameter_stats\(diam_stats, getattr\(cfg\.DATA, tag\)\.INSTANCE_CHANNELS_MASK_DIR, data_type\)\n",
        re.DOTALL,
    )
    patched_src, loop_n = loop_pattern.subn(fullvolume_replacement, patched_src, count=1)
    if loop_n != 1:
        raise RuntimeError(
            "Could not replace patch loop with full-volume Zarr GT write in create_instance_channels"
        )

    exec_globals = orig.__globals__
    exec_globals["_biapy_write_fullvolume_instance_gt"] = _biapy_write_fullvolume_instance_gt
    exec(
        compile(
            patched_src,
            inspect.getfile(orig) + "<zarr_patch>",
            "exec",
        ),
        exec_globals,
    )
    patched = pp.create_instance_channels
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        if getattr(mod, "create_instance_channels", None) is orig:
            setattr(mod, "create_instance_channels", patched)
    print(
        "Patched BiaPy create_instance_channels: float32 channels + "
        "full-volume Zarr v2 GT export"
    )


_patch_biapy_zarr_create_instance_channels()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config-file',
        required=True,
        help=f'file name of your YAML configuration file in {CONFIG_DIR}'
    )
    parser.add_argument(
        '-r', '--result-dir',
        default=RESULT_DIR,
        help=f'dir to store the results in {RESULT_DIR}'
    )

    parser.add_argument(
        '-m',
        '--mode',
        default='train',
        help='Control config state, train or testing or preprocessing'
    )
    parser.add_argument(
        '--job-name',
        default='some-job',
        help='Name of the job'
    )
    parser.add_argument(
        '--run-id',
        default='0',
        help='Run ID for logging/versioning'
    )
    parser.add_argument(
        '--num-cpus',
        type=int,
        default=None,
        help='Override SYSTEM.NUM_CPUS/NUM_WORKERS '
             '(default: SLURM_CPUS_PER_TASK, else YAML)',
    )
    return parser.parse_args()


_TIFF_SUFFIXES = {'.tif', '.tiff'}


def _is_train_sample(path):
    """True for a TIFF file or a .zarr directory."""
    suffix = path.suffix.lower()
    if path.is_file() and suffix in _TIFF_SUFFIXES:
        return True
    return path.is_dir() and suffix == '.zarr'


def _gt_sample_exists(gt_path):
    suffix = gt_path.suffix.lower()
    if suffix in _TIFF_SUFFIXES:
        return gt_path.is_file()
    if suffix == '.zarr':
        return gt_path.is_dir()
    return False


def _copy_sample(src, dest):
    """Copy a TIFF file or a Zarr directory to dest (same basename)."""
    dest = dest / src.name
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)


def list_train_pairs(raw_dir, gt_dir):
    """Return (raw_path, gt_path) pairs for matching TIFF files or Zarr dirs."""
    raw_dir = Path(raw_dir)
    gt_dir = Path(gt_dir)
    raw_files = sorted(p for p in raw_dir.iterdir() if _is_train_sample(p))
    if not raw_files:
        raise FileNotFoundError(
            f'No TIFF files or Zarr directories found in train raw dir: {raw_dir}'
        )

    pairs = []
    missing = []
    for raw_path in raw_files:
        gt_path = gt_dir / raw_path.name
        if not _gt_sample_exists(gt_path):
            missing.append(raw_path.name)
            continue
        pairs.append((raw_path, gt_path))

    if missing:
        raise FileNotFoundError(
            f'Missing {len(missing)} label sample(s) under {gt_dir}; '
            f'first missing: {missing[0]}'
        )
    return pairs


def partition_pairs(pairs, size=TRAIN_PARTITION_SIZE):
    """Shuffle pairs and split into chunks of at most ``size``."""
    shuffled = list(pairs)
    random.shuffle(shuffled)
    return [shuffled[i:i + size] for i in range(0, len(shuffled), size)]


def stage_partition(pairs, dest_raw, dest_gt, workers):
    """Copy paired raw/label TIFFs or Zarr dirs into dest directories for BiaPy."""
    dest_raw = Path(dest_raw)
    dest_gt = Path(dest_gt)
    if dest_raw.exists():
        shutil.rmtree(dest_raw)
    if dest_gt.exists():
        shutil.rmtree(dest_gt)
    dest_raw.mkdir(parents=True, exist_ok=True)
    dest_gt.mkdir(parents=True, exist_ok=True)

    def _copy_one(pair):
        raw_path, gt_path = pair
        _copy_sample(raw_path, dest_raw)
        _copy_sample(gt_path, dest_gt)

    max_workers = max(1, min(int(workers), len(pairs)))
    print(
        f'Staging {len(pairs)} train pair(s) → {dest_raw.parent} '
        f'({max_workers} workers)'
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(tqdm(
            pool.map(_copy_one, pairs),
            total=len(pairs),
            desc='stage_partition',
            unit='pair',
        ))
    print(f'Finished staging {len(pairs)} pair(s) to {dest_raw.parent}')


def load_base_config(config_path):
    """Load YAML as a deep-copied dict (base values for mode overrides)."""
    with open(config_path) as f:
        return copy.deepcopy(yaml.safe_load(f))


def apply_mode_overrides(cfg, mode):
    """Overwrite ENABLE / LOAD_CHECKPOINT flags for the requested mode."""
    cfg.setdefault('TRAIN', {})
    cfg.setdefault('TEST', {})
    cfg.setdefault('MODEL', {})

    if mode == 'preprocessing':
        # Init-only: leave TRAIN/TEST/MODEL flags as in the YAML base.
        print(
            f'Mode overrides (preprocessing): leaving YAML flags as-is '
            f'(TRAIN.ENABLE={cfg["TRAIN"].get("ENABLE")}, '
            f'TEST.ENABLE={cfg["TEST"].get("ENABLE")}, '
            f'MODEL.LOAD_CHECKPOINT={cfg["MODEL"].get("LOAD_CHECKPOINT")})'
        )
        return cfg

    if mode == 'train':
        cfg['TRAIN']['ENABLE'] = True
        cfg['TEST']['ENABLE'] = False
        cfg['MODEL']['LOAD_CHECKPOINT'] = False
    elif mode == 'test':
        cfg['TRAIN']['ENABLE'] = False
        cfg['TEST']['ENABLE'] = True
        cfg['MODEL']['LOAD_CHECKPOINT'] = True
    else:
        raise ValueError(f'Unknown mode for overrides: {mode!r}')

    print(
        f'Mode overrides ({mode}): '
        f'TRAIN.ENABLE={cfg["TRAIN"]["ENABLE"]}, '
        f'TEST.ENABLE={cfg["TEST"]["ENABLE"]}, '
        f'MODEL.LOAD_CHECKPOINT={cfg["MODEL"]["LOAD_CHECKPOINT"]}'
    )
    return cfg


def apply_cpu_overrides(cfg, n_cpus=None):
    """Set SYSTEM.NUM_CPUS/NUM_WORKERS from CLI or SLURM_CPUS_PER_TASK."""
    cfg.setdefault('SYSTEM', {})
    if n_cpus is None:
        env = os.environ.get('SLURM_CPUS_PER_TASK')
        if env:
            n_cpus = int(env)
    if n_cpus is None:
        print(
            'CPU overrides: none '
            f'(SYSTEM.NUM_CPUS={cfg["SYSTEM"].get("NUM_CPUS")}, '
            f'SYSTEM.NUM_WORKERS={cfg["SYSTEM"].get("NUM_WORKERS")})'
        )
        return cfg

    cfg['SYSTEM']['NUM_CPUS'] = n_cpus
    cfg['SYSTEM']['NUM_WORKERS'] = -1
    print(
        f'CPU overrides: SYSTEM.NUM_CPUS={n_cpus}, '
        f'SYSTEM.NUM_WORKERS={n_cpus}'
    )
    return cfg


def main():
    args = get_args()

    # BiaPy 3.7.0 only accepts str/dict/CfgNode, not Path.
    config_path = (CONFIG_DIR / args.config_file).as_posix()
    cfg = apply_mode_overrides(load_base_config(config_path), args.mode)
    cfg = apply_cpu_overrides(cfg, args.num_cpus)

    match args.mode:
        case 'preprocessing':
            biapy = BiaPy(
                config=cfg,
                result_dir=RESULT_DIR.as_posix(),
                name=args.job_name,
                run_id=args.run_id,
                verbose=True
            )

        case 'train':
            # YAML GT_PATH is the raw instance-ID dir; BiaPy may rewrite it to a
            # multi-channel label_F... dir during prepare_instance_data.
            yaml_train_gt = cfg['DATA']['TRAIN']['GT_PATH']

            biapy = BiaPy(
                config=cfg,
                result_dir=RESULT_DIR.as_posix(),
                name=args.job_name,
                run_id=args.run_id,
                gpu='0',
                verbose=True
            )

            train_gt = biapy.cfg.DATA.TRAIN.GT_PATH
            data_channels = list(
                biapy.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS or []
            )
            if (
                len(data_channels) > 1
                and Path(train_gt).resolve() == Path(yaml_train_gt).resolve()
            ):
                raise RuntimeError(
                    f'Expected BiaPy to rewrite DATA.TRAIN.GT_PATH to a '
                    f'multi-channel label_F... dir for channels '
                    f'{data_channels}, but it is still {train_gt!r}. '
                    f'Run preprocessing first so the derived GT exists.'
                )

            print(
                f'Training on PATH={biapy.cfg.DATA.TRAIN.PATH} '
                f'GT_PATH={train_gt}'
            )
            biapy.train()

        # Partitioned staging train. Restore by uncommenting this case and
        # commenting the simple 'train' case above.
        # case 'train':
        #     # YAML GT_PATH is the raw instance-ID dir; BiaPy may rewrite it to a
        #     # multi-channel label_F... dir during prepare_instance_data.
        #     yaml_train_gt = cfg['DATA']['TRAIN']['GT_PATH']
        #
        #     biapy = BiaPy(
        #         config=cfg,
        #         result_dir=RESULT_DIR.as_posix(),
        #         name=args.job_name,
        #         run_id=args.run_id,
        #         gpu='0',
        #         verbose=True
        #     )
        #
        #     # Stage from post-init paths so multi-channel masks are used.
        #     train_raw = biapy.cfg.DATA.TRAIN.PATH
        #     train_gt = biapy.cfg.DATA.TRAIN.GT_PATH
        #     data_channels = list(
        #         biapy.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS or []
        #     )
        #     if (
        #         len(data_channels) > 1
        #         and Path(train_gt).resolve() == Path(yaml_train_gt).resolve()
        #     ):
        #         raise RuntimeError(
        #             f'Expected BiaPy to rewrite DATA.TRAIN.GT_PATH to a '
        #             f'multi-channel label_F... dir for channels '
        #             f'{data_channels}, but it is still {train_gt!r}. '
        #             f'Run preprocessing first so the derived GT exists.'
        #         )
        #
        #     print(
        #         f'Staging train pairs from PATH={train_raw} '
        #         f'GT_PATH={train_gt}'
        #     )
        #
        #     # YAML base name (e.g. label) vs BiaPy prepared multi-channel
        #     # basename (e.g. label_F.erosion-0...). Stage into the prepared
        #     # name and symlink the base name to it so both
        #     # INSTANCE_CHANNELS_MASK_DIR == GT_PATH and
        #     # INSTANCE_CHANNELS_MASK_DIR == GT_PATH+_F... checks succeed.
        #     gt_base_name = Path(yaml_train_gt).name
        #     gt_prepared_name = Path(train_gt).name
        #
        #     pairs = list_train_pairs(train_raw, train_gt)
        #     partitions = partition_pairs(pairs, size=TRAIN_PARTITION_SIZE)
        #     print(
        #         f'Train partitions: {len(partitions)} chunk(s) from '
        #         f'{len(pairs)} sample(s), max size {TRAIN_PARTITION_SIZE}'
        #     )
        #
        #     tmp_root = (
        #         Path(os.environ.get('SLURM_TMPDIR', tempfile.gettempdir()))
        #         / f'biapy_train_parts_{args.job_name}_{args.run_id}'
        #     )
        #     try:
        #         for i, part in enumerate(partitions):
        #             part_dir = tmp_root / f'part_{i}'
        #             part_raw = part_dir / 'raw'
        #             part_gt_prepared = part_dir / gt_prepared_name
        #             part_gt_base = part_dir / gt_base_name
        #
        #             stage_partition(
        #                 part, part_raw, part_gt_prepared,
        #                 workers=biapy.cfg.SYSTEM.NUM_CPUS,
        #             )
        #
        #             if gt_base_name != gt_prepared_name:
        #                 if part_gt_base.exists() or part_gt_base.is_symlink():
        #                     if part_gt_base.is_dir() and not part_gt_base.is_symlink():
        #                         shutil.rmtree(part_gt_base)
        #                     else:
        #                         part_gt_base.unlink()
        #                 part_gt_base.symlink_to(
        #                     part_gt_prepared.resolve(),
        #                     target_is_directory=True,
        #                 )
        #                 part_gt = part_gt_base
        #             else:
        #                 part_gt = part_gt_prepared
        #
        #             load_ckpt = i > 0
        #             print(
        #                 f'Partition {i + 1}/{len(partitions)}: '
        #                 f'{len(part)} sample(s), LOAD_CHECKPOINT={load_ckpt}'
        #             )
        #
        #             updates = {
        #                 'TRAIN.ENABLE': True,
        #                 'TEST.ENABLE': False,
        #                 'DATA.TRAIN.PATH': part_raw.as_posix(),
        #                 'DATA.TRAIN.GT_PATH': part_gt.as_posix(),
        #                 'MODEL.LOAD_CHECKPOINT': load_ckpt,
        #             }
        #             # if biapy.cfg.TRAIN.BATCH_SIZE > len(part):
        #             #     updates['TRAIN.BATCH_SIZE'] = len(part)
        #
        #             biapy.update_config(updates)
        #             biapy.train()
        #     finally:
        #         if tmp_root.exists():
        #             shutil.rmtree(tmp_root, ignore_errors=True)

        case 'test':
            biapy = BiaPy(
                config=cfg,
                result_dir=RESULT_DIR.as_posix(),
                name=args.job_name,
                run_id=args.run_id,
                gpu='0',
                verbose=True
            )
            biapy.test()

        case _:
            raise ValueError(
                f'Unknown mode {args.mode!r}; '
                f'expected preprocessing, train, or test'
            )


if __name__ == "__main__":
    main()
