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


def _rebind_biapy_symbol(name, new_fn, orig_fn=None):
    """Rebind ``name`` on biapy modules that still point at ``orig_fn`` (or any prior value)."""
    import sys

    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        mod_name = getattr(mod, "__name__", "") or ""
        if not mod_name.startswith("biapy"):
            continue
        if not hasattr(mod, name):
            continue
        cur = getattr(mod, name)
        if orig_fn is not None and cur is not orig_fn and cur is not new_fn:
            continue
        setattr(mod, name, new_fn)


def _patch_biapy_fast_rot90():
    """Use np.rot90 for unit-scale 90°-multiple affines instead of SciPy resample.

    BiaPy folds ROT90 into affine_transform even when scale is 1.0; that full float
    resample dominates online DA CPU cost. Preserve semantics for zoom / RANDOM_ROT
    (non-unit scale or non-90° angles still use stock affine_transform).
    """
    import numpy as np
    import biapy.data.generators.augmentors as aug

    orig = aug.affine_transform
    if getattr(orig, "_biapy_fast_rot90", False):
        return

    def affine_transform(
        img,
        mask=None,
        heat=None,
        scale_xy=1.0,
        scale_z=1.0,
        angle=0.0,
        mode="reflect",
        mask_type="mask",
        flow_heat=None,
        **kwargs,
    ):
        # Compatible with biapy 3.7.0 (no img_type) and newer forks (optional img_type kw).
        if abs(float(scale_xy) - 1.0) <= 1e-3 and abs(float(scale_z) - 1.0) <= 1e-3:
            ang = float(angle) % 360.0
            # Identity already handled in stock path; also catch exact 0 here.
            if ang == 0.0:
                return img if mask is None else (img, mask, heat)
            # Exact 90° multiples: np.rot90 matches in-plane CCW rotation without resample.
            nearest_k = int(round(ang / 90.0))
            if abs(ang - 90.0 * nearest_k) <= 1e-6:
                k = nearest_k % 4
                if k != 0:
                    assert img.ndim in (3, 4), f"Image must be 3D or 4D, got shape {img.shape}"
                    axes = (1, 2) if img.ndim == 4 else (0, 1)
                    img = np.rot90(img, k=k, axes=axes)
                    if mask is not None:
                        mask = np.rot90(mask, k=k, axes=axes)
                    if heat is not None:
                        heat = np.rot90(heat, k=k, axes=axes)
                        heat = aug.rotate_flow_vectors(heat, flow_heat, angle)
                    return img if mask is None else (img, mask, heat)

        return orig(
            img,
            mask=mask,
            heat=heat,
            scale_xy=scale_xy,
            scale_z=scale_z,
            angle=angle,
            mode=mode,
            mask_type=mask_type,
            flow_heat=flow_heat,
            **kwargs,
        )

    affine_transform._biapy_fast_rot90 = True
    affine_transform.__wrapped__ = orig
    aug.affine_transform = affine_transform
    _rebind_biapy_symbol("affine_transform", affine_transform, orig_fn=orig)
    print("Patched BiaPy apply_transform: fast np.rot90 for ROT90-only")


def _patch_biapy_inplace_intensity_augs():
    """Avoid redundant brightness/contrast .copy() on float arrays.

    load_sample already copies each patch, so mutating floats in place is safe and
    skips an extra ~0.5GB allocation per sample when those augs fire.
    """
    import numpy as np
    import biapy.data.generators.augmentors as aug

    if getattr(aug.brightness, "_biapy_inplace_intensity", False):
        return

    orig_brightness = aug.brightness
    orig_contrast = aug.contrast

    def brightness(image, brightness_factor=(0, 0)):
        assert image.ndim in (3, 4), f"Image must be 3D or 4D, got {image.shape}"

        lo, hi = float(brightness_factor[0]), float(brightness_factor[1])
        if lo == 0.0 and hi == 0.0:
            return image
        if lo > hi:
            lo, hi = hi, lo
        if image.size == 0:
            return image

        delta = float(np.random.uniform(lo, hi))
        if np.issubdtype(image.dtype, np.floating):
            image += delta
            return image

        info = np.iinfo(image.dtype)
        tmp = image.astype(np.float32, copy=False) + delta
        np.clip(tmp, info.min, info.max, out=tmp)
        return tmp.astype(image.dtype, copy=False)

    def contrast(image, contrast_factor=(0, 0)):
        assert image.ndim in (3, 4), f"Image must be 3D or 4D, got {image.shape}"

        lo, hi = float(contrast_factor[0]), float(contrast_factor[1])
        if lo == 0.0 and hi == 0.0:
            return image
        if lo > hi:
            lo, hi = hi, lo
        if image.size == 0:
            return image

        scale = 1.0 + float(np.random.uniform(lo, hi))
        if np.issubdtype(image.dtype, np.floating):
            image *= scale
            return image

        info = np.iinfo(image.dtype)
        tmp = image.astype(np.float32, copy=False) * scale
        np.clip(tmp, info.min, info.max, out=tmp)
        return tmp.astype(image.dtype, copy=False)

    brightness._biapy_inplace_intensity = True
    contrast._biapy_inplace_intensity = True
    brightness.__wrapped__ = orig_brightness
    contrast.__wrapped__ = orig_contrast

    aug.brightness = brightness
    aug.contrast = contrast
    _rebind_biapy_symbol("brightness", brightness, orig_fn=orig_brightness)
    _rebind_biapy_symbol("contrast", contrast, orig_fn=orig_contrast)
    print("Patched BiaPy brightness/contrast: in-place for float arrays")


_patch_biapy_fast_rot90()
_patch_biapy_inplace_intensity_augs()


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
    cfg['SYSTEM']['NUM_WORKERS'] = 14
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
