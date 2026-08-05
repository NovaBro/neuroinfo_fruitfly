import os
import random
import shutil
import tempfile
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from tqdm import tqdm
from biapy import BiaPy

CONFIG_DIR = Path("BiaPy/configs")
RESULT_DIR = Path('metrics/biapy')
TRAIN_PARTITION_SIZE = 18

# Change to load weight safety!
import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load


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
    return parser.parse_args()


def list_train_pairs(raw_dir, gt_dir):
    """Return (raw_path, gt_path) pairs for matching TIFF filenames."""
    raw_dir = Path(raw_dir)
    gt_dir = Path(gt_dir)
    raw_files = sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {'.tif', '.tiff'}
    )
    if not raw_files:
        raise FileNotFoundError(f'No TIFF files found in train raw dir: {raw_dir}')

    pairs = []
    missing = []
    for raw_path in raw_files:
        gt_path = gt_dir / raw_path.name
        if not gt_path.is_file():
            missing.append(raw_path.name)
            continue
        pairs.append((raw_path, gt_path))

    if missing:
        raise FileNotFoundError(
            f'Missing {len(missing)} label TIFF(s) under {gt_dir}; '
            f'first missing: {missing[0]}'
        )
    return pairs


def partition_pairs(pairs, size=TRAIN_PARTITION_SIZE):
    """Shuffle pairs and split into chunks of at most ``size``."""
    shuffled = list(pairs)
    random.shuffle(shuffled)
    return [shuffled[i:i + size] for i in range(0, len(shuffled), size)]


def stage_partition(pairs, dest_raw, dest_gt, workers):
    """Copy paired raw/label TIFFs into dest directories for BiaPy."""
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
        shutil.copy2(raw_path, dest_raw / raw_path.name)
        shutil.copy2(gt_path, dest_gt / gt_path.name)

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


def main():
    args = get_args()

    # BiaPy 3.7.0 only accepts str/dict/CfgNode, not Path.
    config_path = (CONFIG_DIR / args.config_file).as_posix()

    match args.mode:
        case 'preprocessing':
            biapy = BiaPy(
                config=config_path, 
                result_dir=RESULT_DIR.as_posix(), 
                name=args.job_name, 
                run_id=args.run_id, 
                verbose=True
            )

        case 'train':
            # Snapshot YAML paths before BiaPy() — prepare_instance_data may
            # rewrite DATA.TRAIN.GT_PATH to a multi-channel label_F... dir.
            with open(config_path) as f:
                yaml_cfg = yaml.safe_load(f)
            train_raw = yaml_cfg['DATA']['TRAIN']['PATH']
            train_gt = yaml_cfg['DATA']['TRAIN']['GT_PATH']

            biapy = BiaPy(
                config=config_path, 
                result_dir=RESULT_DIR.as_posix(), 
                name=args.job_name, 
                run_id=args.run_id, 
                gpu='0', 
                verbose=True
            )

            print(
                f'Staging train pairs from YAML PATH={train_raw} '
                f'GT_PATH={train_gt}'
            )
            print(
                f'(after init) cfg DATA.TRAIN.GT_PATH='
                f'{biapy.cfg.DATA.TRAIN.GT_PATH}'
            )

            pairs = list_train_pairs(train_raw, train_gt)
            partitions = partition_pairs(pairs, size=TRAIN_PARTITION_SIZE)
            print(
                f'Train partitions: {len(partitions)} chunk(s) from '
                f'{len(pairs)} sample(s), max size {TRAIN_PARTITION_SIZE}'
            )

            tmp_root = (
                Path(os.environ.get('SLURM_TMPDIR', tempfile.gettempdir()))
                / f'biapy_train_parts_{args.job_name}_{args.run_id}'
            )
            try:
                for i, part in enumerate(partitions):
                    part_raw = tmp_root / f'part_{i}' / 'raw'
                    part_gt = tmp_root / f'part_{i}' / 'label'
                    stage_partition(
                        part, part_raw, part_gt,
                        workers=biapy.cfg.SYSTEM.NUM_CPUS,
                    )

                    load_ckpt = i > 0
                    print(
                        f'Partition {i + 1}/{len(partitions)}: '
                        f'{len(part)} sample(s), LOAD_CHECKPOINT={load_ckpt}'
                    )

                    updates = {
                        'TRAIN.ENABLE': True,
                        'TEST.ENABLE': False,
                        'DATA.TRAIN.PATH': part_raw.as_posix(),
                        'DATA.TRAIN.GT_PATH': part_gt.as_posix(),
                        'MODEL.LOAD_CHECKPOINT': load_ckpt,
                    }
                    # if biapy.cfg.TRAIN.BATCH_SIZE > len(part):
                    #     updates['TRAIN.BATCH_SIZE'] = len(part)

                    biapy.update_config(updates)
                    biapy.train()
            finally:
                if tmp_root.exists():
                    shutil.rmtree(tmp_root, ignore_errors=True)

        case 'test':
            biapy = BiaPy(
                config=config_path, 
                result_dir=RESULT_DIR.as_posix(), 
                name=args.job_name, 
                run_id=args.run_id, 
                gpu='0', 
                verbose=True
            )
            biapy.update_config(
                {
                    'TRAIN.ENABLE':False,
                    'TEST.ENABLE':True,
                    'MODEL.LOAD_CHECKPOINT':True
                }
            )
            biapy.test()


if __name__ == "__main__":
    main()
