import os
import copy
import argparse
from pathlib import Path

import yaml
from biapy import BiaPy

CONFIG_DIR = Path("biapy_work_folder/configs")
RESULT_DIR = Path('metrics/biapy')
TRAIN_PARTITION_SIZE = 18


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


def apply_stage_path_overrides(cfg):
    """Point TRAIN/VAL paths at $BIAPY_STAGE_ROOT when sbatch staged data there.

    GT_PATH keeps basename ``label`` so BiaPy still rewrites it to the staged
    ``label_F…`` sibling directory.
    """
    root = os.environ.get('BIAPY_STAGE_ROOT')
    if not root:
        return cfg

    root = Path(root)
    cfg.setdefault('DATA', {})
    cfg['DATA'].setdefault('TRAIN', {})
    cfg['DATA'].setdefault('VAL', {})
    cfg['DATA']['TRAIN']['PATH'] = str(root / 'train' / 'raw')
    cfg['DATA']['TRAIN']['GT_PATH'] = str(root / 'train' / 'label')
    cfg['DATA']['VAL']['PATH'] = str(root / 'val' / 'raw')
    cfg['DATA']['VAL']['GT_PATH'] = str(root / 'val' / 'label')
    print(f'Path overrides (staged): TRAIN/VAL under {root}')
    return cfg


def main():
    args = get_args()

    # BiaPy 3.7.0 only accepts str/dict/CfgNode, not Path.
    config_path = (CONFIG_DIR / args.config_file).as_posix()
    cfg = apply_mode_overrides(load_base_config(config_path), args.mode)
    cfg = apply_cpu_overrides(cfg, args.num_cpus)
    cfg = apply_stage_path_overrides(cfg)

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
            # Compare basenames (not Path.resolve): staging may symlink
            # label -> label_F…, which would make resolve() a false positive.
            yaml_train_gt = str(cfg['DATA']['TRAIN']['GT_PATH'])

            biapy = BiaPy(
                config=cfg,
                result_dir=RESULT_DIR.as_posix(),
                name=args.job_name,
                run_id=args.run_id,
                gpu='0',
                verbose=True
            )

            train_gt = str(biapy.cfg.DATA.TRAIN.GT_PATH)
            data_channels = list(
                biapy.cfg.PROBLEM.INSTANCE_SEG.DATA_CHANNELS or []
            )
            if (
                len(data_channels) > 1
                and Path(train_gt).name == Path(yaml_train_gt).name
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
