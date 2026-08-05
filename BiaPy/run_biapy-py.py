import shutil
import logging
import argparse
from pathlib import Path
from biapy import BiaPy

CONFIG_DIR = Path("BiaPy/configs")
RESULT_DIR = Path('metrics/biapy')

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
                    'TRAIN.ENABLE':True,
                    'TEST.ENABLE':False,
                    'MODEL.LOAD_CHECKPOINT':False
                }
            )
            biapy.train()

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