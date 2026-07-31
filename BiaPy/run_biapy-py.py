import logging
import argparse
from pathlib import Path
from biapy import BiaPy
# NOTE: build_config Does Not Exist !?
# from biapy import build_config
from biapy.config.config import update_dependencies
from biapy.data.pre_processing import create_instance_channels

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

CONFIG_DIR = Path("BiaPy/configs")
RESULT_DIR = Path('metrics/biapy')

# Change to load weight safety!
import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load


# config_path = "/path/to/config.yaml"   # Path to your YAML configuration file
# result_dir = "/path/to/results"        # Directory to store the results
# job_name = "my_biapy_job"              # Name of the job
# run_id = 1                             # Run ID for logging/versioning
# gpu = "0"                              # GPU to use (as string, e.g., "0")

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

    # logging.basicConfig(
    #     filename=f"logging_{args.config_file}.txt",
    #     level=logging.WARNING,
    # )
    if args.mode in ['test', 'train']:
        biapy = BiaPy(
            config=CONFIG_DIR / args.config_file, 
            result_dir=RESULT_DIR.as_posix(), 
            name=args.job_name, 
            run_id=args.run_id, 
            gpu='0'
        )
        biapy.cfg
        biapy.run_job()
    elif args.mode in ['preprocessing']:
        biapy = BiaPy(
            config=CONFIG_DIR / args.config_file, 
            result_dir=RESULT_DIR.as_posix(), 
            name=args.job_name, 
            run_id=args.run_id, 
        )
        create_instance_channels(biapy.cfg)

    # if args.mode == 'train':
    #     biapy.cfg['TRAIN']['ENABLE'] = True
    #     biapy.cfg['MODEL']['LOAD_CHECKPOINT'] = False
    #     biapy.cfg['TEST']['ENABLE'] = False

    # elif args.mode == 'test':
    #     biapy.cfg['TRAIN']['ENABLE'] = False
    #     biapy.cfg['MODEL']['LOAD_CHECKPOINT'] = True
    #     biapy.cfg['TEST']['ENABLE'] = True

    # else:
    #     # logger.error("argument mode is neither 'train' nor 'test' !")
    #     raise ValueError("argument mode is neither 'train' nor 'test' !")

    # NOTE: update_dependencies like this seems not intended. 
    # However, at current biapy version, we need to update config. setting cfg['option'] here is too late.
    # TODO: Update implementation here for proper api implementation (build_config()).
    # update_dependencies(biapy.cfg)

    # biapy.print_config()       # the full resolved configuration
    # biapy.print_train_info()   # training overview: model, patch, epochs, LR, optimizer, augmentations, files
    # biapy.print_test_info()

if __name__ == "__main__":
    main()