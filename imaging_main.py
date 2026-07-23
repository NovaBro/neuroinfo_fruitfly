import os
import sys
import argparse
import logging
import tifffile

import numpy as np
from pathlib import Path

# from imaging_helpers_hpc import globals as ih_globals
from imaging_helpers_hpc.paths import BiapyDataPaths, FisbeDataPaths, AnalysisOutputPaths
from imaging_helpers_hpc.imaging import gen_biapy_mip_4panel, gen_basic_mip, gen_instance_projection, gen_rotations_and_projections
from imaging_helpers_hpc.loading import get_sample_stem, load_biapy_test_sample, load_fisbe_completely, load_any_tif
from imaging_helpers_hpc.analysis import get_stats_in_dir, get_stats_in_one_image


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.WARNING, 
        format='%(asctime)s - %(levelname)-8s: %(message)s',
        handlers= [
            logging.FileHandler('imaging_helpers_hpc.txt', 'w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command", required=True, help='Specify the source of data of interest')

    parser_any = subparser.add_parser('any')
    parser_any.add_argument("-i", "--input-file", help="Specify the split and sample [train / test / val]/[sample]. Do not include extension")
    # parser_any.add_argument("-m", "--mode", required=True, help="Mode for what to do with BiaPy data")

    parser_biapy = subparser.add_parser('biapy')
    parser_biapy.add_argument("-m", "--mode", required=True, help="Mode for what to do with BiaPy data")
    parser_biapy.add_argument("-c", "--bia-config-name", default="train_Dn_0707", help="Select config file for source of data within biapy")

    parser_biapy.add_argument("-s", "--sample", default="JRC_SS04989-20160318_24_A2.zarr.tiff", help="What sample to view")
    parser_biapy.add_argument("-w", "--watershed", default="seed_map", help="What part of each watershed sample")

    parser_fisbe = subparser.add_parser('fisbe')
    parser_fisbe.add_argument("-m", "--mode", required=True, help="Mode for what to do with Fisbe data. Options: ['mip' | 'rotate' | 'channel']")
    parser_fisbe.add_argument("-i", "--input-file", help="Specify the split and sample [train / test / val]/[sample]. Do not include extension")
    parser_fisbe.add_argument("-v", "--volume", default="raw", help="Specify what part of the image volume, e.g raw or gt_instance")

    parser_stats = subparser.add_parser('stats')
    parser_stats.add_argument("-m", "--mode", help="Mode for general stats")
    parser_stats.add_argument("-i", "--input")

    args = parser.parse_args()

    analysis_output_paths = AnalysisOutputPaths("imaging_helpers_hpc/output")

    match args.command:
        case 'any':
            tif_image = load_any_tif(args.input_file)
            gen_basic_mip(tif_image, f'any_sample_{Path(args.input_file).stem}', analysis_output_paths, axis=0)

        case 'biapy':
            biapy_paths = BiapyDataPaths(args.bia_config_name)

            match args.mode:
                case '4-pane-mip':
                    sample_names = sorted(get_sample_stem(p) for p in biapy_paths.per_image.glob("*.tif"))
                    logger.info(f"{len(sample_names)} test volumes of BiaPy outputs: {sample_names}")
                    sample = sample_names[2]

                    raw_vol, prob_vol, inst_vol = load_biapy_test_sample(sample, biapy_paths)
                    gen_biapy_mip_4panel(raw_vol, prob_vol, inst_vol, analysis_output_paths, title_prefix=f"{sample}")
                case 'watershed':
                    input_file_path = biapy_paths.watershed / args.sample / f"{args.watershed}.tif"
                    output_file_name=f'{args.watershed}_{args.sample}'
                    watershed_files = [input_file_path]

                    get_stats_in_one_image(input_file_path)
                    watershed_image = tifffile.imread(input_file_path)[np.newaxis, ...]
                    logger.info(f"watershed_image shape: {watershed_image.shape}")

                    if args.watershed == 'topografic_surface':
                        gen_basic_mip(
                            watershed_image,
                            output_file_name,
                            analysis_output_paths
                        )
                    else:
                        gen_instance_projection(
                            watershed_image,
                            output_file_name,
                            analysis_output_paths,
                        )

        case 'fisbe':
            fisb_paths = FisbeDataPaths()
            assert args.input_file, "define --input-file for this mode"
            try:
                split, sample = str.split(args.input_file, '/')
            except ValueError:
                print("Check if you are using the split/sample formating")

            raw_np, gt_instance_np = load_fisbe_completely(sample, fisb_paths, split)

            match args.mode:
                case 'mip-gt':
                    gen_basic_mip(raw_np, f"raw_{split}_{sample}", analysis_output_paths)
                    gen_instance_projection(gt_instance_np, f"gt_{split}_{sample}", analysis_output_paths)

                case 'rotate':
                    rand_axis_int = np.random.randint(0, 3)
                    k_rotations = np.random.randint(1, 4)
                    logger.info(
                        f"shared rotation: rand_axis_int={rand_axis_int}, k_rotations={k_rotations}"
                    )
                    logger.info('rotating raw')
                    gen_rotations_and_projections(
                        raw_np, f"raw_{split}_{sample}", analysis_output_paths,
                        volume='raw', rand_axis_int=rand_axis_int, k_rotations=k_rotations,
                    )
                    logger.info('rotating gt')
                    gen_rotations_and_projections(
                        gt_instance_np, f"gt_{split}_{sample}", analysis_output_paths,
                        volume='gt_instance', rand_axis_int=rand_axis_int, k_rotations=k_rotations,
                    )

                case 'channel':
                    random_order = np.random.permutation(3)
                    shuffled_image = raw_np[random_order, ...]
                    gen_basic_mip(shuffled_image, f"raw_shuffled_channel_{split}_{sample}", analysis_output_paths)

        case 'stats':
            input_path = Path(args.input)
            if input_path.is_file():
                print(f"Getting Stats at file {input_path}")
                get_stats_in_one_image(input_path)
            else:
                print(f"Getting Stats at dir {input_path}")
                get_stats_in_dir(input_path)

            # if os.isdir(Path(args.input)):
            #     print(f"Getting Stats at dir {args.input}")
            #     get_stats_in_dir(Path(args.input))
            # else:
            #     print(f"Getting Stats at file {args.input}")
            #     get_stats_in_one_image(args.input)

            # match args.mode:
            #     case 'dir':
            #         print(f"Getting Stats at {args.input}")
            #         get_stats_in_dir(Path(args.input))
            #     case 'file':
            #         get_stats_in_one_image(args.input)

        case _:
            logger.warning("Need to choose an action")
            print("Need to choose an action.") 

