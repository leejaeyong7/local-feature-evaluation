# hack to add parent path
import sys
sys.path.append('..')

import logging
import argparse
import os
import shutil

from evaluater.matching_pipeline import matching_pipeline
from evaluater.reconstruction_pipeline import reconstruction_pipeline

logging.basicConfig(level='INFO',
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

DATASET_NAMES = [
    'Fountain', 'Herzjesu', 'South-Building',
    'Madrid_Metropolis', 'Gendarmenmarkt', 'Tower_of_London',
    'Oxford5k', 'Alamo', 'Roman_Forum', 'ArtsQuad_dataset'
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # logging related arguments

    # path to config file
    # config file name (without .yaml / .yml) is treated as experiment name
    parser.add_argument('--extractor-type', type=str, required=True, 
                        choices=['superpoint'],
                        help='extractor type')
    
    # path to dataset directory for tensorboard
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='folder for dataset without modification')

    parser.add_argument('--output-dir', type=str, required=True, 
                        help='folder used for cloning dataset-dir\n \
                              the datawill be copied over \
                              output-dir/extractor-type')

    parser.add_argument('---results-dir', type=str, required=True, 
                        help='output directory for the results')

    parser.add_argument('--colmap-dir', type=str, required=False, 
                        default='~/colmap/build', 
                        help='output for colmap build binary')

    args = parser.parse_args()
    logging.debug('Arguments receieved: {}'.format(str(vars(args))))

    # we want to copy dataset_dir into output_dir/extractor_type
    dataset_target_dir = os.path.join(args.output_dir, args.extractor_type)
    if(not os.path.exists(dataset_target_dir)):
        shutil.copytree(args.dataset_dir, dataset_target_dir)

    # perform matching
    matching_pipeline(args.extractor_type, 
                      dataset_target_dir, 
                      DATASET_NAMES, 
                      args.colmap_dir)

    # perform recon
    for dataset_name in DATASET_NAMES:
        dataset_full_path = os.path.join(dataset_target_dir, dataset_name)
        reconstruction_pipeline(args.extractor_type, 
                                dataset_full_path, 
                                args.colmap_dir, 
                                args.results_dir)
