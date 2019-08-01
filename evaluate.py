import logging
import argparse

from evaluater.matching_pipeline import matching_pipeline

logging.basicConfig(level='INFO',
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

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
                        help='folder for dataset')

    # path to logging directory for tensorboard
    parser.add_argument('--colmap-dir', type=str, required=False, 
                        default='~/colmap/build', 
                        help='output for colmap build binary')
    args = parser.parse_args()
    logging.debug('Arguments receieved: {}'.format(str(vars(args))))

    matching_pipeline(**vars(args))