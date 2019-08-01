# -*- coding: utf-8 -*-
""" Entry for matching pipeline. """

# module imports
import os
import logging
from os import path
from .exhaustive_matching import exhaustive_matching
from .approximate_matching import approximate_matching
# from .extracors.powerpoint import PowerPoint
from .extract_features import extract_features
# from .extracors.lf_net import LFNet
# from .extracors.d2_net import D2Net


# global variables
DATASET_NAMES = [
    'Fountain', 'Herzjesu', 'South-Building',
    'Madrid_Metropolis', 'Gendarmenmarkt', 'Tower_of_London',
    'Oxford5k', 'Alamo', 'Roman_Forum', 'ArtsQuad_dataset'
]
DATASET_NAMES = [
    'Fountain'
]

def mkdirp(p):
    if(not path.exists(p)):
        os.makedirs(p)

def matching_pipeline(extractor_type, dataset_dir, colmap_dir):
    logging.info('= Performing matching evaluation pipeline for {}'.format(extractor_type))
    for dataset_name in DATASET_NAMES:
        # Set the pipeline parameters.
        #       contain an "images" folder and a "database.db" file.
        logging.info('= Processing dataset {}'.format(dataset_name))
        dataset_path = path.join(dataset_dir, dataset_name)

        COLMAP_PATH = colmap_dir

        # Radius of local patches around each keypoint.
        patch_radius = 32

        # Whether to run matching on GPU.
        match_gpu = False

        # Number of images to match in one block.
        match_block_size = 50

        # maximum distance ratio between first and second best matches.
        match_max_dist_ratio = 0.8

        # mnimum number of matches between two images.
        min_num_matches = 15

        #  Setup the pipeline environment.
        logging.info('= Setting up paths & Reading files...')
        image_path = path.join(dataset_path, 'images')
        keypoint_path = path.join(dataset_path, 'keypoints')
        descriptor_path = path.join(dataset_path, 'descriptors')
        match_path = path.join(dataset_path, 'matches')
        database_path = path.join(dataset_path, 'database.db')

        #  Create the output directories.
        mkdirp(keypoint_path)
        mkdirp(descriptor_path)
        mkdirp(match_path)

        #  Extract the image names and paths.
        image_names = [f for f in sorted(os.listdir(image_path)) if (f != '.' and f != '..')]

        num_images = len(image_names)
        image_paths = []
        keypoint_paths = []
        descriptor_paths = []

        for image_name in image_names:
            image_paths.append(path.join(image_path, image_name))
            keypoint_paths.append(path.join(keypoint_path, image_name + '.bin'))
            descriptor_paths.append(path.join(descriptor_path, image_name + '.bin'))

        #  Compute the keypoints and descriptors.
        logging.info('= Extracting Features ... ')
        extract_features(extractor_type, image_paths, keypoint_paths, descriptor_paths)

        dataset_paths = {
            'image': image_path,
            'keypoint': keypoint_path,
            'descriptor': descriptor_path,
            'match': match_path,
            'database': database_path,
            'dataset': dataset_path,
        }
        feature_paths = {
            'names': image_names,
            'images': image_paths,
            'keypoints': keypoint_paths,
            'descriptors': descriptor_paths,
        }

        # Match the descriptors.
        #
        # NOTE: - You must exhaustively match Fountain, Herzjesu, South Building,
        #         Madrid Metropolis, Gendarmenmarkt, and Tower of London.
        #       - You must approximately match Alamo, Roman Forum, Cornell.
        logging.info('= Matching Features...')
        if num_images < 2000:
            exhaustive_matching(feature_paths, dataset_paths, match_max_dist_ratio, min_num_matches)
        else:
            vocab_tree_path = path.join(dataset_path, 'Oxford5k/vocab-tree.bin');
            approximate_matching(feature_paths, dataset_paths, match_max_dist_ratio, min_num_matches, COLMAP_PATH, vocab_tree_path)
        logging.info('= Done!')
