# -*- coding: utf-8 -*-
""" Entry for matching pipeline. """

# module imports
import os
from os import path
from .exhaustive_matching import exhaustive_matching
from .approximate_matching import approximate_matching 
from .feature_extraction_powerpoint import feature_extraction_powerpoint


# global variables
DATASET_NAMES = [
    'Fountain', 'Herzjesu', 'South-Building', 
    'Madrid_Metropolis', 'Gendarmenmarkt', 'Tower_of_London', 
    'Oxford5k', 'Alamo', 'Roman_Forum', 'ArtsQuad_dataset'
]

def mkdirp(p):
    if(not path.exists(p)):
        os.makedirs(p)

for dataset_name in DATASET_NAMES:
    # Set the pipeline parameters.
    # TODO: Change this to where your dataset is stored. This directory should
    #       contain an "images" folder and a "database.db" file.
    dataset_path = path.join('datasets/', dataset_name)

    # TODO: Change this to where the COLMAP build directory is located.
    COLMAP_PATH = 'colmap/build'

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
    image_names = [f for f in os.listdir(image_path) if (f != '.' and f != '..')]

    num_images = len(image_names)
    image_names = []
    image_paths = []
    keypoint_paths = []
    descriptor_paths = []

    for image_name in image_names:
        image_names.append(image_name)
        image_paths.append(path.join(image_path, image_name))
        image_paths.append(path.join(image_path, image_name))
        keypoint_paths.append(path.join(keypoint_path, image_name + '.bin'))
        descriptor_paths.append(path.join(descriptor_path, image_name + '.bin'))

    #  Compute the keypoints and descriptors.
    feature_extraction_powerpoint(image_paths, keypoint_paths, descriptor_paths)

    dataset_paths = {
        'image': image_path,
        'keypoint': keypoint_path,
        'descriptor': descriptor_path,
        'match': match_path,
        'database': database_path,
        'dataset': dataset_path,
    }
    feature_paths= {
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
    if num_images < 2000:
        exhaustive_matching(feature_paths, dataset_paths, match_max_dist_ratio, min_num_matches)
    else:
        vocab_tree_path = path.join(dataset_path, 'Oxford5k/vocab-tree.bin');
        approximate_matching(feature_paths, dataset_paths, match_max_dist_ratio, min_num_matches, COLMAP_PATH, vocab_tree_path)
