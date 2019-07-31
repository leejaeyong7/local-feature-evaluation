# -*- coding: utf-8 -*-
""" Python code for exhaustive matching. """

import torch
from os import path
from .fileio import *
from .match_descriptors import match_descriptors

def exhaustive_matching(feature_paths,dataset_paths, match_max_dist_ratio, min_num_matches):
    image_names = feature_paths['names']
    descriptor_paths = feature_paths['descriptors']
    match_path = dataset_paths['match']
    num_images = len(feature_paths['names'])

    for idx1 in range(num_images):
        for idx2 in range(idx1, num_images):
            desc1_path = descriptor_paths[idx1]
            desc2_path = descriptor_paths[idx2]
            desc1 = read_descriptors(desc1_path)
            desc2 = read_descriptors(desc2_path)
            matches_path = path.join('{}---{}.bin'.format(image_names[idx1], image_names[idx2]))

            if(path.exists(matches_path)):
                continue

            # actually perform matching
            matches = match_descriptors(desc1, desc2, match_max_dist_ratio)
            num_matches = matches.shape[0]

            # set matches to 0 if there are no matches found
            if(num_matches < min_num_matches):
                matches = torch.zeros((0, 2)).Long()

            write_matches(matches_path, matches)
            