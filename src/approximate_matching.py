# -*- coding: utf-8 -*-
""" Approximate matching pipeline using Bag-of-Words as a nearest neighbor metric. """

from os import path
from .match_descriptors import match_descriptors
from .fileio import *

def approximate_matching(feature_paths,dataset_paths, match_max_dist_ratio, min_num_matches, colmap_path, vocab_tree_path):
    binary = path.join(colmap_path, 'src', 'tools', 'vocab_tree_retreiver_float')
    database_path = dataset_paths['database']
    dataset_path = dataset_paths['dataset']
    descriptor_path = dataset_paths['descriptor']
    descriptor_paths = feature_paths['descriptors']

    command = [
        binary,
        '--database_path', database_path,
        '--descriptor_path', descriptor_path,
        '--vocab_tree_path', vocab_tree_path
    ]

    # TODO: run vocabtree command

    # retrieve results
    status, output = subprocess.call(command)
    assert(status == 0)


    # write results to file
    results_path =path.join(dataset_path)
    with open(results_path, 'w'):
        f.write(output)

    with open(results_path) as f:
        result_lines = f.readlines()


    image_names = feature_paths['names']
    num_images = len(image_names)
    image_name_to_idx = {
        image_name: i
        for i, image_name in enumerate(image_names)
    }

    num_matched_images = 0;


    result_iterator = iter(result_lines)
    for tline in result_iterator:
        if(tline[:8] == 'Querying'):
            query_results = tline[19:].split('[')
            query_image_idx = image_name_to_idx[query_results[0]]
            retrieved_image_idxs = []


            tline = next(result_iterator)
            while(tline[:8] != 'Querying'):
                retrieval_results = tline.split(',')
                retrieved_image_name = retrieval_results[2]
                retrieved_image_name = retrieved_image_name[12:]
                retrieved_image_idxs.append(image_name_to_idx[retrieved_image_name])
                tline = next(result_iterator)
            num_matched_images = num_matched_images + 1;

        # load desctipros
        descriptors = {}
        
        # perform matching
        for retrieved_image_idx in retrieved_image_idxs:
            # Avoid self-matching.
            if query_image_idx == retrieved_image_idx:
                continue

            if(query_image_idx < retrieved_image_idx):
                idx1 = query_image_idx
                idx2 = retrieved_image_idx
            else:
                idx2 = query_image_idx
                idx1 = retrieved_image_idx

            # load in descriptor iff it is loaded
            if(idx1 not in descriptors):
                desc1_path = descriptor_paths[idx1]
                desc1 = read_descriptors(desc1_path)
                descriptors[idx1] = desc1
            else:
                desc1 = descriptors[desc1]

            if(idx2 not in descriptors):
                desc2_path = descriptor_paths[idx2]
                desc2 = read_descriptors(desc2_path)
                descriptors[idx2] = desc2
            else:
                desc2 = descriptors[desc2]

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
