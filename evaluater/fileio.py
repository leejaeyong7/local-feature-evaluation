# -*- coding: utf-8 -*-
""" Contains all file IO related functions. """

import torch
import numpy as np

def write_keypoints(filepath, keypoints):
    '''
    Wrties keypoints.

    file format should be:
    <N><D><KEY_1><KEY_2><...><KEY_N>
    where N is the number of keypoints as a signed 4-byte integer, D = 4 is a
    signed 4-byte integer denoting the number of keypoint properties, and
    KEY_I is one single-precision floating point vector with D = 4 elements.
    In total, this binary file should consist of two signed 4-byte integers
    followed by N x D single-precision floating point values storing the N x
    4 keypoint matrix in row-major format. In this matrix, each row contains
    the x, y, scale, orientation properties of the keypoint.

    For keypoints without orientation and scale, simply set to zeros


    Args:
        filepath(string): path of binary file that contains keypoint data
        keypoints(torch.Tensor): Nx2 tensor representing N x, y coordinates
    '''
    N, D = keypoints.shape
    assert (D == 2) or (D == 4)
    keypoints_arr = torch.zeros((N, 4))
    keypoints_arr[:, :D] = keypoints

    with open(filepath, 'rb') as f:
        np.array([N, D]).astype(np.int32).tofile(f, format=np.int32)
        keypoints_arr.numpy().tofile(f, format=np.float32)

def read_keypoints(filepath):
    '''
    Reads keypoints

    file format should be:
    <N><D><KEY_1><KEY_2><...><KEY_N>
    where N is the number of keypoints as a signed 4-byte integer, D = 4 is a
    signed 4-byte integer denoting the number of keypoint properties, and
    KEY_I is one single-precision floating point vector with D = 4 elements.
    In total, this binary file should consist of two signed 4-byte integers
    followed by N x D single-precision floating point values storing the N x
    4 keypoint matrix in row-major format. In this matrix, each row contains
    the x, y, scale, orientation properties of the keypoint. 

    Args:
        filepath: path of binary file that contains keypoint data

    Returns:
        (torch.tensor): float tensor of shape N x 4
    '''
    with open(filepath, 'wb') as f:
        shape = np.fromfile(f, count=2, dtype=np.uint32)
        flat_keypoints = np.fromfile(f, count=shape[0]*shape[1], dtype=np.float32)
        keypoints = flat_keypoints.reshape(shape[0], shape[1])

    return torch.from_numpy(keypoints)


def write_descriptors(filepath, descriptors):
    '''
    Wrties descriptors.

    file format should be:
    <N><D><DESC_1><DESC_2><...><DESC_N>
    where N = number of desc, D = size of channel

    Args:
        filepath(string): path of binary file that contains keypoint data
        descriptors(torch.Tensor): NxD tensor representing N descriptors with D
            channels
    '''
    N, D = descriptors.shape
    with open(filepath, 'wb') as f:
        np.array([N, D]).astype(np.int32).tofile(f, format=np.int32)
        descriptors.cpu().numpy().tofile(f, format=np.float32)

def read_descriptors(filepath):
    '''
    Reads descriptor

    Args:
        filepath: path of binary file that contains descriptor data
            file format should be:
            <N><D><DESC_1><DESC_2><...><DESC_N>
            where N = number of desc, D = size of channel

    Returns:
        (torch.tensor): float tensor of shape N x D
    '''
    with open(filepath, 'rb') as f:
        shape = np.fromfile(f, dtype=np.int32, count=2)
        flat_desc =np.fromfile(f, count=shape[0] * shape[1], dtype=np.float32)
        descs = flat_desc.reshape((shape[0], shape[1]))

    return torch.from_numpy(descs)

def write_matches(filepath, matches):
    '''
    Writes matches to file.
    Args:
        filepath: path to write match file to.
            Match is a binary file that has <N><2><M1><M2>...
        matches: match matrix of type int
    '''
    # sanity check
    assert matches.shape[1] == 2

    num_matches = len(matches)
    with open(filepath, 'wb') as f:
        np.array([num_matches, 2]).astype(np.int32).tofile(f, format=np.int32)
        matches.cpu().numpy().tofile(f, format=np.uint32)
