from os import path
from PIL import Image

from tqdm import tqdm
import torch
import torch.nn.functional as NF
import torchvision.transforms.functional as F

from .extractors.superpoint import SuperPoint
from .fileio import write_keypoints, write_descriptors

def extract_features(extractor_type: str, 
                     image_paths:str, 
                     keypoint_paths:str, 
                     descriptor_paths:str, 
                     device:str = 'cpu'):
    ''' Extracts feature and saves it to binary file

    Given N image paths, and N keypoint / descriptor paths, call extractor's 
    extract function that will extract out binaries for evaluation.

    Arguments:
        extractor_type(str): superpoint | s
        image_paths(str): N paths to input image files. 
        keypoint_paths(str): N paths to output keypoint binary files. 
        descriptor_paths(str): N paths to output descriptor files. 
    '''
    dev = torch.device(device)
    if(extractor_type.lower() == 'superpoint'):
        extractor = SuperPoint()
        extractor = extractor.to(dev).eval()

    for i, image_path in tqdm(enumerate(image_paths)):
        keypoint_path = keypoint_paths[i]
        descriptor_path = descriptor_paths[i]

        keypoint_exists = path.exists(keypoint_path)
        descriptor_exists = path.exists(descriptor_path)

        # if(keypoint_exists and descriptor_exists):
        #     continue

        with Image.open(image_path) as img:
            torch_image = F.to_tensor(img)
            keypoints, descriptors = extractor.extract(torch_image)

        write_keypoints(keypoint_path, keypoints.cpu())
        write_descriptors(descriptor_path, descriptors.cpu())