from .extractors.superpoint import SuperPoint
from .fileio import *
import torch.nn.functional as NF
import torchvision.transforms.functional as F
from PIL import Image

def extract_features(extractor_type: str, image_paths, keypoint_paths, descriptor_paths):
    if(extractor_type.lower() == 'superpoint'):
        extractor = SuperPoint()

    for i, image_path in enumerate(image_paths):
        keypoint_path = keypoint_paths[i]
        descriptor_paths = descriptor_paths[i]

        with Image.open(image_path) as img:
            torch_image = F.to_tensor(img)
            keypoints, descriptors = extractor.extract(torch_image)
        write_keypoints(keypoint_path, keypoints)
        write_descriptors(descriptor_paths, descriptors)