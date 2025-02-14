import torch
import cv2
import numpy as np
from torch import nn
from model.lf_net import LFNet
import torch.nn.functional as NF
from .detector import Detector
from utils.features import get_heatmaps, get_keypoints, get_descriptors 

bce_loss = nn.BCELoss()
class LFNetDetector(Detector):
    def __init__(self, config):
        super(LFNetDetector, self).__init__(config)
        self.model = LFNet()
        lf_net_pretrained_path = '../pretrained/lf_net.pth'
        self.model.load_state_dict(torch.load(lf_net_pretrained_path))
        self.model.eval()

    def detect(self, torch_image):
        with torch.no_grad():
            kpts, desc = self.model(torch_image.unsqueeze(0))
        return kpts

    def extract(self, torch_images, num_points=-1):
        keypoints = []
        descriptors = []
        with torch.no_grad():
            for torch_image in torch_images:
                kpts, desc = self.model(torch_image.unsqueeze(0))
                keypoints.append(kpts[:num_points])
                descriptors.append(desc[:num_points])
        return keypoints, descriptors

    def compute_loss(self, torch_image, heatmaps):
        raise NotImplementedError

    def soft_detection_loss(self, torch_image, soft_detection):
        raise NotImplementedError