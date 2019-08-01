import torch
import cv2
import numpy as np
from torch import nn
from model.powerpoint import PowerPointNet
import torch.nn.functional as NF
from .detector import Detector
from utils.features import *

bce_loss = nn.BCELoss()
class PowerPointDetector(Detector):
    def __init__(self, config):
        super(PowerPointDetector, self).__init__(config)
        self.model = PowerPointNet(config)
        self.model.eval()

    def detect(self, torch_image):
        N, C, H, W = torch_image.shape
        with torch.no_grad():
            heatmap,_  = self.model(torch_image)
        return heatmap

    def extract(self, torch_images):
        keypoints = []
        descriptors = []
        with torch.no_grad():
            for i, torch_image in enumerate(torch_images):
                detection, description = self.model(torch_image.unsqueeze(0))
                heatmaps = detection.squeeze(1)

                kpts = get_keypoints_fast(self.config.conf_thresh, self.config.nms_thresh, heatmaps)
                desc = get_descriptors_by_list(kpts, description)
                keypoints.append(kpts[0].float())
                descriptors.append(desc[0])

        return keypoints, descriptors

    def compute_loss(self, torch_image, heatmaps):
        raise NotImplementedError

    def soft_detection_loss(self, torch_image, soft_detection):
        raise NotImplementedError