import torch
import cv2
import numpy as np
from .detector import Detector
# import harris_affine
from torch import nn

bce_loss = nn.BCELoss()
class HarrisAffineDetector(Detector):
    def __init__(self, config):
        super(HarrisAffineDetector, self).__init__(config)

    def detect(self, torch_image):
        N, C, H, W = torch_image.shape
        gray_data = torch_image.mean(1).cpu()
        gray_numpy = gray_data.numpy()
        heatmap_images = torch.zeros_like(gray_data).byte()
        for n in range(N):
            kps = harris_affine.detect(gray_numpy[n], -1, 3, 0.0001, 10)
            kps_index = kps[:, 4].argsort(0)[:: -1]
            skps= kps[kps_index]
            heatmap_images[n, skps[:300, 1].astype(np.int), skps[:300, 0].astype(np.int)] = 1
        return heatmap_images

    def extract(self, torch_images, num_points=-1):
        keypoints = []
        for torch_image in torch_images:
            gray_image = torch_image.mean(0).cpu().numpy()

            kps = harris_affine.detect(gray_image, -1, 3, 0.0001, 10)
            kps_index = kps[:, 4].argsort(0)[:: -1]
            skps= kps[kps_index]
            keypoints.append(torch.from_numpy(skps[:num_points, [1, 0]]))
        return keypoints, []

    def compute_loss(self, torch_image, heatmaps):
        s_heatmap = self.detect(torch_image).to(self.device)
        positives = -torch.log(heatmaps[s_heatmap]).mean()
        negatives = -torch.log(1 - heatmaps[~s_heatmap]).mean()
        return positives + negatives

    def soft_detection_loss(self, torch_image, soft_detection):
        '''
        Arguments:
            - detection: 
        '''
        orb_heatmap = self.detect(torch_image).to(self.device)
        N, C, H8, W8 = soft_detection.shape

        orb_detection = orb_heatmap.view(N, H8, 8, W8, 8).permute(0, 2, 4, 1, 3)
        od = orb_detection.contiguous().view(N, 64, H8, W8)
        nod = od.sum(1).unsqueeze(1) == 0

        detection_class = torch.cat((od, nod), dim=1).float()

        # clamp probability for stability
        loss = bce_loss(soft_detection.clamp(1e-5, 0.9999), detection_class)
        return loss
