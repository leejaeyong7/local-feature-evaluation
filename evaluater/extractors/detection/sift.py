import torch
import cv2
import numpy as np
from torch import nn
from .detector import Detector

bce_loss = nn.BCELoss()
class SIFTDetector(Detector):
    def __init__(self, config):
        super(SIFTDetector, self).__init__(config)
        self.sift = cv2.xfeatures2d.SIFT_create()

    def detect(self, torch_image):
        N, C, H, W = torch_image.shape
        gray_data = torch_image.mean(1).unsqueeze(1).cpu()
        gray_numpy = (gray_data.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        corner_tensors = []
        heatmap_images = torch.zeros_like(gray_data).byte().squeeze(1)
        for n in range(N):
            kps = self.sift.detect(gray_numpy[n], None)
            for kp in kps:
                heatmap_images[n, int(kp.pt[1]), int(kp.pt[0])] = 1
        return heatmap_images

    def extract(self, torch_images, num_points=-1):
        keypoints = []
        descriptions = []
        if(num_points > 0):
            sift = self.sift.create(num_points)
        else:
            sift = self.sift

        for n in range(len(torch_images)):
            gray_data = torch_images[n].mean(0).unsqueeze(0).cpu()
            gray_numpy = (gray_data.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            kps, desc = sift.detectAndCompute(gray_numpy, None)

            kpst = torch.zeros((len(kps), 2)).float()
            for i, kp in enumerate(kps):
                kpst[i, 0] = kp.pt[1]
                kpst[i, 1] = kp.pt[0]
            keypoints.append(kpst.to(self.device))
            descriptions.append(torch.from_numpy(desc).to(self.device))
        return keypoints, descriptions

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
