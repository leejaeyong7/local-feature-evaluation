import torch
import cv2
import numpy as np
from torch import nn

class Sift():
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()

    def extract(self, image, num_points=-1):
        if(num_points > 0):
            sift = self.sift.create(num_points)
        else:
            sift = self.sift

        gray_data = image.mean(0).unsqueeze(0).cpu()
        gray_cv = (gray_data.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        kps, desc = sift.detectAndCompute(gray_cv, None)

        kpst = torch.zeros((len(kps), 4)).float()
        for i, kp in enumerate(kps):
            kpst[i, 0] = kp.pt[0]
            kpst[i, 1] = kp.pt[1]
            kpst[i, 3] = kp.scale
            kpst[i, 4] = kp.orientation

        keypoints = kpst.to(self.device)
        descriptions = torch.from_numpy(desc).to(self.device)
        return keypoints, descriptions