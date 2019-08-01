import torch
import cv2
import numpy as np
from torch import nn
from .detector import Detector
from utils.features import *


bce_loss = nn.BCELoss()
class OrbDetector(Detector):
    def __init__(self, config):
        super(OrbDetector, self).__init__(config)
        self.orb = cv2.ORB_create()

    def detect(self, torch_image):
        N, C, H, W = torch_image.shape
        gray_data = torch_image.mean(1).unsqueeze(1).cpu()
        gray_numpy = (gray_data.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        corner_tensors = []
        heatmap_images = torch.zeros_like(gray_data).byte().squeeze(1)
        for n in range(N):
            kps = self.orb.detect(gray_numpy[n], None)
            for kp in kps:
                heatmap_images[n, int(kp.pt[1]), int(kp.pt[0])] = 1
        return heatmap_images

    def extract(self, torch_images, num_points=-1):

        keypoints = []
        descriptions = []

        if(num_points > 0):
            orb= self.orb.create(num_points)
        else:
            orb = self.orb


        for n in range(len(torch_images)):
            gray_data = torch_images[n].mean(0).unsqueeze(0).cpu()
            gray_numpy = (gray_data.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            kps, desc = orb.detectAndCompute(gray_numpy, None)

            kpst = torch.zeros((len(kps), 2)).float()
            for i, kp in enumerate(kps):
                kpst[i, 0] = kp.pt[1]
                kpst[i, 1] = kp.pt[0]
            if(desc is not None):
                dss = torch.from_numpy(desc).to(self.device)
            else:
                dss = []
            keypoints.append(kpst.to(self.device))
            descriptions.append(dss)
        return keypoints, descriptions

    def compute_loss(self, torch_image, heatmaps):
        s_heatmap = self.detect(torch_image).to(self.device)
        positives = -torch.log(heatmaps[s_heatmap].clamp(1e-4, 0.9999)).mean()
        negatives = -torch.log(1 - heatmaps[~s_heatmap].clamp(1e-4, 0.9999)).mean()

        return positives + negatives

    def compute_positive_loss(self, torch_image, heatmaps):
        s_heatmap = self.detect(torch_image).to(self.device)
        positives = -torch.log(heatmaps[s_heatmap].clamp(1e-4, 0.9999)).mean()
        return positives

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

    def compute_repeatable_loss(self, torch_image, intrinsics, extrinsics, depths, heatmaps):
        '''
        Given camera poses and gt depths, compute loss that uses 
        repeatable detection as positive loss
        '''
        raw_keypoints, desc= self.extract(torch_image)
        keypoints = filter_keypoints(raw_keypoints, depths)
        gt_matches = get_gt_matches(self.config.dist_eps, intrinsics, extrinsics, depths, keypoints)
        if(gt_matches is None):
            return 0

        zp_heatmaps = torch.zeros_like(heatmaps).byte()
        gt_ref_indices = gt_matches[0]
        gt_src_indices = gt_matches[1]
        occluded_rts = gt_matches[2]
        occluded_str = gt_matches[3]

        if((len(gt_ref_indices.size()) == 0) or (len(gt_src_indices.size()) == 0)  or (gt_ref_indices.shape[0] == 0) or (gt_src_indices.shape[0] == 0)):
            return 0
        # set positive examples from ground truth matches
        ref_kps = keypoints[0][gt_ref_indices].long()
        src_kps = keypoints[1][gt_src_indices].long()
        zp_heatmaps[0, ref_kps[:, 0], ref_kps[:, 1]] = 1
        zp_heatmaps[1, src_kps[:, 0], src_kps[:, 1]] = 1

        # set negative examples from all points that are not occluded that doesn't have match
        ref_occluded = keypoints[0][occluded_rts]
        src_occluded = keypoints[1][occluded_str]

        zn_heatmaps = zp_heatmaps.clone()
        zn_heatmaps[0, ref_occluded[:, 0], ref_occluded[:, 1]] = 1
        zn_heatmaps[1, src_occluded[:, 0], src_occluded[:, 1]] = 1

        positives = -torch.log(heatmaps[zp_heatmaps].clamp(1e-4, 0.9999)).mean()
        negatives = -torch.log(1 - heatmaps[~zn_heatmaps].clamp(1e-4, 0.9999)).mean()
        return positives + negatives