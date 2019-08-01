import torch
import cv2
import numpy as np
from torch import nn
from model.superpoint import SuperPointNet
import torch.nn.functional as NF
from .detector import Detector
from utils.features import *

bce_loss = nn.BCELoss()
class SuperPointDetector(Detector):
    def __init__(self, config):
        super(SuperPointDetector, self).__init__(config)
        # self.device = torch.device('cpu')
        self.model = SuperPointNet()
        superpoint_pretrained_path = '../pretrained/superpoint.pth'
        self.model.load_state_dict(torch.load(superpoint_pretrained_path))
        self.model = self.model.to(self.device)
        self.model.eval()

    def detect(self, torch_image):
        N, C, H, W = torch_image.shape
        gray_data = torch_image.mean(1).unsqueeze(1)
        with torch.no_grad():
            detection, _ = self.model(gray_data)
        soft_detection = NF.softmax(detection, 1)
        return soft_detection

    def extract(self, torch_images, num_points=None):
        keypoints = []
        descriptors = []
        with torch.no_grad():
            for torch_image in torch_images:
                C, H, W = torch_image.shape
                gray_data = torch_image.mean(0).view(1, 1, H, W).cpu()
                detection, description = self.model(gray_data)

                soft_detection = NF.softmax(detection, dim=1)
                heatmaps = get_heatmaps(soft_detection)
                kpts = get_keypoints(0.025, self.config.nms_thresh, heatmaps)
                desc = get_descriptors_by_list(kpts, description)
                keypoints.append(kpts[0].float()[:num_points])
                descriptors.append(desc[0][:num_points])

        return keypoints, descriptors


    def compute_loss(self, torch_image, heatmaps):
        soft_detection = self.detect(torch_image).to(self.device)
        N, _, H8, W8 = soft_detection.shape

        H = H8 * 8
        W = W8 * 8
        # return ((s_soft_detection - soft_detection) ** 2).mean()
        # nodust  64xH/8xW/8
        # we want to convert it to 8xH8x 8xW8
        # then, H x W
        nodust = soft_detection[:, :-1]
        heatmap = nodust.view(N, 8, 8, H8, W8)
        heatmap = heatmap.permute(0, 3, 1, 4, 2)
        heatmap = heatmap.contiguous().view(N, H, W)
        s_heatmaps = get_heatmaps(soft_detection)
        kpts = get_keypoints(0.025, self.config.nms_thresh, s_heatmaps)
        z_heatmaps = torch.zeros_like(s_heatmaps).byte()
        for n, z_heatmap in enumerate(z_heatmaps):
            z_heatmap[kpts[n][:, 0], kpts[n][:, 1]] = 1

        positives = -torch.log(heatmaps[z_heatmaps].clamp(1e-4, 0.9999))
        negatives = -torch.log(1 - heatmaps[~z_heatmaps].clamp(1e-4, 0.9999))
        return positives.mean() + negatives.mean()

    def compute_repeatable_loss(self, torch_image, intrinsics, extrinsics, depths, heatmaps):
        '''
        Given camera poses and gt depths, compute loss that uses 
        repeatable detection as positive loss
        '''
        gray_images = torch_image.mean(1).unsqueeze(1)
        with torch.no_grad():
            detection, description = self.model(gray_images)
        soft_detection = NF.softmax(detection, dim=1)

        N, _, H8, W8 = soft_detection.shape
        if(N != 2):
            raise NotImplementedError

        H = H8 * 8
        W = W8 * 8

        nodust = soft_detection[:, :-1]

        heatmap = nodust.view(N, 8, 8, H8, W8)
        heatmap = heatmap.permute(0, 3, 1, 4, 2)
        heatmap = heatmap.contiguous().view(N, H, W)

        s_heatmaps = get_heatmaps(soft_detection)

        zp_heatmaps = torch.zeros_like(s_heatmaps).byte()

        # get kpts descs
        keypoints = get_keypoints(0.015, self.config.nms_thresh, s_heatmaps)

        keypoints = filter_keypoints(keypoints, depths)
        descriptors = get_descriptors_by_list(keypoints, description.to(self.device))
        gt_matches = get_gt_matches(self.config.dist_eps, intrinsics, extrinsics, depths, keypoints)
        if(gt_matches is None):
            return 0
        gt_ref_indices = gt_matches[0]
        gt_src_indices = gt_matches[1]
        occluded_rts = gt_matches[2]
        occluded_str = gt_matches[3]

        if((len(gt_ref_indices.size()) == 0) or (len(gt_src_indices.size()) == 0)  or (gt_ref_indices.shape[0] == 0) or (gt_src_indices.shape[0] == 0)):
            return 0
        # set positive examples from ground truth matches
        ref_kps = keypoints[0][gt_ref_indices]
        src_kps = keypoints[1][gt_src_indices]
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


    def soft_detection_loss(self, torch_image, soft_detection):
        '''
        Arguments:
            - detection: 
        '''
        s_soft_detection = self.detect(torch_image).to(self.device)
        N, _, H8, W8 = soft_detection.shape
        N, C, H8, W8 = soft_detection.shape

        orb_detection = s_soft_detection.view(N, H8, 8, W8, 8).permute(0, 2, 4, 1, 3)
        od = orb_detection.contiguous().view(N, 64, H8, W8)
        nod = od.sum(1).unsqueeze(1) == 0

        detection_class = torch.cat((od, nod), dim=1).float()

        # clamp probability for stability
        loss = bce_loss(soft_detection.clamp(1e-5, 0.9999), detection_class)
        return loss