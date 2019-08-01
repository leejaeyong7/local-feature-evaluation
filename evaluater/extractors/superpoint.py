#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2018
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Daniel DeTone (ddetone)
#                       Tomasz Malisiewicz (tmalisiewicz)
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%


import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch
import torch.nn.functional as NF

class SuperPoint(torch.nn.Module):
    '''
    Super Point Net from original authors.
    '''
    def __init__(self):
        '''
        Initializes confiden
        '''
        super(SuperPointNet, self).__init__()

        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, 3, 1, 1)
        self.conv1b = torch.nn.Conv2d(c1, c1, 3, 1, 1)
        self.conv2a = torch.nn.Conv2d(c1, c2, 3, 1, 1)
        self.conv2b = torch.nn.Conv2d(c2, c2, 3, 1, 1)
        self.conv3a = torch.nn.Conv2d(c2, c3, 3, 1, 1)
        self.conv3b = torch.nn.Conv2d(c3, c3, 3, 1, 1)
        self.conv4a = torch.nn.Conv2d(c3, c4, 3, 1, 1)
        self.conv4b = torch.nn.Conv2d(c4, c4, 3, 1, 1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, 3, 1, 1)
        self.convPb = torch.nn.Conv2d(c5, 65, 1, 1, 0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, 3, 1, 1)
        self.convDb = torch.nn.Conv2d(c5, d1, 1, 1, 0)

    def extract(self, image, nms_thresh=4, num_points=-1):
        '''
        '''
        with torch.no_grad():
            C, H, W = image.shape
            gray_data = image.mean(0).view(1, 1, H, W).cpu()
            det, desc =  self.forward(gray_data)
            soft_detection = NF.softmax(det, dim=1)
            heatmaps = self.get_heatmaps(soft_detection)
            kpts = self.get_keypoints(0.025, nms_thresh, heatmaps)
            desc = self.get_descriptors(kpts, desc)
            # keypoints should be in x, y order
            keypoints = kpts[0].float()[:num_points, [1, 0]]
            descriptors = desc[0][:num_points]
        return keypoints, descriptors

    def get_heatmaps(self, soft_detection):
        '''
        Given detection map, 
        Argument:
            - detection: N x 65 x H/8 x W/8
        Returns:
            - heatmap: N x H x W attention values
        '''
        # attention = 1 x H x W
        # description = 256 x H/8 x W/8
        N, _, H8, W8 = soft_detection.shape
        H = H8 * 8
        W = W8 * 8

        # nodust  64xH/8xW/8
        # we want to convert it to 8xH8x 8xW8
        # then, H x W
        nodust = soft_detection[:, :-1]
        heatmap = nodust.view(N, 8, 8, H8, W8)
        heatmap = heatmap.permute(0, 3, 1, 4, 2)
        heatmap = heatmap.contiguous().view(N, H, W)
        return heatmap

    def get_keypoints_fast(conf_thresh, nms_thresh, heatmaps, max_num_keys=300):

        '''
        Arguments:
            - heatmap: N x H x W attention values
            - conf_thresh: threshold for filtering heatmap
            - nms_thresh: threshold for non-maximum-suppression
        Returns:
            - feature points: N x 2 coordinates for features, in yx order
        '''

        # perform non maximum suppression
        ks = nms_thresh * 2 + 1

        max_h = NF.max_pool2d(heatmaps, ks, padding=nms_thresh, stride=1)
        nms_map = heatmaps == max_h
        conf_map = heatmaps > conf_thresh

        filtereds = heatmaps * (nms_map & conf_map).float()
        keypoints = []
        for filtered in filtereds.squeeze(1):
            kpts = filtered.nonzero()
            values = filtered[kpts[:, 0], kpts[:, 1]]
            val, inds = torch.sort(values, descending=True)
            keypoints.append(kpts[inds[:max_num_keys]])
        return keypoints

    def get_keypoints(self, conf_thresh, nms_thresh, heatmaps):

        '''
        Arguments:
            - heatmap: N x H x W attention values
            - conf_thresh: threshold for filtering heatmap
            - nms_thresh: threshold for non-maximum-suppression
        Returns:
            - feature points: N x 2 coordinates for features, in yx order
        '''
        keypoints = []
        for heatmap in heatmaps:
            H, W = heatmap.shape
            

            conf_heatmap = heatmap > conf_thresh
            confidences = heatmap[conf_heatmap]
            points = (conf_heatmap).nonzero()

            sorted_conf = torch.argsort(confidences, descending=True)
            feature_points = points[sorted_conf]
            dev = torch.device(feature_points.device)

            # apply non-max suppression
            suppressed = _non_max_supp(W, H, feature_points.cpu(), nms_thresh)
            keypoints.append(suppressed.to(dev))
        return keypoints

    def get_descriptors(self, points, descriptions):
        '''
        Arguments:
            - points: AxNx2 coordinates
            - description: AxCx H/8 x W/8 descriptor tensor
        Returns:
            - feature_descrption: AxN x C feature descriptors
        '''
        # if((points[0].shape[0] == 0 or points[1].shape[0] == 0)):
        #     return [[], []]

        descriptors = []
        for i, description in enumerate(descriptions):
            C, Hc, Wc = description.shape
            W = Wc * 8
            H = Hc * 8

            grid_intrinsics = torch.tensor([
                2.0 / W, 0, -1,
                0, 2.0 / H, -1,
                0, 0, 1
            ], device=description.device).view(3, 3)

            num_points = points[i].shape[0]
            nms_homo_coord = torch.ones((num_points, 3), device=description.device).float()
            nms_homo_coord[:, 0] = points[i][:, 1]
            nms_homo_coord[:, 1] = points[i][:, 0]
            nms_grid_coord = grid_intrinsics.matmul(nms_homo_coord.t()).t()
            nms_grid_coord = nms_grid_coord[:, :2]
            grid_sample_coords = nms_grid_coord.view(1, 1, -1, 2)
            
            
            desc = NF.grid_sample(description.unsqueeze(0), grid_sample_coords).view(C, -1)
            desc_norm = torch.sqrt((desc ** 2).sum(0).unsqueeze(0))

            descriptors.append((desc / desc_norm).t())
            
        return descriptors

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
        x: Image pytorch tensor shaped N x 1 x H x W.
        Output
        semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
        desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
        return semi, desc
