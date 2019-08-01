import torch
import torch.nn as nn
import logging
import time
import torchvision.transforms.functional as F
import torch.nn.functional as NF
import numpy as np

def batch_norm_2d(ic):
    return nn.Sequential(
        nn.BatchNorm2d(ic, momentum=0.9),
        nn.LeakyReLU(0.2))

def batch_norm_1d(ic):
    return nn.Sequential(
        nn.BatchNorm1d(ic, momentum=0.9),
        nn.LeakyReLU(0.2))

def feat_building_block(ks, ic, oc):
    pad = (ks - 1) // 2 
    bb_pre_bn = batch_norm_2d(ic)
    bb_conv_1 = nn.Conv2d(ic, oc, ks, padding=pad, stride=1)
    bb_mid_bn = batch_norm_2d(oc)
    bb_conv_2 = nn.Conv2d(oc, oc, ks, padding=pad, stride=1)
    return nn.Sequential(bb_pre_bn, bb_conv_1, bb_mid_bn, bb_conv_2)

def desc_conv(ic, oc):
    return nn.Sequential(
        nn.Conv2d(ic, oc, 3, padding=1, stride=2),
        batch_norm_2d(oc)
    )
class LFNet(nn.Module):
    '''
    LFNet PyTorch Implementation.
    '''
    def __init__(self):
        '''
        '''
        super(LFNet, self).__init__()
        self.max_scale = np.sqrt(2)
        self.min_scale = self.max_scale / 2.0
        self.num_scales = 5
        self.top_k = 500
        self.crop_radius = 16
        self.nms_thresh = 0.0
        self.nms_ksize = 5
        self.patch_size = 32

        self.scale_log_factors = np.linspace(np.log(self.max_scale), np.log(self.min_scale), self.num_scales)
        self.scale_factors = np.exp(self.scale_log_factors)

        # feature extraction
        self.init_conv = nn.Conv2d(1, 16, 5, padding=2, stride=1)
        self.bb_1 = feat_building_block(5, 16, 16)
        self.bb_2 = feat_building_block(5, 16, 16)
        self.bb_3 = feat_building_block(5, 16, 16)
        self.fin_bn = batch_norm_2d(16)

        # detection
        self.scale_1 = nn.Conv2d(16, 1, 5, padding=2, stride=1)
        self.scale_2 = nn.Conv2d(16, 1, 5, padding=2, stride=1)
        self.scale_3 = nn.Conv2d(16, 1, 5, padding=2, stride=1)
        self.scale_4 = nn.Conv2d(16, 1, 5, padding=2, stride=1)
        self.scale_5 = nn.Conv2d(16, 1, 5, padding=2, stride=1)

        self.scale_convs = [
            self.scale_1, self.scale_2, self.scale_3, self.scale_4, self.scale_5
        ]
        self.orient = nn.Conv2d(16, 2, 5, padding=2, stride=1)

        self.desc_1 = desc_conv(1, 64)
        self.desc_2 = desc_conv(64, 128)
        self.desc_3 = desc_conv(128, 256)

        self.desc_convs = [self.desc_1, self.desc_2, self.desc_3]

        self.desc_fc_1 = nn.Linear(4096, 512)
        self.desc_fc_bn_1 = batch_norm_1d(512)
        self.desc_fc_2 = nn.Linear(512, 256)

        # yolo parameters
        self.device = torch.device('cpu')

    def to(self, device):
        super(LFNet, self).to(device)
        self.device = device
        return self

    def feature_extract(self, gray_image):
        x = self.init_conv(gray_image)
        x = self.bb_1(x) + x
        x = self.bb_2(x) + x
        x = self.bb_3(x) + x
        feat_map = self.fin_bn(x)
        return feat_map

    def orient_features(self, feat_map):
        B, C, H, W = feat_map.shape
        ori_maps = self.orient(feat_map)
        ori_maps = ori_maps / ori_maps.norm(2, dim=1).unsqueeze(1)
        score_maps = []
        for i, scale in enumerate(self.scale_factors):
            resized_feat_map = NF.interpolate(feat_map, scale_factor=1/scale, mode='bilinear')
            resized_score_map = self.scale_convs[i](resized_feat_map)
            normalized_resized_score_map = NF.instance_norm(resized_score_map, eps=1e-3)
            normalized_score_map = NF.interpolate(normalized_resized_score_map, size=(H, W), mode='bilinear')
            score_maps.append(normalized_score_map)
        scale_maps = torch.cat(score_maps, 1)
        return scale_maps, ori_maps

    def softmax_heatmap(self, scale_maps):
        sf = torch.tensor(self.scale_factors).float().to(self.device)
        scale_heatmaps = self._softmaxpool3d(scale_maps, ksize=15, com_strength=3.0)
        max_heatmaps, max_scales = self._soft_max_and_argmax_1d(scale_heatmaps, sf)
        return max_heatmaps, max_scales

    def detect(self, max_heatmaps):
        full_pad_size = 10
        _, _, H, W = max_heatmaps.shape
        eof_masks_pad = self._end_of_frame_masks(H, W, full_pad_size).view(1, 1, H, W)
        max_heatmaps = max_heatmaps * eof_masks_pad

        # Extract Top-K keypoints
        eof_masks_crop = self._end_of_frame_masks(H, W, self.crop_radius).view(1, 1, H, W)
        nms_maps = self._non_max_suppression(max_heatmaps, self.nms_thresh, self.nms_ksize).float()
        nms_scores = max_heatmaps * nms_maps * eof_masks_crop

        top_ks = self._make_top_k_sparse_tensor(nms_scores, k=self.top_k)
        top_ks = top_ks * nms_maps

        kpts, batch_inds, num_kpts = self._extract_keypoints(top_ks)
        return kpts, batch_inds, num_kpts

    def describe(self, patches):
        x = patches
        for dc in self.desc_convs:
            x = dc(x)
        FS = 4096
        x = self.desc_fc_1(x.view(-1, FS))
        x = self.desc_fc_bn_1(x)
        x = self.desc_fc_2(x)
        raw_feats = x
        return raw_feats / raw_feats.norm(2, dim=1).unsqueeze(1)

    def get_keypoint_orientation(self, max_scales, ori_maps, batch_inds, kpts):
        kpts_scale = self._batch_gather_keypoints(max_scales, batch_inds, kpts)
        kpts_ori = self._batch_gather_keypoints(ori_maps, batch_inds, kpts)
        return kpts_scale, kpts_ori


    def forward(self, x):
        B, C, H, W = x.shape
        data = x.mean(1).unsqueeze(1)

        # extract features
        feat_map = self.feature_extract(data)

        # extract scale / orientation
        scale_maps, ori_maps = self.orient_features(feat_map)

        # get heatmap for detection
        max_heatmaps, max_scales = self.softmax_heatmap(scale_maps)

        # detect features
        kpts, batch_inds, num_kpts = self.detect(max_heatmaps)

        # get patch orientation
        kpts_scale, kpts_ori = self.get_keypoint_orientation(max_scales, ori_maps, batch_inds, kpts)

        # get patches
        patches = self._transformer_crop(data, batch_inds, kpts, kpts_scale=kpts_scale, kpts_ori=kpts_ori)

        # get descriptors
        desc = self.describe(patches)
        yx_kpts = kpts[:, [1, 0]]
        return yx_kpts.float(), desc.float()

    #####################
    ## Helper functions
    def _softmaxpool3d(self, scale_maps, ksize, com_strength=1.0):
        B, C, H, W = scale_maps.shape
        pad = (ksize - 1) // 2
        scale_maps_3d = scale_maps.view(B, 1, C, H, W)
        nmp = NF.max_pool3d(scale_maps_3d, (C, ksize, ksize), stride=(C, 1, 1), padding=(0, pad, pad))
        nmp_img = nmp.view(B, 1, H, W)
        
        weight = torch.ones((1, 1, C, ksize, ksize)).float().to(self.device)
        exps_img = torch.exp(com_strength* (scale_maps - nmp_img))
        exps = exps_img.view(B, 1, C, H, W)
        sums = NF.conv3d(exps, weight, stride=(C, 1, 1), padding=(0, pad, pad))
        sums_img = sums.view(B, 1, H, W)
        return exps_img / (sums_img + 1e-6)

    def _soft_max_and_argmax_1d(self, inputs, inputs_index, strength1=100.0, strength2=100.0):
        inputs_exp1 = torch.exp(strength1 * (inputs - inputs.max(1)[0].unsqueeze(1)))
        inputs_softmax1 = inputs_exp1 / (inputs_exp1.sum(1).unsqueeze(1) + 1e-8)
        
        
        inputs_exp2 = torch.exp(strength2 * (inputs - inputs.max(1)[0].unsqueeze(1)))
        inputs_softmax2 = inputs_exp2 / (inputs_exp2.sum(1).unsqueeze(1) + 1e-8)
        
        inputs_max = (inputs * inputs_softmax1).sum(1).unsqueeze(1)
        
        inputs_amax = (inputs_index.view(1, -1, 1, 1) * inputs_softmax2).sum(1).unsqueeze(1)
        return inputs_max, inputs_amax

    def _end_of_frame_masks(self, height, width, radius):
        mask = torch.ones((height, width)).float().to(self.device)
        mask[:radius, :] = 0
        mask[:, :radius] = 0
        mask[-radius:, :] = 0
        mask[:, -radius:] = 0
        return mask

    def _non_max_suppression(self, inputs, thresh=0.0, ksize=3):
        B, C, H, W = inputs.shape
        
        hk = ksize // 2
        zeros = torch.zeros_like(inputs).to(self.device)
        works = inputs.clone()
        works[inputs < thresh] = 0
        works_pad = NF.pad(works, (2*hk, 2*hk, 2*hk, 2*hk), mode='constant')
        map_augs = []
        
        # works_pad = B x C x (-hk, H, hk) x (-hk, W, hk)
        
        for i in range(ksize):
            for j in range(ksize):
                curr_in = works_pad[:, :,  i:i + (H + 2*hk), j:j+(W+2*hk)]
                map_augs.append(curr_in)
        # all_maps = torch.stack(map_augs)
        # max_maps = all_maps.max(0)[1]
        # num_map = len(map_augs) # ksize*ksize
        # return max_maps == num_map // 2

        num_map = len(map_augs) # ksize*ksize
        center_map = map_augs[num_map//2]
        peak_mask = center_map > map_augs[0]
        for n in range(1, num_map):
            if n == num_map // 2:
                continue
            peak_mask = peak_mask & (center_map > map_augs[n])
            
        peak_mask = peak_mask[:, :, hk:hk+H, hk:hk+W]
        return peak_mask

    def _soft_argmax_2d(self, patches, do_softmax=True, com_strength=10):
        # patches = N x 1 x PS x PS
        # Returns the relative soft-argmax position, in the -1 to 1 coordinate
        # system of the patch
        N, O, PH, PW = patches.shape
        
        H = PH
        W = PW

        # H x W
        x_t = (torch.ones((H, 1)).float() * torch.linspace(-1, 1, W).unsqueeze(0)).to(self.device)
        y_t = (torch.linspace(-1, 1, H).unsqueeze(1) * torch.ones((1, W)).float()).to(self.device)
        
        # xy_grid = 1 x 2 x H x W => 1 x 2 x (HW)
        xy_grid = torch.stack([x_t, y_t], dim=1).unsqueeze(0).view(1, 2, -1)

        maxes = patches
        if do_softmax:
            # flat patches = N x 1 x HW
            flat_patches = patches.view(N, O, -1)
            
            # exps = N x 1 x HW
            exps = torch.exp(com_strength * (flat_patches - flat_patches.max(2, keep_dims=True)))
            
            # maxes = N x 1 x HW
            maxes = exps / (exps.sum(2, keep_dims=True) + 1e-8)
        # N x 2 x HW => N x 2
        dxdy = (xy_grid * maxes).sum(2).view(-1, 2)

        return dxdy

    def _meshgrid(self, H, W):
        '''
        Given H, W, outputs 3 x HW tensor
        '''
        xt = (torch.ones((H, 1)) * torch.linspace(-1, 1, W).unsqueeze(0)).to(self.device)
        yt = (torch.linspace(-1, 1, H).unsqueeze(1) * torch.ones((1, W))).to(self.device)
        
        xtf = xt.view(1, -1)
        ytf = yt.view(1, -1)
        
        os = torch.ones_like(xtf).to(self.device)
        return torch.cat((xtf, ytf, os))
        
    def _transformer_crop(self, images, batch_inds, kpts_xy, kpts_scale=None, kpts_ori=None, thetas=None):
        '''
        # images : [B,H,W,C]
        # out_size : (out_width, out_height)
        # batch_inds : [B*K,] tf.int32 [0,B)
        # kpts_xy : [B*K,2] tf.float32 or whatever
        # kpts_scale : [B*K,] tf.float32
        # kpts_ori : [B*K,2] tf.float32 (cos,sin)
        
        
        kpts_scale = B, tensor of scales
        thetas = B x 3 x 3
        grid = B x 3 x 1
        '''
        PH = self.patch_size
        PW = self.patch_size
        hoW = PW // 2
        hoH = PH // 2
        B, C, H, W = images.shape
        NK = batch_inds.shape[0]
        
        max_y = (H - 1)
        max_x = (W - 1)
        
        # 3 x (HW) => B x 3 x (HW)
        grid = self._meshgrid(PH, PW).unsqueeze(0).repeat(NK, 1, 1)
        if(thetas is None):
            # thetas = B x 2 x 3
            thetas = torch.eye(3)[:2].unsqueeze(0).repeat(NK, 1, 1).to(self.device)
            
            # apply scale
            if(kpts_scale is not None):
                thetas = thetas * kpts_scale[:, None]
            ones = torch.tensor([[[0, 0, 1]]]).float().repeat(NK, 1, 1).to(self.device)
            
            # thetas = B x 3 x 3
            thetas = torch.cat((thetas, ones), 1) 
            
            # multiply orientation
            if(kpts_ori is not None):
                # cos = B x 1
                cos = kpts_ori[:, 0:1]
                sin = kpts_ori[:, 1:2]
                zeros = torch.zeros_like(cos).to(self.device)
                ones = torch.ones_like(cos).to(self.device)
                
                # B x 3 x 3
                R = torch.cat([
                    cos, -sin, zeros,
                    sin, cos, zeros,
                    zeros, zeros, ones
                ], dim=-1).view(NK, 3, 3)
                
                thetas = thetas.matmul(R)
        
        T_g = thetas.matmul(grid)
        x = T_g[:, 0:1, :]
        y = T_g[:, 1:2, :]
        
        x = x * PW / 2.0
        y = y * PH / 2.0
        kpts_xy = kpts_xy.float()
        
        # 
        kp_x_offset = kpts_xy[:, 0:1].unsqueeze(1)
        kp_y_offset = kpts_xy[:, 1:2].unsqueeze(1)
        
        # create grid cell coords
        x = x + kp_x_offset
        y = y + kp_y_offset
        
        x = x.view(-1)
        y = y.view(-1)
        
        x0 = x.floor().long()
        x1 = x0 + 1
        y0 = y.floor().long()
        y1 = y0 + 1
        
        # clamp to appropriate range
        x0 = x0.clamp(0, max_x)
        x1 = x1.clamp(0, max_x)
        y0 = y0.clamp(0, max_y)
        y1 = y1.clamp(0, max_y)
        
        base = batch_inds.unsqueeze(1).repeat(1, PW * PH).view(-1) * W * H
        base_y0 = base + y0 * W
        base_y1 = base + y1 * W
        
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1
        
        im_flat = images.permute(0, 2, 3, 1).view(-1, C)
        Ia = im_flat[idx_a, :]
        Ib = im_flat[idx_b, :]
        Ic = im_flat[idx_c, :]
        Id = im_flat[idx_d, :]

        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        
        wa = ((x1_f-x) * (y1_f-y)).unsqueeze(1)
        wb = ((x1_f-x) * (y-y0_f)).unsqueeze(1)
        wc = ((x-x0_f) * (y1_f-y)).unsqueeze(1)
        wd = ((x-x0_f) * (y-y0_f)).unsqueeze(1)

        output = wa*Ia + wb*Ib + wc*Ic + wd*Id
        output = output.view(NK, PH, PW, C).permute(0, 3, 1, 2)
        return output
        
    def _make_top_k_sparse_tensor(self, heatmaps, k=256, get_kpts=False):
        B, C, H, W = heatmaps.shape
        heatmaps_flt = heatmaps.view(B, -1)
        imsize = C * H * W
        res = torch.topk(heatmaps_flt, k, sorted=False)
        values = res.values
        xy_indices = res.indices
        
        # boffset = B,
        boffset = (torch.tensor(list(range(B))) * imsize).unsqueeze(1).to(self.device)
        indices = xy_indices + boffset
        indices = indices.view(1, -1)
        
        ones = torch.ones((B*k,)).to(self.device)
        top_k_maps = torch.sparse.FloatTensor(indices, ones, torch.Size([B * imsize])).to_dense()
        top_k_maps = top_k_maps.view(B, C, H, W).float()
        
        return top_k_maps

    def _extract_keypoints(self, top_k):
        B, C, H, W = top_k.shape
        coords = (top_k > 0).nonzero()
        num_kpts = top_k.view(B, -1).sum(1)
        
        batch_inds, _, kp_y, kp_x = torch.split(coords, 1, dim=-1)
        batch_inds = batch_inds.view(-1)
        
        kpts = torch.cat([kp_x, kp_y], dim=1)

        num_kpts = num_kpts.long()
        # kpts: [N,2] (N=B*K)
        # batch_inds: N,
        # num_kpts: B
        return kpts, batch_inds, num_kpts

    def _batch_gather_keypoints(self, inputs, batch_inds, kpts, xy_order=True):
        # kpts: [N,2] x,y or y,x
        # batch_inds: [N]
        # outputs = inputs[b,y,x]
        if xy_order:
            kp_x, kp_y = torch.split(kpts, 1, dim=1)
        else:
            kp_y, kp_x = torch.split(kpts, 1, dim=1)
            
        byx = torch.cat([batch_inds.unsqueeze(1), kp_y, kp_x], dim=1)
        return inputs[byx[:, 0], :, byx[:, 1], byx[:, 2]]

