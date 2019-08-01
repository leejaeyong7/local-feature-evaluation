import torch
import cv2
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.nn.functional as NF
import numpy as np
import scipy
import scipy.ndimage
from scipy.linalg import lu_factor, lu_solve
import six
from six.moves import xrange
import time

# global configs


def ghh(data, num_in_sum, num_in_max):
    B = data.shape[0]
    C = data.shape[1]

    # data_to_max = C // num_in_max tuple
    data_to_max = torch.split(data, num_in_max ,dim=1)
    
    maxed_data = torch.cat([
        d.max(1)[0].unsqueeze(1)
        for d in data_to_max
    ], 1)
    
    # Create delta
    delta = (1.0 - 2.0 * (torch.arange(num_in_sum) % 2)).float()
    delta = delta.view(1, 1, 1, num_in_sum)
    
    # 
    data_to_sum = torch.split(maxed_data, num_in_sum, dim=1)

    return torch.cat([
        d.sum(1).unsqueeze(1)
        for d in data_to_sum
    ], dim=1)

def cap(ic, oc, ks):
    c = nn.Conv2d(ic, oc, ks, stride=1, padding=0)
    a = nn.ReLU()
    p = nn.MaxPool2d(2)
    return nn.Sequential(c, a, p)

def cam(ic, oc, ks, ps):
    c = nn.Conv2d(ic, oc, ks, stride=1, padding=0)
    a = nn.ReLU()
    m = nn.AvgPool2d(ps)
    return nn.Sequential(c, a, m)

def cs_norm(data):
    eps = 1e-10
    cur_in_abs_max = torch.abs(data).max(1)[0].unsqueeze(1).clamp_min(eps)
    x = data / cur_in_abs_max
    eps = 1e-3
    x = x + (x >= 0).float() * eps - (x < 0).float() * eps
    x_n = torch.sqrt((x ** 2).sum(1).unsqueeze(1))
    return x / x_n

class LIFT(nn.Module):
    '''
    '''
    def __init__(self):
        '''
        '''
        super(LIFT, self).__init__()
        self.scl_intv = 4
        self.min_scale_log2 = 1
        self.max_scale_log2 = 4
        self.kp_input_size = 48
        self.kp_base_scale = 2.0
        self.kp_filter_size = 25
        self.desc_support_ratio = 6.0
        self.desc_input_size = 64
        self.ori_input_size = 64
        self.test_nearby_ratio = 1.0
        self.test_num_keypoint = 1000
        self.ratio_scale = (float(self.kp_input_size) / 2.0) / self.kp_base_scale
        self.patch_size = float(self.desc_input_size) * self.ratio_scale / self.desc_support_ratio
        nearby = int(0.5 * (self.kp_input_size - 1) * float(self.desc_input_size) / float(self.patch_size))
        self.fNearbyRatio = self.test_nearby_ratio * 0.25
        self.nearby = max(int(nearby * self.fNearbyRatio), 1)
        self.nms_intv = 2
        self.edge_th = 10.0
        self.kp_mean = 115.39129698199088
        self.kp_std = 71.30249864214325
        self.ori_mean = 128.0
        self.ori_std = 128.0
        self.desc_mean = 128.0
        self.desc_std = 128.0


        # ---------------------
        # Modules
        self.conv_ghh_1 = nn.Conv2d(1, 16, self.kp_filter_size, stride=1, padding=0)
        self.ori_conv_act_pool_1 = cap(1, 10, 5)
        self.ori_conv_act_pool_2 = cap(10, 20, 5)
        self.ori_conv_act_pool_3 = cap(20, 50, 3)
        self.ori_fc_1 = nn.Linear(50*5*5, 100*4*4)
        self.ori_fc_2 = nn.Linear(100, 2*4*4)
        self.desc_conv_act_pool_norm_1 = cam(1, 32, 7, 2)
        self.desc_conv_act_pool_norm_2 = cam(32, 64, 6, 3)
        self.desc_conv_act_pool_norm_3 = cam(64, 128, 5, 4)


    def _detect_features(self, x):
        x = self.conv_ghh_1(x)
        return ghh(x, 4, 4)

    def detect(self, x, scales_to_test, resize_to_test):
        '''
        takes gray image, outputs array of score maps
        '''
        N, C, H, W = x.shape

        x = (x * 255 - self.kp_mean) / self.kp_std

        # Run for each scale
        test_res_list = []
        for resize in resize_to_test:
            NH = int(H * resize)
            NW = int(W * resize)
            r_gray_image = NF.interpolate(x, size=(NH, NW), mode='bilinear')
            
            scoremap = self._detect_features(r_gray_image).detach()
            ps = (self.kp_filter_size - 1) // 2
            padded_scoremap = NF.pad(scoremap, pad=(ps, ps), value=-float('Inf'), mode='constant')
            test_res_list.append(padded_scoremap)
        res_list = [
            rl.permute(0, 2, 3, 1)[0, :, :, 0].numpy()
            for rl in test_res_list
        ]
        XYZS = self._get_XYZS_from_res_list(
            res_list, resize_to_test, scales_to_test, do_interpolation=True,
        )
        XYZS = XYZS[:self.test_num_keypoint]
        kp_list = self._XYZS2kpList(XYZS)

        # create np array
        kp = np.asarray(kp_list)
        return kp

    def orient(self, patch, kp, xyz):
        patch = (patch.astype('float') - self.ori_mean) / self.ori_std

        # crop patch
        thetas = self._make_theta(xyz=xyz, cs=None, rr=(self.desc_input_size / self.patch_size))
        patch_tensor = torch.from_numpy(patch).float() / 255
        cropped = self._transformer(patch_tensor, thetas=thetas, output_size=int(self.ori_input_size))
        # compute orientation
        x = cropped
        x = self.ori_conv_act_pool_1(x)
        x = self.ori_conv_act_pool_2(x)
        x = self.ori_conv_act_pool_3(x)
        x = x.view(-1, 50*5*5)
        x = self.ori_fc_1(x)
        x = ghh(x, 4, 4)
        x = self.ori_fc_2(x)
        x = ghh(x, 4, 4)
        orientations = cs_norm(x).detach()
        # orient patch
        thetas = self._make_theta(xyz=xyz, cs=orientations.numpy(), rr=(float(self.desc_input_size) / float(self.patch_size)))
        patch_tensor = torch.from_numpy(patch).float() / 255
        oriented = self._transformer(patch_tensor, thetas=thetas, output_size=int(self.desc_input_size))
        oriented = oriented * 255
        return oriented

    def describe(self, oriented):
        x = (oriented - self.desc_mean) / self.desc_std
        x = self.desc_conv_act_pool_norm_1(x)
        x = self.desc_conv_act_pool_norm_2(x)
        x = self.desc_conv_act_pool_norm_3(x)
        return x.view(-1, 128).detach()

    def get_patches(self, x, kp):
        IDX_ANGLE = 3
        gray_np_img = (x.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

        # Use load patches function
        # Assign dummy values to y, ID, angle
        y = np.zeros((len(kp),))
        ID = np.zeros((len(kp),), dtype='int64')
        angle = np.pi / 180.0 * kp[:, IDX_ANGLE]  # store angle in radians
        return self._load_patches(gray_np_img[0], kp, y, ID, angle, self.ratio_scale, int(self.patch_size), self.desc_input_size)

    def forward(self, x):
        x = x.mean(1).unsqueeze(1)

        scales_to_test, resize_to_test = self._set_scale_ranges(x)
        kp = self.detect(x, scales_to_test, resize_to_test)

        # patches = N x 1 x 128 x 128 np array
        patch, xyz, angles, kp = self.get_patches(x, kp)

        # oriented = N x 1 x 64 x 64 tensor
        oriented = self.orient(patch, kp, xyz)
        description = self.describe(oriented)

        kps = torch.from_numpy(np.stack((kp[:, 1], kp[:, 0]), axis=1)).float()

        return kps, description.float()


    def _set_scale_ranges(self, x):
        '''
        Given image, check scales to test
        Returns:
          - array of scales, resizes to test
        '''
        N, C, H, W = x.shape

        # Test starting with double scale if small image
        min_hw = min(H, W)

        min_scale_log2 = self.min_scale_log2
        max_scale_log2 = self.max_scale_log2

        # for the case of testing on same scale, do not double scale
        if min_hw <= 1600 and min_scale_log2!=max_scale_log2:
            min_scale_log2 -= 1

        # range of scales to check
        num_division = (max_scale_log2 - min_scale_log2) * (self.scl_intv + 1) + 1
        scales_to_test = 2**np.linspace(min_scale_log2, max_scale_log2,
                                        num_division)

        # convert scale to image resizes
        resize_to_test = ((float(self.kp_input_size - 1) / 2.0) / (self.ratio_scale * scales_to_test))

        # check if resize is valid
        min_hw_after_resize = resize_to_test * min_hw
        is_resize_valid = min_hw_after_resize > self.kp_filter_size + 1

        # if there are invalid scales and resizes
        if not np.prod(is_resize_valid):
            # find first invalid
            # first_invalid = np.where(True - is_resize_valid)[0][0]
            first_invalid = np.where(~is_resize_valid)[0][0]

            # remove scales from testing
            scales_to_test = scales_to_test[:first_invalid]
            resize_to_test = resize_to_test[:first_invalid]
        return scales_to_test, resize_to_test


    #---------------------------------
    # Helper functions
    #---------------------------------
    def _get_XYZS_from_res_list(self, res_list, resize_to_test, scales_to_test, do_interpolation=False, fScaleEdgeness=0.0):
        # NMS
        nms_res = self._nonMaxSuppression(res_list)

        # check if it is none
        if len(nms_res) == 1:
            XYZS = self._get_subpixel_XYZ(res_list, nms_res, resize_to_test, scales_to_test, do_interpolation, fScaleEdgeness)
        else:
            XYZS = self._get_subpixel_XYZS(res_list, nms_res, resize_to_test, scales_to_test, do_interpolation, fScaleEdgeness)

        # sort by score
        XYZS = XYZS[np.argsort(XYZS[:, 3])[::-1]]

        return XYZS


    def _get_subpixel_XYZ(self, score_list, nms_list, resize_to_test, scales_to_test, do_interpolation, fScaleEdgeness):
        # this doos not consider scales, works for single scale

        #    log_scales = np.log2(scales_to_test)
        # avoid crash when testing on single scale
        #    if len(scales_to_test)>1:
        #        log_scale_step = ((np.max(log_scales) - np.min(log_scales)) /
        #                          (len(scales_to_test) - 1.0))
        #    else:
        #        log_scale_step = 0 #is this right??? I (Lin) added this line here

        X = [()] * len(nms_list)
        Y = [()] * len(nms_list)
        Z = [()] * len(nms_list)
        S = [()] * len(nms_list)
        for idxScale in xrange(len(nms_list)):
            nms = nms_list[idxScale]

            pts = np.where(nms)
            # when there is no nms points, jump out from this loop
            if len(pts[0]) == 0:
                continue

            # will assert when 0>0 , I changed here ****
    #        assert idxScale > 0 and idxScale < len(nms_list) - 1
            if len(nms_list) != 1:
                assert idxScale > 0 and idxScale < len(nms_list) - 1

            # the conversion function
            def at(dx, dy):
                if not isinstance(dx, np.ndarray):
                    dx = np.ones(len(pts[0]),) * dx
                if not isinstance(dy, np.ndarray):
                    dy = np.ones(len(pts[0]),) * dy
                new_pts = (pts[0] + dy, pts[1] + dx)
                new_pts = tuple([np.round(v).astype(int)
                                for v in zip(new_pts)])
                scores_to_return = np.asarray([
                    score_list[idxScale][_y, _x]
                    for _x, _y in zip(
                        new_pts[1], new_pts[0]
                    )
                ])
                return scores_to_return

            # compute the gradient
            Dx = 0.5 * (at(+1, 0) - at(-1, 0))
            Dy = 0.5 * (at(0, +1) - at(0, -1))

            # compute the Hessian
            Dxx = (at(+1, 0) + at(-1, 0) - 2.0 * at(0, 0))
            Dyy = (at(0, +1) + at(0, -1) - 2.0 * at(0, 0))

            Dxy = 0.25 * (at(+1, +1) + at(-1, -1) -
                        at(-1, +1) - at(+1, -1))

            # filter out all keypoints which we have inf
            is_good = (np.isfinite(Dx) * np.isfinite(Dy) * np.isfinite(Dxx) *
                    np.isfinite(Dyy) * np.isfinite(Dxy))
            Dx = Dx[is_good]
            Dy = Dy[is_good]
            Dxx = Dxx[is_good]
            Dyy = Dyy[is_good]
            Dxy = Dxy[is_good]

            pts = tuple([v[is_good[0]] for v in pts])
    #        pts = tuple([v[is_good] for v in pts])

            # check if empty
            if len(pts[0]) == 0:
                continue

            # filter out all keypoints which are on edges
            if self.edge_th > 0:

                # # re-compute the Hessian
                # Dxx = (at(b[:, 0] + 1, b[:, 1], b[:, 2]) +
                #        at(b[:, 0] - 1, b[:, 1], b[:, 2]) -
                #        2.0 * at(b[:, 0], b[:, 1], b[:, 2]))
                # Dyy = (at(b[:, 0], b[:, 1] + 1, b[:, 2]) +
                #        at(b[:, 0], b[:, 1] - 1, b[:, 2]) -
                #        2.0 * at(b[:, 0], b[:, 1], b[:, 2]))

                # Dxy = 0.25 * (at(b[:, 0] + 1, b[:, 1] + 1, b[:, 2]) +
                #               at(b[:, 0] - 1, b[:, 1] - 1, b[:, 2]) -
                #               at(b[:, 0] - 1, b[:, 1] + 1, b[:, 2]) -
                #               at(b[:, 0] + 1, b[:, 1] - 1, b[:, 2]))

                # H = np.asarray([[Dxx, Dxy, Dxs],
                #                 [Dxy, Dyy, Dys],
                #                 [Dxs, Dys, Dss]]).transpose([2, 0, 1])

                edge_score = (Dxx + Dyy) * (Dxx + Dyy) / (Dxx * Dyy - Dxy * Dxy)
                is_good = ((edge_score >= 0) *
                        (edge_score < (self.edge_th + 1.0)**2 / self.edge_th))

                Dx = Dx[is_good]
                Dy = Dy[is_good]
                Dxx = Dxx[is_good]
                Dyy = Dyy[is_good]
                Dxy = Dxy[is_good]
                pts = tuple([v[is_good] for v in pts])
                # check if empty
                if len(pts[0]) == 0:
                    continue

            b = np.zeros((len(pts[0]), 3))
            if do_interpolation:
                # from VLFEAT

                # solve linear system
                A = np.asarray([[Dxx, Dxy],
                                [Dxy, Dyy], ]).transpose([2, 0, 1])

                b = np.asarray([-Dx, -Dy]).transpose([1, 0])

                b_solved = np.zeros_like(b)
                for idxPt in xrange(len(A)):
                    b_solved[idxPt] = lu_solve(lu_factor(A[idxPt]), b[idxPt])

                b = b_solved

            # throw away the ones with bad subpixel localizatino
            is_good = ((abs(b[:, 0]) < 1.5) * (abs(b[:, 1]) < 1.5))
            b = b[is_good]
            pts = tuple([v[is_good] for v in pts])
            # check if empty
            if len(pts[0]) == 0:
                continue

            x = pts[1] + b[:, 0]
            y = pts[0] + b[:, 1]
            log_ds = np.zeros_like(b[:, 0])

            S[idxScale] = at(b[:, 0], b[:, 1])
            X[idxScale] = x / resize_to_test[idxScale]
            Y[idxScale] = y / resize_to_test[idxScale]
            Z[idxScale] = scales_to_test[idxScale] * 2.0**(log_ds)
    #        Z[idxScale] = scales_to_test[idxScale] * 2.0**(log_ds * log_scale_step)

        X = np.concatenate(X)
        Y = np.concatenate(Y)
        Z = np.concatenate(Z)
        S = np.concatenate(S)

        XYZS = np.concatenate([X.reshape([-1, 1]),
                            Y.reshape([-1, 1]),
                            Z.reshape([-1, 1]),
                            S.reshape([-1, 1])],
                            axis=1)

        return XYZS


    def _get_subpixel_XYZS(self, score_list, nms_list, resize_to_test, scales_to_test, do_interpolation, fScaleEdgeness):

        log_scales = np.log2(scales_to_test)
        # avoid crash when testing on single scale
        if len(scales_to_test) > 1:
            log_scale_step = ((np.max(log_scales) - np.min(log_scales)) /
                            (len(scales_to_test) - 1.0))
        else:
            log_scale_step = 0  # is this right??? I (Lin) added this line here

        X = [()] * len(nms_list)
        Y = [()] * len(nms_list)
        Z = [()] * len(nms_list)
        S = [()] * len(nms_list)
        for idxScale in xrange(len(nms_list)):
            nms = nms_list[idxScale]

            pts = np.where(nms)
            # when there is no nms points, jump out from this loop
            if len(pts[0]) == 0:
                continue

            # will assert when 0>0 , I changed here ****
    #        assert idxScale > 0 and idxScale < len(nms_list) - 1
            if len(nms_list) != 1:
                assert idxScale > 0 and idxScale < len(nms_list) - 1

            # compute ratio for coordinate conversion
            fRatioUp = (
                (np.asarray(score_list[idxScale + 1].shape, dtype='float') - 1.0) /
                (np.asarray(score_list[idxScale].shape, dtype='float') - 1.0)
            ).reshape([2, -1])
            fRatioDown = (
                (np.asarray(score_list[idxScale - 1].shape, dtype='float') - 1.0) /
                (np.asarray(score_list[idxScale].shape, dtype='float') - 1.0)
            ).reshape([2, -1])

            # the conversion function
            def at(dx, dy, ds):
                if not isinstance(dx, np.ndarray):
                    dx = np.ones(len(pts[0]),) * dx
                if not isinstance(dy, np.ndarray):
                    dy = np.ones(len(pts[0]),) * dy
                if not isinstance(ds, np.ndarray):
                    ds = np.ones(len(pts[0]),) * ds
                new_pts = (pts[0] + dy, pts[1] + dx)
                ds = np.round(ds).astype(int)
                fRatio = ((ds == 0).reshape([1, -1]) * 1.0 +
                        (ds == -1).reshape([1, -1]) * fRatioDown +
                        (ds == 1).reshape([1, -1]) * fRatioUp)
                assert np.max(ds) <= 1 and np.min(ds) >= -1
                new_pts = tuple([np.round(v * r).astype(int)
                                for v, r in zip(new_pts, fRatio)])
                scores_to_return = np.asarray([
                    score_list[idxScale + _ds][_y, _x]
                    for _ds, _x, _y in zip(
                        ds, new_pts[1], new_pts[0]
                    )
                ])
                return scores_to_return

            # compute the gradient
            Dx = 0.5 * (at(+1, 0, 0) - at(-1, 0, 0))
            Dy = 0.5 * (at(0, +1, 0) - at(0, -1, 0))
            Ds = 0.5 * (at(0, 0, +1) - at(0, 0, -1))

            # compute the Hessian
            Dxx = (at(+1, 0, 0) + at(-1, 0, 0) - 2.0 * at(0, 0, 0))
            Dyy = (at(0, +1, 0) + at(0, -1, 0) - 2.0 * at(0, 0, 0))
            Dss = (at(0, 0, +1) + at(0, 0, -1) - 2.0 * at(0, 0, 0))

            Dxy = 0.25 * (at(+1, +1, 0) + at(-1, -1, 0) -
                        at(-1, +1, 0) - at(+1, -1, 0))
            Dxs = 0.25 * (at(+1, 0, +1) + at(-1, 0, -1) -
                        at(-1, 0, +1) - at(+1, 0, -1))
            Dys = 0.25 * (at(0, +1, +1) + at(0, -1, -1) -
                        at(0, -1, +1) - at(0, +1, -1))

            # filter out all keypoints which we have inf
            is_good = (np.isfinite(Dx) * np.isfinite(Dy) * np.isfinite(Ds) *
                    np.isfinite(Dxx) * np.isfinite(Dyy) * np.isfinite(Dss) *
                    np.isfinite(Dxy) * np.isfinite(Dxs) * np.isfinite(Dys))
            Dx = Dx[is_good]
            Dy = Dy[is_good]
            Ds = Ds[is_good]
            Dxx = Dxx[is_good]
            Dyy = Dyy[is_good]
            Dss = Dss[is_good]
            Dxy = Dxy[is_good]
            Dxs = Dxs[is_good]
            Dys = Dys[is_good]
            pts = tuple([v[is_good] for v in pts])
            # check if empty
            if len(pts[0]) == 0:
                continue

            # filter out all keypoints which are on edges
            if self.edge_th > 0:

                # # re-compute the Hessian
                # Dxx = (at(b[:, 0] + 1, b[:, 1], b[:, 2]) +
                #        at(b[:, 0] - 1, b[:, 1], b[:, 2]) -
                #        2.0 * at(b[:, 0], b[:, 1], b[:, 2]))
                # Dyy = (at(b[:, 0], b[:, 1] + 1, b[:, 2]) +
                #        at(b[:, 0], b[:, 1] - 1, b[:, 2]) -
                #        2.0 * at(b[:, 0], b[:, 1], b[:, 2]))

                # Dxy = 0.25 * (at(b[:, 0] + 1, b[:, 1] + 1, b[:, 2]) +
                #               at(b[:, 0] - 1, b[:, 1] - 1, b[:, 2]) -
                #               at(b[:, 0] - 1, b[:, 1] + 1, b[:, 2]) -
                #               at(b[:, 0] + 1, b[:, 1] - 1, b[:, 2]))

                # H = np.asarray([[Dxx, Dxy, Dxs],
                #                 [Dxy, Dyy, Dys],
                #                 [Dxs, Dys, Dss]]).transpose([2, 0, 1])

                edge_score = (Dxx + Dyy) * (Dxx + Dyy) / (Dxx * Dyy - Dxy * Dxy)
                is_good = ((edge_score >= 0) *
                        (edge_score < (self.edge_th + 1.0)**2 / self.edge_th))

                if fScaleEdgeness > 0:
                    is_good = is_good * (
                        abs(Dss) > fScaleEdgeness
                    )

                Dx = Dx[is_good]
                Dy = Dy[is_good]
                Ds = Ds[is_good]
                Dxx = Dxx[is_good]
                Dyy = Dyy[is_good]
                Dss = Dss[is_good]
                Dxy = Dxy[is_good]
                Dxs = Dxs[is_good]
                Dys = Dys[is_good]
                pts = tuple([v[is_good] for v in pts])
                # check if empty
                if len(pts[0]) == 0:
                    continue

            b = np.zeros((len(pts[0]), 3))
            if do_interpolation:
                # from VLFEAT

                # solve linear system
                A = np.asarray([[Dxx, Dxy, Dxs],
                                [Dxy, Dyy, Dys],
                                [Dxs, Dys, Dss]]).transpose([2, 0, 1])

                b = np.asarray([-Dx, -Dy, -Ds]).transpose([1, 0])

                b_solved = np.zeros_like(b)
                for idxPt in xrange(len(A)):
                    b_solved[idxPt] = lu_solve(lu_factor(A[idxPt]), b[idxPt])

                b = b_solved

            # throw away the ones with bad subpixel localizatino
            is_good = ((abs(b[:, 0]) < 1.5) * (abs(b[:, 1]) < 1.5) *
                    (abs(b[:, 2]) < 1.5))
            b = b[is_good]
            pts = tuple([v[is_good] for v in pts])
            # check if empty
            if len(pts[0]) == 0:
                continue

            x = pts[1] + b[:, 0]
            y = pts[0] + b[:, 1]
            log_ds = b[:, 2]

            S[idxScale] = at(b[:, 0], b[:, 1], b[:, 2])
            X[idxScale] = x / resize_to_test[idxScale]
            Y[idxScale] = y / resize_to_test[idxScale]
            Z[idxScale] = scales_to_test[idxScale] * 2.0**(log_ds * log_scale_step)

        X = np.concatenate(X)
        Y = np.concatenate(Y)
        Z = np.concatenate(Z)
        S = np.concatenate(S)

        XYZS = np.concatenate([X.reshape([-1, 1]),
                            Y.reshape([-1, 1]),
                            Z.reshape([-1, 1]),
                            S.reshape([-1, 1])],
                            axis=1)

        return XYZS


    def _nonMaxSuppression(self, score_img_or_list):
        """ Performs Non Maximum Suppression.

        Parameters
        ----------
        score_img_or_list: nparray or list
            WRITEME
        """

        filter_size = (self.nearby * 2 + 1,) * 2

        if isinstance(score_img_or_list, list):
            bis2d = False
        else:
            bis2d = True

        if bis2d:
            score = score_img_or_list
            # max score in region
            max_score = scipy.ndimage.filters.maximum_filter(
                score, filter_size, mode='constant', cval=-np.inf
            )
            # second score in region
            second_score = scipy.ndimage.filters.rank_filter(
                score, -2, filter_size, mode='constant', cval=-np.inf
            )
            # min score in region to check infs
            min_score = scipy.ndimage.filters.minimum_filter(
                score, filter_size, mode='constant', cval=-np.inf
            )
            nonmax_mask_or_list = ((score == max_score) *
                                (max_score > second_score) *
                                np.isfinite(min_score))

        else:
            
            max2d_list = [
                scipy.ndimage.filters.maximum_filter(
                    score, filter_size, mode='constant', cval=-np.inf
                )
                for score in score_img_or_list
            ]

            second2d_list = [
                scipy.ndimage.filters.rank_filter(
                    score, -2, filter_size, mode='constant', cval=-np.inf
                )
                for score in score_img_or_list
            ]

            min2d_list = [
                scipy.ndimage.filters.minimum_filter(
                    score, filter_size, mode='constant', cval=-np.inf
                )
                for score in score_img_or_list
            ]

            nonmax2d_list = [(score == max_score) * (max_score > second_score) *
                            np.isfinite(min_score)
                            for score, max_score, second_score, min_score in
                            zip(score_img_or_list,
                                max2d_list,
                                second2d_list,
                                min2d_list)
                            ]

            nonmax_mask_or_list = [None] * len(nonmax2d_list)

            # for the single scale, there is no need to compare on multiple scales
            # and can directly jump out loop from here!
            if len(nonmax2d_list) == 1:
                for idxScale in xrange(len(nonmax2d_list)):
                    nonmax2d = nonmax2d_list[idxScale]
                    coord2d_max = np.where(nonmax2d)
                    nonmax_mask = np.zeros_like(nonmax2d)
                    # mark surviving points
                    nonmax_mask[coord2d_max] = 1.0
                nonmax_mask_or_list[idxScale] = nonmax_mask
                return nonmax_mask_or_list

            for idxScale in xrange(len(nonmax2d_list)):

                nonmax2d = nonmax2d_list[idxScale]
                max2d = max2d_list[idxScale]

                # prep output
                nonmax_mask = np.zeros_like(nonmax2d)

                # get coordinates of the local max positions of nonmax2d
                coord2d_max = np.where(nonmax2d)

                # range of other scales to look at
                scl_diffs = np.arange(-self.nms_intv, self.nms_intv + 1)
                scl_diffs = scl_diffs[scl_diffs != 0]

                # skip if we don't have the complete set
                if (idxScale + np.min(scl_diffs) < 0 or
                        idxScale + np.max(scl_diffs) > len(nonmax2d_list) - 1):
                    continue

                # Test on multiple scales to see if it is scalespace local max
                for scl_diff in scl_diffs:

                    scl_to_compare = idxScale + scl_diff

                    # look at the other scales max
                    max2d_other = max2d_list[scl_to_compare]
                    # compute ratio for coordinate conversion
                    fRatio \
                        = (np.asarray(max2d_other.shape, dtype='float') - 1.0) \
                        / (np.asarray(nonmax2d.shape, dtype='float') - 1.0)
                    # get indices for lower layer
                    coord_other = tuple([np.round(v * r).astype(int)
                                        for v, r in zip(coord2d_max, fRatio)])
                    # find good points which should survive
                    idxGood = np.where((max2d[coord2d_max] >
                                        max2d_other[coord_other]) *
                                    np.isfinite(max2d_other[coord_other])
                                    )

                    # copy only the ones that are good
                    coord2d_max = tuple([v[idxGood] for v in coord2d_max])

                # mark surviving points
                nonmax_mask[coord2d_max] = 1.0

                # special case when we are asked with single item in list
                # no chance to enter into here, move this out
                if len(nonmax2d_list) == 1:
                    # get coordinates of the local max positions of nonmax2d
                    coord2d_max = np.where(nonmax2d)
                    # mark surviving points
                    nonmax_mask[coord2d_max] = 1.0

                # add to list
                nonmax_mask_or_list[idxScale] = nonmax_mask

        return nonmax_mask_or_list


    def _update_affine(self, kp):
        """Returns an updated version of the keypoint.

        Note
        ----
        This function should be applied only to individual keypoints, not a list.

        """
        IDX_ANGLE = 3
        IDX_a, IDX_b, IDX_c = (6, 7, 8)
        IDX_A0, IDX_A2, IDX_A1, IDX_A3 = (9, 10, 11, 12)

        # Compute A0, A1, A2, A3
        S = np.asarray([[kp[IDX_a], kp[IDX_b]], [kp[IDX_b], kp[IDX_c]]])
        invS = np.linalg.inv(S)
        a = np.sqrt(invS[0, 0])
        b = invS[0, 1] / max(a, 1e-18)
        A = np.asarray([[a, 0], [b, np.sqrt(max(invS[1, 1] - b**2, 0))]])

        # We need to rotate first!
        cos_val = np.cos(np.deg2rad(kp[IDX_ANGLE]))
        sin_val = np.sin(np.deg2rad(kp[IDX_ANGLE]))
        R = np.asarray([[cos_val, -sin_val], [sin_val, cos_val]])

        A = np.dot(A, R)

        kp[IDX_A0] = A[0, 0]
        kp[IDX_A1] = A[0, 1]
        kp[IDX_A2] = A[1, 0]
        kp[IDX_A3] = A[1, 1]

        return kp

    def _create_perturb(self, orig_pos, nPatchSize, nDescInputSize, fPerturbInfo):
        # Generate random perturbations
        perturb_xyz = ((2.0 * np.random.rand(orig_pos.shape[0], 3) - 1.0) *
                    fPerturbInfo.reshape([1, 3]))

        return perturb_xyz

    def _apply_perturb(self, orig_pos, perturb_xyz, maxRatioScale):
        # get the new scale
        new_pos_s = orig_pos[2] * (2.0**(-perturb_xyz[2]))

        # get xy to pixels conversion
        xyz_to_pix = new_pos_s * maxRatioScale

        # Get the new x and y according to scale. Note that we multiply the
        # movement we need to take by 2.0**perturb_xyz since we are moving at a
        # different scale
        new_pos_x = orig_pos[0] - perturb_xyz[0] * 2.0**perturb_xyz[2] * xyz_to_pix
        new_pos_y = orig_pos[1] - perturb_xyz[1] * 2.0**perturb_xyz[2] * xyz_to_pix

        perturbed_pos = np.asarray([new_pos_x, new_pos_y, new_pos_s])

        return perturbed_pos

    def _get_crop_range(self, xx, yy, half_width):
        """Function for retrieving the crop coordinates"""

        xs = np.cast['int'](np.round(xx - half_width))
        xe = np.cast['int'](np.round(xx + half_width))
        ys = np.cast['int'](np.round(yy - half_width))
        ye = np.cast['int'](np.round(yy + half_width))

        return xs, xe, ys, ye

    def _crop_patch(self, img, cx, cy, clockwise_rot, resize_ratio, nPatchSize):
        # Below equation should give (nPatchSize-1)/2 when M x [cx, 0, 1]',
        # 0 when M x [cx - (nPatchSize-1)/2*resize_ratio, 0, 1]', and finally,
        # nPatchSize-1 when M x [cx + (nPatchSize-1)/2*resize_ratio, 0, 1]'.
        dx = (nPatchSize - 1.0) * 0.5 - cx / resize_ratio
        dy = (nPatchSize - 1.0) * 0.5 - cy / resize_ratio
        M = np.asarray([[1. / resize_ratio, 0.0, dx],
                        [0.0, 1. / resize_ratio, dy],
                        [0.0, 0.0, 1.0]])
        # move to zero base before rotation
        R_pre = np.asarray([[1.0, 0.0, -(nPatchSize - 1.0) * 0.5],
                            [0.0, 1.0, -(nPatchSize - 1.0) * 0.5],
                            [0.0, 0.0, 1.0]])
        # rotate
        theta = clockwise_rot / 180.0 * np.pi
        R_rot = np.asarray([[np.cos(theta), -np.sin(theta), 0.0],
                            [np.sin(theta), np.cos(theta), 0.0],
                            [0.0, 0.0, 1.0]])
        # move back to corner base
        R_post = np.asarray([[1.0, 0.0, (nPatchSize - 1.0) * 0.5],
                            [0.0, 1.0, (nPatchSize - 1.0) * 0.5],
                            [0.0, 0.0, 1.0]])
        # combine
        R = np.dot(R_post, np.dot(R_rot, R_pre))

        crop = cv2.warpAffine(img, np.dot(R, M)[:2, :], (nPatchSize, nPatchSize))
        return crop

    def _load_patches(self, img, kp_in, y_in, ID_in, angle_in, fRatioScale, nPatchSize, nDescInputSize):
        fMaxScale = 1.0
        in_dim = 1
        bPerturb = False
        fPerturbInfo = np.zeros((3,))
        bReturnCoords = True
        sAugmentCenterRandMethod="uniform"
        nPatchSizeAug=nPatchSize
        fAugmentCenterRandStrength=0.0
        fAugmentRange=180.0
        nAugmentedRotations=1
        
        # get max possible scale ratio
        maxRatioScale = fRatioScale * fMaxScale

        # pre-allocate maximum possible array size for data
        x = np.zeros((kp_in.shape[0] * nAugmentedRotations, in_dim,
                    nPatchSizeAug, nPatchSizeAug), dtype='uint8')
        y = np.zeros((kp_in.shape[0] * nAugmentedRotations,), dtype='float32')
        ID = np.zeros((kp_in.shape[0] * nAugmentedRotations,), dtype='int')
        pos = np.zeros((kp_in.shape[0] * nAugmentedRotations, 3), dtype='float')
        angle = np.zeros((kp_in.shape[0] * nAugmentedRotations,), dtype='float32')
        coords = np.tile(np.zeros_like(kp_in), (nAugmentedRotations, 1))

        # create perturbations
        # Note: the purturbation still considers only the nPatchSize
        perturb_xyz = self._create_perturb(kp_in, nPatchSize,
                                    nDescInputSize, fPerturbInfo)
        
        perturb_xyz[y_in == 0] = 0

        idxKeep = 0
        for idx in six.moves.xrange(kp_in.shape[0]):

            # current kp position
            cur_pos = self._apply_perturb(kp_in[idx], perturb_xyz[idx], maxRatioScale)
            cx = cur_pos[0]
            cy = cur_pos[1]
            cs = cur_pos[2]

            # retrieve the half width acording to scale
            max_hw = cs * maxRatioScale

            # get crop range considering bigger area
            xs, xe, ys, ye = self._get_crop_range(cx, cy, max_hw * np.sqrt(2.0))

            # boundary check with safety margin
            safety_margin = 1
            # if xs < 0 or xe >= img.shape[1] or ys < 0 or ys >= img.shape[0]:
            if (xs < safety_margin or xe >= img.shape[1] - safety_margin or
                    ys < safety_margin or ys >= img.shape[0] - safety_margin):
                continue

            # create an initial center orientation
            center_rand = 0
            
            # Note that the below will give zero when
            # `fAugmentCenterRandStrength == 0`. This effectively disables the
            # random perturbation.
            center_rand = ((np.random.rand() * 2.0 - 1.0) *
                        fAugmentCenterRandStrength)
            
            # create an array of rotations to used
            rot_diff_list = np.arange(nAugmentedRotations).astype(float)
            # compute the step between subsequent rotations
            rot_step = 2.0 * fAugmentRange / float(nAugmentedRotations)
            rot_diff_list *= rot_step

            for rot_diff in rot_diff_list:

                # the rotation to be applied for this patch
                crot_deg = rot_diff + center_rand
                crot_rad = crot_deg * np.pi / 180.0
                
                cur_patch = self._crop_patch(
                    img, cx, cy, crot_deg,
                    max_hw / (float(nPatchSizeAug - 1) * 0.5),
                    nPatchSizeAug)
                if len(cur_patch.shape) == 2:
                    #                pdb.set_trace()
                    cur_patch = cur_patch[..., np.newaxis]

                x[idxKeep] = cur_patch.transpose(2, 0, 1)
                # update target value and id
                y[idxKeep] = y_in[idx]
                ID[idxKeep] = ID_in[idx]
                # add crot (in radians), note that we are between -2pi and 0 for
                # compatiblity
                # angle[idxKeep] = crot_rad
                angle[idxKeep] = ((angle_in[idx] + crot_rad) % (2.0 * np.pi) -
                                (2.0 * np.pi))

                # Store the perturbation (center of the patch is 0,0,0)
                new_perturb_xyz = perturb_xyz[idx].copy()
                xx, yy, zz = new_perturb_xyz
                rx = np.cos(crot_rad) * xx - np.sin(crot_rad) * yy
                ry = np.sin(crot_rad) * xx + np.cos(crot_rad) * yy
                rz = zz
                pos[idxKeep] = np.asarray([rx, ry, rz])

                # store the original pixel coordinates
                new_kp_in = kp_in[idx].copy()
                new_kp_in[3] = ((new_kp_in[3] + crot_rad) % (2.0 * np.pi) -
                                (2.0 * np.pi))
                coords[idxKeep] = new_kp_in

                idxKeep += 1

        # Delete unassigned
        x = x[:idxKeep]
        y = y[:idxKeep]
        ID = ID[:idxKeep]
        pos = pos[:idxKeep]
        angle = angle[:idxKeep]
        coords = coords[:idxKeep]

        patch = x.astype('uint8')
        xyz = pos
        angle = angle.reshape(-1, 1)
        kps = coords

        return patch, xyz, angle, coords


    def _make_theta(self, xyz, cs=None, scale=None, rr=0.5):
        """Make the theta to be used for the spatial transformer

        If cs is None, simply just do the translation only.

        """

        # get dx, dy, dz
        dx = xyz[:, 0]
        dy = xyz[:, 1]
        dz = xyz[:, 1]
        # compute the resize from the largest scale image
        reduce_ratio = rr
        dr = (reduce_ratio) * (2.0)**dz
        if cs is None:
            c = np.ones_like(dx)
            s = np.zeros_like(dx)
        else:
            c = cs[:, 0]
            s = cs[:, 1]
        theta = np.stack(
            [dr * c, -dr * s, dx,
            dr * s,  dr * c, dy], axis=1)

        return torch.from_numpy(theta).view(-1, 2, 3).float()

    def _meshgrid(self, H, W):
        '''
        Given H, W, outputs 3 x HW tensor
        '''
        xt = (torch.ones((H, 1)) * torch.linspace(-1, 1, W).unsqueeze(0))
        yt = (torch.linspace(-1, 1, H).unsqueeze(1) * torch.ones((1, W)))

        xtf = xt.view(1, -1)
        ytf = yt.view(1, -1)

        os = torch.ones_like(xtf)
        return torch.cat((xtf, ytf, os))

    def _transformer(self, patch, thetas, output_size):
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
        N, C, PH, PW = patch.shape
        OH = output_size
        OW = output_size
        hoW = PW // 2
        hoH = PH // 2

        max_y = (PH - 1)
        max_x = (PW - 1)

        # 3 x (HW) => B x 3 x (HW)
        grid = self._meshgrid(OH, OW).unsqueeze(0).repeat(N, 1, 1)
        

        T_g = thetas.matmul(grid)
        x = T_g[:, 0:1, :]
        y = T_g[:, 1:2, :]
        
        # create grid cell coords
        x = (x + 1.0) * (PW - 1.0) / 2.0
        y = (y + 1.0) * (PH - 1.0) / 2.0

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

        base = torch.tensor(list(range(N))).unsqueeze(1).repeat(1, OW * OH).view(-1) * PW * PH
        base_y0 = base + y0 * PW
        base_y1 = base + y1 * PW

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        im_flat = patch.permute(0, 2, 3, 1).view(-1, C)
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
        output = output.view(N, OH, OW, C).permute(0, 3, 1, 2)
        return output

    def _XYZS2kpList(self, XYZS):
        KP_LIST_LEN = 13
        IDX_X, IDX_Y, IDX_SIZE, IDX_ANGLE, IDX_RESPONSE, IDX_OCTAVE = (0, 1, 2, 3, 4, 5)
        IDX_a, IDX_b, IDX_c = (6, 7, 8)

        kp_list = [None] * XYZS.shape[0]
        for idxKp in xrange(XYZS.shape[0]):

            kp = np.zeros((KP_LIST_LEN, ))
            kp[IDX_X] = XYZS[idxKp, 0]
            kp[IDX_Y] = XYZS[idxKp, 1]
            kp[IDX_SIZE] = XYZS[idxKp, 2]
            kp[IDX_ANGLE] = 0
            kp[IDX_RESPONSE] = XYZS[idxKp, 3]

            # computing the octave should be dealt with caution. We compute in the
            # SIFT way. The SIFT code of openCV computes scale in the following
            # way.
            # >>> scale = 1.6 * 2**((layer+xi) / 3) * 2**octave
            # where octave is packed by octave = layer << 8 + octv
            layer_n_octv = np.log2(kp[IDX_SIZE] / 1.6)
            layer_n_octv = max(0, layer_n_octv)  # TODO: FOR NOW LET'S JUST DO THIS
            octv = int(np.floor(layer_n_octv))
            layer_n_xi = (layer_n_octv - np.floor(layer_n_octv)) * 3.0
            layer = int(np.floor(layer_n_xi))
            xi = layer_n_xi - np.floor(layer_n_xi)
            # make sure we have the layer correctly by checking that it is smaller
            # than 3
            assert layer < 3
            # pack octave in the same way as openCV
            # kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16);
            # also remember
            # octave = octave < 128 ? octave : (-128 | octave);
            # which means
            # kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255)
            # so if octave is negative & 255 will give otcv >= 128 later...
            octv = octv & 255
            octave = octv + (layer << 8) + (int(np.round((xi + 0.5) * 255.)) << 16)
            kp[IDX_OCTAVE] = octave

            # Compute a,b,c for vgg affine
            kp[IDX_a] = 1. / (kp[IDX_SIZE]**2)
            kp[IDX_b] = 0.
            kp[IDX_c] = 1. / (kp[IDX_SIZE]**2)
            # Compute A0, A1, A2, A3 and update
            kp = self._update_affine(kp)
            # kp[IDX_CLASSID] = 0

            kp_list[idxKp] = kp

        return kp_list