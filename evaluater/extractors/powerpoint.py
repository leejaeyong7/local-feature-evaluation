import cv2
import torch
import torch.nn as nn
import logging
import time
import torch.nn.functional as NF

from os import path
from .backbone.resnet_101 import ResNet101
from .backbone.yolo_v3_lite import YoloV3Lite
from .backbone.yolo_v3 import YoloV3
from .backbone.custom import CustomNet

def CBR(ic, oc, ks=3, stride=1):
    pad = (ks - 1) // 2
    return nn.Sequential(
        nn.Conv2d(ic, oc, ks, padding=pad, stride=stride, bias=False),
        nn.BatchNorm2d(oc, eps=1e-5, momentum=0.9, affine=True, 
                       track_running_stats=True),
        nn.LeakyReLU(negative_slope=0.1)
    )

def CTBR(ic, oc, ks=3, stride=1):
    pad = (ks - 1) // 2
    return nn.Sequential(
        nn.ConvTranspose2d(ic, oc, ks, padding=pad, stride=stride, bias=False),
        nn.BatchNorm2d(oc, eps=1e-5, momentum=0.9, affine=True, 
                       track_running_stats=True),
        nn.LeakyReLU(negative_slope=0.1)
    )

class PowerPointDetector(nn.Module):
    def __init__(self, input_channel, mid_channel, decoder_type='convtransposed'):
        super(PowerPointDetector, self).__init__()

        self.type = decoder_type
        if(decoder_type == 'convtransposed'):
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(input_channel, mid_channel, 4, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(mid_channel, mid_channel, 4, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(mid_channel, 1, 4, stride=2, padding=1),
                nn.Sigmoid()
            )
        else:
            self.decoder = nn.Sequential(
                nn.Conv2d(input_channel, mid_channel, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(mid_channel, 65, 1, 1, 0)
            )

    def forward(self, x):
        if(self.type == 'convtransposed'):
            decoded = self.decoder(x)
            zp = torch.ones_like(decoded)
            return decoded
            # N, _, H, W = zp.shape
            # zp[:, :, :8, :] = 0
            # zp[:, :, :, :8] = 0
            # zp[:, :, -8:, :] = 0
            # zp[:, :, :, -8:] = 0
            # return decoded * zp
        else:
            return self.decoder(x)


class PowerPointNet(nn.Module):
    def __init__(self, config):
        super(PowerPointNet, self).__init__()
        self.config = config
        self.feature_extractor = self._setup_feature_extractor()
        self.detector = self._setup_detector(self.feature_extractor)
        self.descriptor = self._setup_descriptor(self.feature_extractor)

        # setup concat targat hash
        self.device = torch.device('cpu')

    def to(self, device):
        super(PowerPointNet, self).to(device)
        self.device = device
        return self

    def freeze(self, should_freeze):
        self.feature_extractor.freeze(should_freeze)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        """
        f = self.feature_extractor(x)
        detections = self.detector(f)
        descriptions = self.descriptor(f)

        return detections, descriptions

    def _setup_detector(self, feature_extractor):
        """Detector network
        """
        input_channel = feature_extractor.get_num_feature_channels()
        mid_channel = self.config.middle_channel
        return PowerPointDetector(input_channel, mid_channel, self.config.detector_type)


    def _setup_descriptor(self, feature_extractor):
        input_channel = feature_extractor.get_num_feature_channels()
        mid_channel = self.config.middle_channel
        output_channel = self.config.descriptor_size
        descriptor_network = nn.Sequential(
            nn.Conv2d(input_channel, mid_channel, 3, 1, 1),
            nn.Conv2d(mid_channel, output_channel, 1, 1, 0)
        )
        return descriptor_network

    def load_pretrained(self):
        pretrained_dir = self.config.pretrained_dir
        if(self.config.backbone == 'resnet101'):
            return self
        elif(self.config.backbone == 'custom'):
            return self
        elif(self.config.backbone == 'yolo3'):
            pretrained_filename = 'yolo_v3.pth'
        elif(self.config.backbone == 'yolo3-lite'):
            pretrained_filename = 'yolo_v3_lite.pth'

        pretrained_file = path.join(pretrained_dir, pretrained_filename)
        pretrained_weights = torch.load(pretrained_file)
        self.feature_extractor.load_state_dict(pretrained_weights)
        return self

    def _setup_feature_extractor(self):
        """Sets up backbone feature extractor
        """
        if(self.config.backbone == 'resnet101'):
            return ResNet101(self.config)
        elif(self.config.backbone == 'yolo3'):
            return YoloV3(self.config)
        elif(self.config.backbone == 'yolo3-lite'):
            return YoloV3Lite(self.config)
        elif(self.config.backbone == 'custom'):
            return CustomNet(self.config)
        else:
            raise NotImplementedError
