# -*- coding: utf-8 -*-
"""Module for Supervised Detection."""
import torch

class Detector(object):
    """ Detector Abstrat base class. """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        return

    def detect(self, torch_image):
        """Abstract function that performs detection.

        This serves as a base function for inherited classes.

        Args:
            torch_image (list): list of torch image tensors. 
                This can also be N x C x H x W tensor.

        Returns:
            list: list of M x 2 tensors, where M differs for each element
                in a list.
        
        Raises:
            NotImplementedError: Exception raised if used without inheritance.

        """
        raise NotImplementedError

    def extract(self, torch_image):
        """ Abstract function that performs detection and feature description

        Args:
            torch_image (list): list of torch image tensors. 
                This can also be N x C x H x W tensor.

        Returns:
            tuple: tuple of keypoints and descriptions.
                keypoints are list of M x 2 tensors, where M differs for 
                each element in a list.
                descriptions are list of M x C tensors where M differs for
                each element in a list
        
        Raises:
            NotImplementedError: Exception raised if used without inheritance.

        """
        raise NotImplementedError
    
    def compute_loss(self, torch_image, *args):
        raise NotImplementedError