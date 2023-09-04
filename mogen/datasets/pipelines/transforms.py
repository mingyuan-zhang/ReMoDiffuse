import math
import random

import mmcv
import numpy as np

from ..builder import PIPELINES
import torch
from typing import Optional, Tuple, Union


@PIPELINES.register_module()
class Crop(object):
    r"""Crop motion sequences.
    
    Args:
        crop_size (int): The size of the cropped motion sequence.
    """
    def __init__(self,
                 crop_size: Optional[Union[int, None]] = None):
        self.crop_size = crop_size
        assert self.crop_size is not None
        
    def __call__(self, results):
        motion = results['motion']
        length = len(motion)
        if length >= self.crop_size:
            idx = random.randint(0, length - self.crop_size)
            motion = motion[idx: idx + self.crop_size]
            results['motion_length'] = self.crop_size
        else:
            padding_length = self.crop_size - length
            D = motion.shape[1:]
            padding_zeros = np.zeros((padding_length, *D), dtype=np.float32)
            motion = np.concatenate([motion, padding_zeros], axis=0)
            results['motion_length'] = length
        assert len(motion) == self.crop_size
        results['motion'] = motion
        results['motion_shape'] = motion.shape
        if length >= self.crop_size:
            results['motion_mask'] = torch.ones(self.crop_size).numpy()
        else:
            results['motion_mask'] = torch.cat(
                (torch.ones(length), torch.zeros(self.crop_size - length))).numpy()
        return results
        
        
    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(crop_size={self.crop_size})'
        return repr_str

@PIPELINES.register_module()
class RandomCrop(object):
    r"""Random crop motion sequences. Each sequence will be padded with zeros to the maximum length.
    
    Args:
        min_size (int or None): The minimum size of the cropped motion sequence (inclusive).
        max_size (int or None): The maximum size of the cropped motion sequence (inclusive).
    """
    def __init__(self,
                 min_size: Optional[Union[int, None]] = None,
                 max_size: Optional[Union[int, None]] = None):
        self.min_size = min_size
        self.max_size = max_size
        assert self.min_size is not None
        assert self.max_size is not None
        
    def __call__(self, results):
        motion = results['motion']
        length = len(motion)
        crop_size = random.randint(self.min_size, self.max_size)
        if length > crop_size:
            idx = random.randint(0, length - crop_size)
            motion = motion[idx: idx + crop_size]
            results['motion_length'] = crop_size
        else:
            results['motion_length'] = length
        padding_length = self.max_size - min(crop_size, length)
        if padding_length > 0:
            D = motion.shape[1:]
            padding_zeros = np.zeros((padding_length, *D), dtype=np.float32)
            motion = np.concatenate([motion, padding_zeros], axis=0)
        results['motion'] = motion
        results['motion_shape'] = motion.shape
        if length >= self.max_size and crop_size == self.max_size:
            results['motion_mask'] = torch.ones(self.max_size).numpy()
        else:
            results['motion_mask'] = torch.cat((
                torch.ones(min(length, crop_size)),
                torch.zeros(self.max_size - min(length, crop_size))), dim=0).numpy()
        assert len(motion) == self.max_size
        return results
        
        
    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(min_size={self.min_size}'
        repr_str += f', max_size={self.max_size})'
        return repr_str
        
@PIPELINES.register_module()
class Normalize(object):
    """Normalize motion sequences.
    
    Args:
        mean_path (str): Path of mean file.
        std_path (str): Path of std file.
    """
    
    def __init__(self, mean_path, std_path, eps=1e-9):
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        self.eps = eps
        
    def __call__(self, results):
        motion = results['motion']
        motion = (motion - self.mean) / (self.std + self.eps)
        results['motion'] = motion
        results['motion_norm_mean'] = self.mean
        results['motion_norm_std'] = self.std
        return results
