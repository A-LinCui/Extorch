import random
from typing import Union, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.transforms import functional as F

from .functional import _totuple


__all__ = [
    "SegCompose",
    "SegRandomHorizontalFlip",
    "SegNormalize",
    "SegCenterCrop",
    "SegRandomCrop",
    "SegResize",
    "SegRandomResize",
    "SegPILToTensor",
    "SegConvertImageDtype"
]


class SegCompose(T.Compose):
    r"""
    Transform compose for segmentation.
    """
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class SegRandomHorizontalFlip(nn.Module):
    r"""
    Horizontally flip the given image and label randomly with a given probability (`Link`_).
    
    If the image and label are torch Tensor, they are expected to have [..., H, W] shape, 
    where ... means an arbitrary number of leading dimensions.

    Args:
        p (float): probability of the image and label being flipped. Default: 0.5.

    .. _Link:
        https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomHorizontalFlip
    """
    def __init__(self, p: float = 0.5) -> None:
        super(SegRandomHorizontalFlip, self).__init__()
        self.p = p

    def forward(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class SegNormalize(nn.Module):
    r"""
    Normalization for segmentation, where the normalization is only applied on the input image.
    
    Args:
        mean (List[float]): List of means for each channel.
        std (List[float]): List of standard deviations for each channel.
    """
    def __init__(self, mean: List[float], std: List[float]) -> None:
        super(SegNormalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, image, target):
        image = F.normalize(image, mean = self.mean, std = self.std)
        return image, target


class SegCenterCrop(nn.Module):
    r"""
    Crops the given image at the center.

    Args:
        size (Union[int, List[int]]): Height and width of the crop box.
    """
    def __init__(self, size: Union[int, List[int]]):
        super(SegCenterCrop, self).__init__()
        self.size = size

    def forward(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class SegRandomCrop(nn.Module):
    r"""
    Random cropping for segmentation. 
    The Cropping is applied on the image and target at the same time.

    Args:
        size (Union[int, Tuple[int, int]]): Desired output size of the crop.
    """
    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        super(SegRandomCrop, self).__init__()
        self.size = _totuple(size)

    def forward(self, image, target):
        width, height = F.get_image_size(image)
        
        if width < self.size[1]:
            padding = [self.size[1] - width, 0]
            image = F.pad(image, padding, 0, "constant")
            target = F.pad(target, padding, 255, "constant")
        
        if height < self.size[0]:
            padding = [0, self.size[0] - height]
            image = F.pad(image, padding, 0, "constant")
            target = F.pad(target, padding, 255, "constant")

        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class SegResize(nn.Module):
    r"""
    Resize for segmentation.

    Args:
        size (Union[int, Tuple[int, int]]): Desired output size.
    """
    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        super(SegResize, self).__init__()
        self.size = _totuple(size)

    def forward(self, image, target):
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation = T.InterpolationMode.NEAREST)
        return image, target


class SegRandomResize(nn.Module):
    r"""
    Random resize for segmentation.

    Args:
        min_size (Union[int, Tuple[int, int]]): Desired minimum output size.
        max_size (Optional[Union[int, Tuple[int, int]]]): Desired maximum output size. Default: `None`.
    """
    def __init__(self, min_size: Union[int, Tuple[int, int]], 
            max_size: Optional[Union[int, Tuple[int, int]]] = None) -> None:
        super(SegRandomResize, self).__init__()
        self.min_size = _totuple(min_size)
        self.max_size = _totuple(max_size) if max_size else self.min_size

    def forward(self, image, target):
        size = (
            random.randint(self.min_size[0], self.max_size[0]),
            random.randint(self.min_size[1], self.max_size[1])
        )
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation = T.InterpolationMode.NEAREST)
        return image, target


class SegPILToTensor(nn.Module):
    r"""
    PIL to Tensor for segmentation.
    """
    def __init__(self):
        super(SegPILToTensor, self).__init__()

    def forward(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype = torch.int64)
        return image, target


class SegConvertImageDtype(nn.Module):
    r"""
    Convert image dtype for segmentation.
    """
    def __init__(self, dtype) -> None:
        super(SegConvertImageDtype, self).__init__()
        self.dtype = dtype

    def forward(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target
