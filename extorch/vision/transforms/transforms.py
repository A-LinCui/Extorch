from typing import Optional, Dict, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms

from . import functional as extF


class AdaptiveRandomCrop(nn.Module):
    r"""
    Adaptively randomly crop images with uncertain sizes for a certain size.

    Args:
        cropped_size (Union[int, Tuple[int, int]]): The Image size to be cropped out.
    """
    def __init__(self, cropped_size: Union[int, Tuple[int, int]]) -> None:
        super(AdaptiveRandomCrop, self).__init__()
        self.cropped_size = cropped_size if isinstance(cropped_size, tuple) \
                else (cropped_size, cropped_size)
    
    def forward(self, img: Tensor) -> Tensor:
        r"""
        Args:
            img (Tensor): The image to be cropped.

        Returns:
            img (Tensor): The cropped image. For example, if the image has size [H, W] and the 
                          cropped size if [h, w], size of output will be [H - h, W - w].
        """
        width, height = F._get_image_size(img)
        size = (height - self.cropped_size[1], width - self.cropped_size[0])
        i, j, h, w = transforms.RandomCrop.get_params(img, size)
        return F.crop(img, i, j, h, w)


class AdaptiveCenterCrop(nn.Module):
    r"""
    Adaptively center-crop images with uncertain sizes for a certain size.

    Args:
        cropped_size (Union[int, Tuple[int, int]]): The Image size to be cropped out.
    """
    def __init__(self, cropped_size: Union[int, Tuple[int, int]]) -> None:
        super(AdaptiveCenterCrop, self).__init__()
        self.cropped_size = cropped_size if isinstance(cropped_size, tuple) \
                else (cropped_size, cropped_size)
    
    def forward(self, img: Tensor) -> Tensor:
        r"""
        Args:
            img (Tensor): The image to be cropped.

        Returns:
            img (Tensor): The cropped image. For example, if the image has size [H, W] and the 
                          cropped size if [h, w], size of output will be [H - h, W - w].
        """
        width, height = F._get_image_size(img)
        size = (height - self.cropped_size[1], width - self.cropped_size[0])
        return F.center_crop(img, size)


class Cutout(nn.Module):
    r"""
    Cutout: Randomly mask out one or more patches from an image (`Link`_).

    Args:
        length (int): The length (in pixels) of each square patch.
        image (Tensor): Image of size (C, H, W).
        n_holes (int): Number of patches to cut out of each image. Default: 1.

    Examples::
        >>> image = torch.ones((3, 32, 32))
        >>> Cutout_transform = Cutout(16, 1)
        >>> image = Cutout_transform(image)  # Shape: [3, 32, 32]

    .. _Link:
        https://arxiv.org/abs/1708.04552
    """
    
    def __init__(self, length: int, n_holes: int = 1) -> None:
        super(Cutout, self).__init__()
        self.length = length
        self.n_holes = n_holes

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (Tensor): Image of size (C, H, W).

        Returns:
            img (Tensor): Image with n_holes of dimension length x length cut out of it.
        """
        return extF.cutout(img, self.length, self.n_holes)


class DetectionCompose(transforms.Compose):
    r"""
    Tranform compose for detection.
    """
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class DetectionToTensor(nn.Module):
    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
            ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        return image, target
