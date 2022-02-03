from typing import Union, Tuple

import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import functional as F


def cutout(image: Tensor, length: int, n_holes: int = 1) -> Tensor:
    r"""
    Cutout: Randomly mask out one or more patches from an image (`Link`_).

    Args:
        image (Tensor): Image of size (C, H, W).
        length (int): The length (in pixels) of each square patch.
        n_holes (int): Number of patches to cut out of each image. Default: 1.
        
    Returns:
        image (Tensor): Image with n_holes of dimension length x length cut out of it. 

    Examples::
        >>> image = torch.ones((3, 32, 32))
        >>> image = cutout(image, 16, 1)

    .. _Link:
        https://arxiv.org/abs/1708.04552
    """
    h, w = image.shape[1:]
    mask = np.ones((h, w), np.float32)
    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y_1 = np.clip(y - length // 2, 0, h)
        y_2 = np.clip(y + length // 2, 0, h)
        x_1 = np.clip(x - length // 2, 0, w)
        x_2 = np.clip(x + length // 2, 0, w)
        mask[y_1 : y_2, x_1 : x_2] = 0.
    
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(image)
    image *= mask
    
    return image


def _get_image_size(img: Tensor) -> Tuple[int]:
    r"""
    Returns the size of an image as [width, height].

    Args:
        img (PIL Image or Tensor): The image to be checked.

    Returns:
        width (int): Width of the image.
        height (int): Height of the image.
    """
    try:
        width, height = F._get_image_size(img)
    except:
        width, height = F.get_image_size(img)

    return width, height


def _totuple(size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    r"""
    Transform input size from int to tuple.

    Args:
        size (Union[int, Tuple[int, int]]): Size to be checked and transformed.
    """
    if isinstance(size, int):
        size = (size, size)
    return size
