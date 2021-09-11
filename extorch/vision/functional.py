import numpy as np
import torch
from torch import Tensor


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
