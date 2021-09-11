import torch.nn as nn
from torch import Tensor

from . import functional as extF


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
