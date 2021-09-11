import abc
import random
import copy
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as F


from extorch.utils import expect, InvalidValueException


DATA = Union[Tensor, np.ndarray]
DATA_PAIR = Tuple[DATA, DATA]


class BasePairedTransform(nn.Module):
    r"""
    Base paired transformation.

    For some visual tasks such as image restoration, 
    the parameters of random transformation should be the same on the image and the corresponding label.
    Therefore, we provide this class for constructing paired-transformations which makes the randomness same.
    """
    def __init__(self) -> None:
        super(BasePairedTransform, self).__init__()
    
    @staticmethod
    def check_data(img: DATA, label: DATA) -> None:
        """
        Shapes of the input and its corresponding label must be the same.
        """
        expect(img.shape == label.shape, "Paired data have different shapes", InvalidValueException)

    @abc.abstractmethod
    def forward(self, img: DATA, label: DATA) -> DATA_PAIR:
        """ 
        Transform inputs under the same randomness 
        """


class PairedCompose(transforms.Compose):
    r"""
    Paired compose specially designed for paired transformations.

    Paired tranformations are applied on the image and its corresponding label at the same time.
    Other basic transformations are apllied on the image and its corresponding label respectively.

    Examples::
        >>> transform = PairedCompose([transforms.ToTensor(), PairedRandomHorizontalFlip(p = 0.5)])
        >>> img = np.ones((32, 32, 3))
        >>> label = np.zeros((32, 32, 3))
        >>> img, label = transform(img, label)
    """
    def __call__(self, img: DATA, label: DATA) -> DATA_PAIR:
        """
        Args:
            img (Tensor or np.ndarray): The image to be transformed.
            label (Tensor or np.ndarray): The corresponding label to be transformed.

        Retunes:
            img (Tensor or np.ndarray): The transformed image.
            label (Tensor or np.ndarray): The transformed label.
        """
        for t in self.transforms:
            if isinstance(t, BasePairedTransform):
                img, label = t(img, label)
            else:
                img, label = t(img), t(label)
        return img, label


class PairedRandomIdentity(BasePairedTransform):
    r"""
    Randomly replace the image with its corresponding label.

    Args:
        p (float): probability of the image being replaced. Default: 0.5.
    """
    def __init__(self, p: float) -> None:
        BasePairedTransform.__init__(self)
        self.p = p

    def forward(self, img: DATA, label: DATA) -> DATA_PAIR:
        self.check_data(img, label)
        if random.random() < self.p:
            img = copy.deepcopy(label)
        return img, label


class PairedRandomVerticalFlip(BasePairedTransform, transforms.RandomVerticalFlip):
    r"""
    Vertically flip the given image and label randomly with a given probability (`Link`_).
    If the image and label are torch Tensor, they are expected to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        p (float): probability of the image and label being flipped. Default: 0.5.

    .. _Link:
        https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomVerticalFlip
    """
    def __init__(self, p: float = 0.5) -> None:
        BasePairedTransform.__init__(self)
        transforms.RandomVerticalFlip.__init__(self, p)

    def forward(self, img: DATA, label: DATA) -> DATA_PAIR:
        self.check_data(img, label)
        if torch.rand(1) < self.p:
            return F.vflip(img), F.vflip(label)
        return img, label


class PairedRandomHorizontalFlip(BasePairedTransform, transforms.RandomHorizontalFlip):
    r"""
    Horizontally flip the given image and label randomly with a given probability (`Link`_).
    If the image and label are torch Tensor, they are expected to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        p (float): probability of the image and label being flipped. Default: 0.5.

    .. _Link:
        https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomHorizontalFlip
    """
    def __init__(self, p: float = 0.5) -> None:
        BasePairedTransform.__init__(self)
        transforms.RandomHorizontalFlip.__init__(self, p)

    def forward(self, img: DATA, label: DATA) -> DATA_PAIR:
        self.check_data(img, label)
        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label
