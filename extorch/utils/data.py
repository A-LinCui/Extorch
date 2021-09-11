import numpy as np
import torch
from torch import Tensor

from extorch.utils import expect, InvalidValueException


def mix_data(self, inputs: Tensor, targets: Tensor, alpha: float = 1.0):
    r""" 
    Mixup data for training neural networks on convex combinations of pairs of examples and their labels (`Link`_).

    Args:
        inputs (Tensor): Input examples.
        targets (Tensor): Labels of input examples.
        alpha (float): Parameter of the beta distribution. Default: 1.0.
    
    Returns:
        mixed_inputs (Tensor): Input examples after mixup.
        mixed_targets (Tensor): Labels of mixed inputs.
        _lambda (float): Parameter sampled from the beta distribution.

    .. _Link:
        https://arxiv.org/abs/1710.09412
    """
    expect(alpha > 0. and alpha < 1.0, 
        "alpha () should be in (0., 1.)".format(alpha), InvalidValueException)
    _lambda = np.random.beta(alpha, alpha) if alpha > 0. else 1.
    index = torch.randperm(len(targets)).to(inputs.device)
    mixed_inputs = _lambda * inputs + (1 - _lambda) * inputs[index, :]
    mixed_targets = targets[index]
    return mixed_inputs, mixed_targets, _lambda
