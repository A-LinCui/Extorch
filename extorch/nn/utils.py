import six
from contextlib import contextmanager
from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn as nn


def _substitute_params(module: nn.Module, params: OrderedDict, prefix: str = "") -> None:
    r"""
    Replace the parameters with the given ones.

    Args:
        module (nn.Module): The targeted module.
        params (OrderedDict): The given parameters.
        prefix (str): Only parameters with this prefix will be replaced.
    """
    prefix = (prefix + ".") if prefix else ""
    for n in module._parameters:
        if prefix + n in params:
            module._parameters[n] = params[prefix + n]


@contextmanager
def use_params(module: nn.Module, params: OrderedDict) -> None:
    r"""
    Replace the parameters in the module with the given ones.
    And then recover the old parameters.

    Args:
        module (nn.Module): The targeted module.
        params (OrderedDict): The given parameters.

    Examples:
        >>> m = nn.Conv2d(1, 10, 3)
        >>> params = m.state_dict()
        >>> for p in params.values():
        >>>     p.data = torch.zeros_like(p.data)
        >>> input = torch.ones((2, 1, 10))
        >>> with use_params(m, params):
        >>>     output = m(input)
    """
    backup_params = dict(module.named_parameters())
    for mod_prefix, mod in module.named_modules():
        _substitute_params(mod, params, prefix = mod_prefix)
    yield
    for mod_prefix, mod in module.named_modules():
        _substitute_params(mod, backup_params, prefix = mod_prefix)


def net_device(module: nn.Module) -> torch.device:
    r"""
    Get current device of the network, assuming all weights of the network are on the same device.

    Args:
        module (nn.Module): The network.

    Returns:
        torch.device: The device.

    Examples::
      >>> module = nn.Conv2d(3, 3, 3)
      >>> device = net_device(module) # "cpu"
    """
    if isinstance(module, nn.DataParallel):
        module = module.module
    for submodule in module.children():
        parameters = submodule._parameters
        if "weight" in parameters:
            return parameters["weight"].device
    parameters = module._parameters
    assert "weight" in parameters
    return parameters["weight"].device


class WrapperModel(nn.Module):
    r"""
    A wrapper model for computer vision tasks.
    Normalize the input before feed-forward.

    Args:
        module (nn.Module): A network with input range [0., 1.].
        mean (Tensor): The mean value used for input transforms.
        std (Tensor): The standard value used for input transforms.
    """
    def __init__(self, module: nn.Module, mean: Tensor, std: Tensor) -> None:
        super(WrapperModel, self).__init__()
        self.module = module
        self.mean = mean
        self.std = std

    def forward(self, input: Tensor) -> Tensor:
        return self.module((input - self.mean) / self.std)
