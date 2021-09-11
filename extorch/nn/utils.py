import six
from contextlib import contextmanager
from collections import OrderedDict

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
