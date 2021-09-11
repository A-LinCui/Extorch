import torch.nn as nn


def normal_(module: nn.Module, 
            conv_mean: float = 0., conv_std: float = 1., 
            bn_mean: float = 0., bn_std: float = 1., 
            linear_mean: float = 0., linear_std: float = 1.) -> None:
    r"""
    Initialize the module with values drawn from the normal distribution.

    Args:
        module (nn.Module): A pytorch module.
        conv_mean (float): The mean of the normal distribution for convolution.
        conv_std (float): The standard deviation of the normal distribution for convolution.
        bn_mean (float): The mean of the normal distribution for batch-normalization.
        bn_std (float): The standard deviation of the normal distribution for batch-normalization.
        linear_mean (float): The mean of the normal distribution for linear layers.
        bn_std (float): The standard deviation of the normal distribution linear layers.

    Examples::
        >>> import torch.nn as nn
        >>> module = nn.Sequential(
                nn.Conv2d(5, 10, 3),
                nn.BatchNorm2d(10),
                nn.ReLU()
            )
        >>> module.apply(normal_)
    """
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight, conv_mean, conv_std)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight, bn_mean, bn_std)
        nn.init.zeros_(module.bias)
    elif classname.find("Linear") != -1:
        nn.init.normal_(module.weight, linear_mean, linear_std)
        nn.init.zeros_(module.bias)
