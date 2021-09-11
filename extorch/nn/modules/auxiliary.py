from torch import Tensor
import torch.nn as nn


class AuxiliaryHead(nn.Module):
    r"""
    Auxiliary head for the classification task on CIFAR datasets.

    Args:
        in_channels (int): Number of channels in the input feature.
        num_classes (int): Number of classes.

    Examples::
        >>> import torch
        >>> input = torch.randn((10, 3, 32, 32))
        >>> module = AuxiliaryHead(3, 10)
        >>> output = module(input)
    """
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super(AuxiliaryHead, self).__init__()

        self.features = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.AvgPool2d(5, stride = 3, padding = 0, count_include_pad = False),
            nn.Conv2d(in_channels, 128, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 768, 2, bias = False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace = True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input: Tensor) -> Tensor: #pylint: disable=arguments-differ
        output = self.features(input)
        output = self.classifier(output.view(output.size(0), -1))
        return output


class AuxiliaryHeadImageNet(nn.Module):
    r"""
    Auxiliary head for the classification task on the ImageNet dataset.

    Args:
        in_channels (int): Number of channels in the input feature.
        num_classes (int): Number of classes.

    Examples::
        >>> import torch
        >>> input = torch.randn(10, 5, 32, 32)
        >>> module = AuxiliaryHeadImageNet(5, 10)
        >>> output = module(input)
    """
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super(AuxiliaryHeadImageNet, self).__init__()

        self.features = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.AvgPool2d(5, stride = 2, padding = 0, count_include_pad = False),
            nn.Conv2d(in_channels, 128, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 768, 2, bias = False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace = True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input: Tensor) -> Tensor:
        output = self.features(input)
        output = self.classifier(output.view(output.size(0), -1))
        return output
