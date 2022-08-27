from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from extorch.nn.modules.operation import ConvBNReLU, Identity, ConvBN


__all__ = [
        "SimpleGate",
        "ResNetBasicBlock",
        "ResNetBottleneckBlock"
]


class SimpleGate(nn.Module):
    r"""
    Simple gate proposed in Simple Baselines for Image Restoration (`Link`_).

    .. _Link:
        https://arxiv.org/abs/2204.0467
    """
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim = 1)
        return x1 * x2


class ResNetBasicBlock(nn.Module):
    r"""
    ResNet basic block (`Link`_).

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        stride (int): Stride of the convolution.
        kernel_size (int): Size of the convolving kernel. Default: 3.
        affine (bool): A boolean value that when set to ``True``, the batch-normalization layer
                       has learnable affine parameters. Default: ``True``.

    Examples::
        >>> m = ResNetBasicBlock(3, 10, 2, 3, True)
        >>> input = torch.randn(3, 3, 32, 32)
        >>> output = m(input)

    .. _Link:
        https://arxiv.org/abs/1512.03385
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int, kernel_size: int = 3, affine: bool = True) -> None:
        super(ResNetBasicBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.op1 = ConvBNReLU(in_channels, out_channels, kernel_size, stride, padding, bias = False, affine = affine)
        self.op2 = ConvBN(out_channels, out_channels, kernel_size, 1, padding, bias = False, affine = affine)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBN(in_channels, out_channels, 1, stride, bias = False, affine = affine)
        else:
            self.shortcut = Identity()

    def forward(self, input: Tensor) -> Tensor:
        output = self.op1(input)
        output = self.op2(output)
        output = F.relu(output + self.shortcut(input))
        return output


class ResNetBottleneckBlock(nn.Module):
    r"""
    ResNet bottleneck block (`Link`_).

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        stride (int): Stride of the convolution.
        affine (bool): A boolean value that when set to ``True``, the batch-normalization layer
                       has learnable affine parameters. Default: ``True``.

    Examples::
        >>> m = ResNetBottleneckBlock(10, 10, 2, True)
        >>> input = torch.randn(2, 10, 32, 32)
        >>> output = m(input)

    .. _Link:
        https://arxiv.org/abs/1512.03385
    """
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int, affine: bool = True) -> None:
        super(ResNetBottleneckBlock, self).__init__()
        mid_channels = out_channels // self.expansion
        self.op1 = ConvBNReLU(in_channels, mid_channels, 1, bias = False, affine = affine)
        self.op2 = ConvBNReLU(mid_channels, mid_channels, 3, stride, 1, bias = False, affine = affine)
        self.op3 = ConvBN(mid_channels, out_channels, 1, bias = False, affine = affine)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBN(in_channels, out_channels, 1, stride, bias = False, affine = affine)
        else:
            self.shortcut = Identity()

    def forward(self, input: Tensor) -> Tensor:
        output = self.op1(input)
        output = self.op2(output)
        output = self.op3(output)
        output = F.relu(output + self.shortcut(input))
        return output
