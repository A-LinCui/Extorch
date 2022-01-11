from typing import Tuple, Union, Optional

import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor


class Identity(nn.Module):
    def __init__(self) -> None:
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input


class BNReLU(nn.Module):
    r"""
    A batch-normalization layer followed by relu.

    Args:
        in_channels (int): Number of channels in the input image.
        affine: A boolean value that when set to ``True``, this module has learnable affine parameters. Default: ``True``.

    Examples::
        >>> m = BNReLU(32, True)
        >>> input = torch.randn(2, 32, 10, 3)
        >>> output = m(input)
    """
    def __init__(self, in_channels: int, affine: bool = True) -> None:
        super(BNReLU, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, affine = affine)

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(self.bn(input))


class ReLUBN(nn.Module):
    r"""
    A batch-normalization layer following relu.

    Args:
        in_channels (int): Number of channels in the input image.
        affine: A boolean value that when set to ``True``, this module has learnable affine parameters. Default: ``True``.

    Examples::
        >>> m = ReLUBN(32, True)
        >>> input = torch.randn(2, 32, 10, 3)
        >>> output = m(input)
    """
    def __init__(self, in_channels: int, affine: bool = True) -> None:
        super(ReLUBN, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, affine = affine)

    def forward(self, input: Tensor) -> Tensor:
        return self.bn(F.relu(input))


class ConvReLU(nn.Module):
    r"""
    A convolution followed by a ReLU.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        padding (int, tuple or str, optional): Padding added to all four sides of e input. Default: 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
        groups (Optional[int]): Number of blocked connections from input channels to output channels. Default: 1.
        bias (Optional[bool]): If ``True``, adds a learnable bias to the output. Default: ``True``.
        inplace (Optional[bool]): ReLU can optionally do the operation in-place. Default: ``False``.
        kwargs: Other configurations of the convolution.

    Examples::
        >>> m = ConvReLU(3, 10, 3, 1)
        >>> input = torch.randn(2, 3, 32, 32)
        >>> output = m(input)
    """
    def __init__(self, in_channels: int, out_channels: int, 
            kernel_size: Union[int, Tuple[int, int]], 
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[str, int, Tuple[int, int]] = 0, 
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: Optional[int] = 1, bias: Optional[bool] = True, 
            inplace: Optional[bool] = False, **kwargs) -> None:
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, 
                padding = padding, dilation = dilation, groups = groups, bias = bias, **kwargs)
        self.relu = nn.ReLU(inplace = inplace)
            
    def forward(self, input: Tensor) -> Tensor:
        output = self.conv(input)
        output = self.relu(output)
        return output


class ConvBN(nn.Module):
    r"""
    A convolution followed by batch-normalization.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        padding (int, tuple or str, optional): Padding added to all four sides of e input. Default: 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``.
        affine (bool): A boolean value that when set to ``True``, the batch-normalization layer
                       has learnable affine parameters. Default: ``True``.
        kwargs: Other configurations of the convolution.

    Examples::
        >>> m = ConvBN(3, 10, 3, 1)
        >>> input = torch.randn(2, 3, 32, 32)
        >>> output = m(input)
    """
    def __init__(self, in_channels: int, out_channels: int, 
            kernel_size: Union[int, Tuple[int, int]], 
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[str, int, Tuple[int, int]] = 0, 
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1, bias: bool = True, 
            affine: bool = True, **kwargs) -> None:
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, 
                padding = padding, dilation = dilation, groups = groups, bias = bias, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, affine = affine)
            
    def forward(self, input: Tensor) -> Tensor:
        output = self.conv(input)
        output = self.bn(output)
        return output


class ConvBNReLU(nn.Module):
    r"""
    A convolution followed by batch-normalization and ReLU.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        padding (int, tuple or str, optional): Padding added to all four sides of e input. Default: 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``.
        affine (bool): A boolean value that when set to ``True``, the batch-normalization layer
                       has learnable affine parameters. Default: ``True``.
        kwargs: Other configurations of the convolution.
    
    Examples::
        >>> m = ConvBNReLU(3, 10, 3, 1)
        >>> input = torch.randn(2, 3, 32, 32)
        >>> output = m(input)
    """
    def __init__(self, in_channels: int, out_channels: int, 
            kernel_size: Union[int, Tuple[int, int]], 
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[str, int, Tuple[int, int]] = 0, 
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1, bias: bool = True, 
            affine: bool = True, **kwargs) -> None:
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, 
                padding = padding, dilation = dilation, groups = groups, bias = bias, **kwargs)
        self.bn_relu = BNReLU(out_channels, affine)
            
    def forward(self, input: Tensor) -> Tensor:
        output = self.conv(input)
        output = self.bn_relu(output)
        return output


class ReLUConvBN(nn.Module):
    r"""
    A ReLU followed by convolution and batch-normalization.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        padding (int, tuple or str, optional): Padding added to all four sides of e input. Default: 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``.
        affine (bool): A boolean value that when set to ``True``, the batch-normalization layer
                       has learnable affine parameters. Default: ``True``.
        kwargs: Other configurations of the convolution.
    
    Examples::
        >>> m = ReLUConvBN(3, 10, 3, 1)
        >>> input = torch.randn(2, 3, 32, 32)
        >>> output = m(input)
    """
    def __init__(self, in_channels: int, out_channels: int, 
            kernel_size: Union[int, Tuple[int, int]], 
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[str, int, Tuple[int, int]] = 0, 
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1, bias: bool = True, 
            affine: bool = True, **kwargs) -> None:
        super(ReLUConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, 
                padding = padding, dilation = dilation, groups = groups, bias = bias, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, affine = affine)
            
    def forward(self, input: Tensor) -> Tensor:
        output = self.conv(F.relu(input))
        output = self.bn(output)
        return output
