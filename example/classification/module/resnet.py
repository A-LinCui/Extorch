from typing import Union, List

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from extorch.nn import ResNetBasicBlock, ResNetBottleneckBlock, ConvBNReLU


class CIFARResNet(nn.Module):
    def __init__(self, block: Union[ResNetBasicBlock, ResNetBottleneckBlock], 
            num_blocks: List[int], num_classes: int = 10) -> None:
        super(CIFARResNet, self).__init__()
        self.in_planes = 64
        self.op = ConvBNReLU(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride = 2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: Union[ResNetBottleneckBlock, ResNetBasicBlock], 
            planes: int, num_blocks: int, stride: int) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            out_planes = planes * block.expansion
            layers.append(block(self.in_planes, out_planes, stride))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.op(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def CIFARResNet18(num_classes: int) -> CIFARResNet:
    return CIFARResNet(ResNetBasicBlock, [2, 2, 2, 2], num_classes)


def CIFARResNet34(num_classes: int) -> CIFARResNet:
    return CIFARResNet(ResNetBasicBlock, [3, 4, 6, 3], num_classes)


def CIFARResNet50(num_classes: int) -> CIFARResNet:
    return CIFARResNet(ResNetBottleneckBlock, [3, 4, 6, 3], num_classes)


def CIFARResNet101(num_classes: int) -> CIFARResNet:
    return CIFARResNet(ResNetBottleneckBlock, [3, 4, 23, 3], num_classes)


def CIFARResNet152(num_classes: int) -> CIFARResNet:
    return CIFARResNet(ResNetBottleneckBlock, [3, 8, 36, 3], num_classes)
