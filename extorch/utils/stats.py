from typing import List, Tuple, Union
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
from thop import profile

from extorch.nn.utils import net_device


class OrderedStats(object):
    def __init__(self) -> None:
        self.stat_meters = None

    def __nonzero__(self) -> bool:
        return self.stat_meters is not None

    __bool__ = __nonzero__

    def update(self, stats, n: int = 1) -> None:
        if self.stat_meters is None:
            self.stat_meters = OrderedDict([(name, AverageMeter()) for name in stats])
        [self.stat_meters[name].update(v, n) for name, v in stats.items()]

    def avgs(self) -> Union[OrderedDict, None]:
        if self.stat_meters is None:
            return None
        return OrderedDict((name, meter.avg) for name, meter in self.stat_meters.items())

    def items(self):
        return self.stat_meters.items() if self.stat_meters is not None else None


class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()

    def is_empty(self) -> bool:
        return self.cnt == 0

    def reset(self) -> None:
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0

    def update(self, val: Union[float, int], n: int = 1) -> None:
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(outputs: Tensor, targets: Tensor, topk: Tuple[int] = (1, )) -> List[Tensor]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class SegConfusionMatrix(object):
    r"""
    Confusion matrix as well as metric calculation for image segmentation.

    Args:
        num_classes (int): The number of classes, including the background.
    """
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.mat = None

    def update(self, output: Tensor, target: Tensor) -> None:
        n = self.num_classes
        output = output.argmax(1).flatten()
        target = target.squeeze(0).to(torch.int64).flatten()
       
        k = (target >= 0) & (target < n)
        inds = n * target[k].to(torch.int64) + output[k]

        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype = torch.int64, device = output.device)
        self.mat += torch.bincount(inds, minlength = n ** 2).reshape(n, n)

    def reset(self) -> None:
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu


def get_params(model: nn.Module, only_trainable: bool = False) -> int:
    r"""
    Get the parameter number of the model.

    Args:
        only_trainable (bool): If only_trainable is true, only trainable parameters will be counted.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not only_trainable)


def cal_flops(model: nn.Module, inputs: Tensor) -> float:
    r"""
    Calculate FLOPs of the given model.

    Args:
        model (nn.Module): The model whose FLOPs is to be calculated.
        inputs (Tensor): Example inputs to the model.

    Returns:
        flops (float): FLOPs of the model.

    Examples::
        >>> import torch
        >>> from extorch.nn import AuxiliaryHead
        >>> module = AuxiliaryHead(3, 10)
        >>> input = torch.randn((10, 3, 32, 32))
        >>> flops = cal_flops(module, input)  # 32.109868
    """
    device = net_device(model)
    flops, _ = profile(model, inputs = [inputs.to(device)], verbose = False)
    flops /= len(inputs)
    return flops
