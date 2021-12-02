import torch
import torch.nn as nn
from torch import Tensor

from extorch.utils import expect
from extorch.nn.functional import dec_soft_assignment, mix_data


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, epsilon: float) -> None:
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim = 1)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        num_classes = int(input.shape[-1])
        log_probs = self.logsoftmax(input)
        target = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
        target = (1 - self.epsilon) * target + self.epsilon / num_classes
        loss = - (target * log_probs).mean(0).sum()
        return loss


class CrossEntropyMixupLoss(nn.Module):
    r"""
    CrossEntropyLoss with mixup technique.

    Args:
        alpha (float): Parameter of the beta distribution. Default: 1.0.
        kwargs: Other arguments of torch.nn.CrossEntropyLoss (`Link`_).

    .. _Link:
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#CrossEntropyLoss
    """

    def __init__(self, alpha: float = 1., **kwargs) -> None:
        super(CrossEntropyMixupLoss, self).__init__()
        self.alpha = alpha
        self._criterion = nn.CrossEntropyLoss(**kwargs)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mixed_input, mixed_target, _lambda = mix_data(input, target, self.alpha)
        loss = _lambda * self._criterion(input, target) + \
                (1 - _lambda) * self._criterion(mixed_input, mixed_target)
        return loss


class DECLoss(nn.Module):
    r"""
    Loss used by Deep Embedded Clustering (DEC, `Link`_).

    Args:
        alpha (float): The degrees of freedom of the Student’s tdistribution. Default: 1.0.

    Examples::
        >>> criterion = DECLoss(alpha = 1.)
        >>> embeddings = torch.randn((2, 10))
        >>> centers = torch.randn((3, 10))
        >>> loss = criterion(embeddings, centers)

    .. _Link:
        https://arxiv.org/abs/1511.06335
    """
    def __init__(self, alpha: float = 1.0, **kwargs) -> None:
        super(DECLoss, self).__init__()
        self.alpha = alpha
        self._criterion = nn.KLDivLoss(**kwargs)

    def forward(self, input: Tensor, centers: Tensor) -> Tensor:
        q = dec_soft_assignment(input, centers, self.alpha)
        p = self.target_distribution(q).detach()
        return self._criterion(q.log(), p)

    @staticmethod
    def target_distribution(input: Tensor) -> Tensor:
        weight = (input ** 2) / torch.sum(input, 0)
        return (weight.t() / torch.sum(weight, 1)).t()
