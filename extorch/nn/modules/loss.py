import torch
import torch.nn as nn
from torch import Tensor

from extorch.utils import expect
from extorch.nn.functional import dec_soft_assignment, mix_data


__all__ = [
        "HintonKDLoss",
        "CrossEntropyLabelSmooth",
        "CrossEntropyMixupLoss",
        "DECLoss"
]


class HintonKDLoss(nn.KLDivLoss):
    r"""
    Knowledge distillation loss proposed by Hinton (`Link`_).

    $L = (1 - \alpha) * L_{CE}(P_s, Y_{gt}) + \alpha * T^2 * L_{CE}(P_s, P_t)$

    Args:
        T (float): Temperature parameter (>= 1.) used to smooth the softmax output.
        alpha (float): Trade-off coefficient between distillation and origin loss.
        reduction (str): Specifies the reduction to apply to the output. Default: "mean".
        kwargs: Other configurations for nn.CrossEntropyLoss.

    Examples::
        >>> criterion = HintonKDLoss(T = 4., alpha = 0.9)
        >>> s_output = torch.randn((5, 10))
        >>> t_output = torch.randn((5, 10))
        >>> target = torch.ones(5, dtype = torch.long)
        >>> loss = criterion(s_output, t_output, target)

    .. _Link:
        https://arxiv.org/abs/1503.02531
    """
    def __init__(self, T: float, alpha: float, reduction: str = "mean", **kwargs) -> None:
        kl_reduction = "batchmean" if reduction == "mean" else reduction
        super(HintonKDLoss, self).__init__(reduction = kl_reduction)
        assert T >= 1., "Parameter T should not be smaller than 1."
        self.T = T
        assert 0. <= alpha <= 1., "Parameter alpha should be in [0, 1]."
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(reduction = reduction, **kwargs)

    def forward(self, s_output: Tensor, t_output: Tensor, target: Tensor) -> Tensor:
        r"""
        Args:
            s_output (Tensor): Student network output.
            t_output (Tensor): Teacher network output.
            target (Tensor): Hard label of the input.

        Returns:
            Tensor: The calculated loss.
        """
        if self.alpha == 0.:
            return self.ce_loss(s_output, target)

        soft_loss = super(HintonKDLoss, self).forward(
                torch.log_softmax(s_output / self.T, dim = 1),
                torch.softmax(t_output / self.T, dim = 1)
        )

        if self.alpha == 1.:
            return self.T ** 2 * soft_loss

        hard_loss = self.ce_loss(s_output, target)
        return (1 - self.alpha) * hard_loss + self.alpha * self.T ** 2 * soft_loss
    

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

    def forward(self, input: Tensor, target: Tensor, net: nn.Module) -> Tensor:
        r"""
        Args:
            input (Tensor): Input examples.
            target (Tensor): Label of the input examples.
            net (nn.Module): Network to calculate the loss.

        Returns:
            loss (Tensor): The loss.
        """
        mixed_input, mixed_target, _lambda = mix_data(input, target, self.alpha)
        mixed_output = net(mixed_input)
        loss = _lambda * self._criterion(mixed_output, target) + \
                (1 - _lambda) * self._criterion(mixed_output, mixed_target)
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
