from typing import List, Optional
import random

import numpy as np
import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
import torchvision.transforms as transforms

from extorch.nn.utils import WrapperModel
from extorch.adversarial.base import BaseAdversary


__all__ = [
    "GradientBasedAdversary", 
    "PGDAdversary", 
    "CustomizedPGDAdversary",
    "FGSMAdversary", 
    "MDIFGSMAdversary", 
    "MIFGSMAdversary", 
    "IFGSMAdversary", 
    "DiversityLayer"
]


class GradientBasedAdversary(BaseAdversary):
    r"""
    Gradient-based adversary.

    Args:
        criterion (Optional[_Loss]): Criterion to calculate the loss. Default: ``nn.CrossEntropyLoss``.
        use_eval_mode (bool): Whether use eval mode of the network while running attack. Default: ``False``.
    """

    def __init__(self, criterion: Optional[_Loss] = nn.CrossEntropyLoss(), use_eval_mode: bool = False) -> None:
        super(GradientBasedAdversary, self).__init__(use_eval_mode)
        self.criterion = criterion


class PGDAdversary(GradientBasedAdversary):
    r"""
    Project Gradient Descent (PGD, `Link`_) adversarial adversary.
        
    Args:
        epsilon (float): Maximum distortion of adversarial example compared to origin input.
        n_step (int): Number of attack iterations.
        step_size (float): Step size for each attack iteration.
        rand_init (bool): Whether add random perturbation to origin input before formal attack.
        mean (List[float]): Sequence of means for each channel while normalizing the origin input.
        std (List[float]): Sequence of standard deviations for each channel while normalizing the origin input.
        criterion (Optional[_Loss]): Criterion to calculate the loss. Default: ``nn.CrossEntropyLoss``.
        use_eval_mode (bool): Whether use eval mode of the network while running attack. Default: ``False``.

    .. _Link:
        https://arxiv.org/abs/1706.06083
    """

    def __init__(self, epsilon: float, n_step: int, step_size: float, 
                 rand_init: bool, mean: List[float], std: List[float], 
                 criterion: Optional[_Loss] = nn.CrossEntropyLoss(),
                 use_eval_mode: bool = False) -> None:
        super(PGDAdversary, self).__init__(criterion, use_eval_mode)
        self.epsilon = epsilon
        self.n_step = n_step
        self.step_size = step_size
        self.rand_init = rand_init
        self.mean = torch.reshape(torch.tensor(mean), (3, 1, 1))
        self.std = torch.reshape(torch.tensor(std), (3, 1, 1))

    def generate_adv(self, net: nn.Module, input: Tensor, target: Tensor, output: Tensor) -> Tensor:
        self.mean = self.mean.to(input.device)
        self.std = self.std.to(input.device)

        wrapper_net = WrapperModel(net, self.mean, self.std)

        input_adv = Variable((input.data.clone() * self.std + self.mean), requires_grad = True)
        input_clone = input_adv.data.clone()

        if self.rand_init:
            eta = input.new(input.size()).uniform_(-self.epsilon, self.epsilon)
            input_adv.data = torch.clamp(input_adv.data + eta, 0., 1.)

        for _ in range(self.n_step):
            output = wrapper_net(input_adv)
            loss = self.criterion(output, Variable(target))
            loss.backward()

            eta = self.step_size * input_adv.grad.data.sign()
            input_adv = Variable(input_adv.data + eta, requires_grad = True)
            
            eta = torch.clamp(
                input_adv.data - input_clone, -self.epsilon, self.epsilon
            )
            input_adv.data = torch.clamp(input_clone + eta, 0., 1.)

        input_adv.data = (input_adv.data - self.mean) / self.std
        return input_adv.data


class CustomizedPGDAdversary(PGDAdversary):
    r"""
    Customized Project Gradient Descent (PGD, `Link`_) adversarial adversary.

    .. _Link:
        https://arxiv.org/abs/2109.01983
    """
    def generate_adv(self, net: nn.Module, input: Tensor, target: Tensor, output: Tensor, epsilon_c: float) -> Tensor:
        r"""
        Args:
            epsilon_c (float): 
        """
        self.mean = self.mean.to(input.device)
        self.std = self.std.to(input.device)

        wrapper_net = WrapperModel(net, self.mean, self.std)

        input_adv = Variable((input.data.clone() * self.std + self.mean), requires_grad = True)
        input_clone = input_adv.data.clone()

        if self.rand_init:
            eta = input.new(input.size()).uniform_(-self.epsilon, self.epsilon)
            input_adv.data = torch.clamp(input_adv.data + eta, 0., 1.)

        for step in range(1, self.n_step + 1):
            output = wrapper_net(input_adv)
            loss = self.criterion(output, Variable(target))
            loss.backward(retain_graph = True)

            gradient = input_adv.grad
            gradient_abs = torch.abs(gradient)
            
            g_1 = gradient / torch.sum(gradient_abs)
            g_t = 2 / np.pi * torch.atan(gradient / torch.mean(gradient_abs))
            g_ens = g_1 + self.gamma_1 * g_t + self.gamma_2 * gradient.sign()

            eta = epsilon_c / step * g_ens
            eta = torch.clamp(input_adv + eta - input_clone, -self.epsilon, self.epsilon)

            input_adv = torch.clamp(input_clone + eta, 0., 1.)

        input_adv = (input_adv - self.mean) / self.std
        return input_adv.data


class FGSMAdversary(PGDAdversary):
    r"""
    Fast Gradient Sign Method (FGSM, `Link`_) adversarial adversary.
   
    Args:
        epsilon (float): Maximum distortion of adversarial example compared to origin input.
        rand_init (bool): Whether add random perturbation to origin input before formal attack.
        mean (List[float]): Sequence of means for each channel while normalizing the origin input.
        std (List[float]): Sequence of standard deviations for each channel while normalizing the origin input.
        criterion (Optional[_Loss]): Criterion to calculate the loss. Default: ``nn.CrossEntropyLoss``.
        use_eval_mode (bool): Whether use eval mode of the network while running attack. Default: ``False``.

    .. _Link:
        https://arxiv.org/abs/1412.6572
    """
    def __init__(self, epsilon: float, rand_init: bool, mean: List[float], std: List[float], 
                 criterion: Optional[_Loss] = nn.CrossEntropyLoss(),
                 use_eval_mode: bool = False) -> None:
        super(FGSMAdversary, self).__init__(
                epsilon, 1, epsilon, rand_init, mean, std, criterion, use_eval_mode)


class DiversityLayer(nn.Module):
    r"""
    Diversity input layer for M-DI-FGSM ('Link'_) adversarial adversary.

    Args:
        target (int): Target reshape size.
        p (float): Probability to apply the transformation. Default: 0.5.

    .. _Link:
        https://arxiv.org/abs/1904.02884
    """
    def __init__(self, target: int, p: float = 0.5) -> None:
        super(DiversityLayer, self).__init__()
        self.p = p
        self.target = target

    def forward(self, input: Tensor) -> Tensor:
        if np.random.rand() < self.p:
            ori_size = input.size()[-1]
            first_resize = random.randint(ori_size, self.target - 1)
            up_padding_size = random.randint(0, self.target - first_resize)
            left_padding_size = random.randint(0, self.target - first_resize)
            padding_size = [
                left_padding_size, 
                up_padding_size, 
                self.target - first_resize - left_padding_size, 
                self.target - first_resize - up_padding_size
            ]
            input_transforms = transforms.Compose([
                transforms.Resize(first_resize),
                transforms.Pad(padding_size)]
            )
            return input_transforms(input)
        return input


class MDIFGSMAdversary(PGDAdversary):
    r"""
    Momentum Iterative Fast Gradient Sign Method with Input Diversity Method (M-DI-FGSM, 'Link'_).

    Args:
        epsilon (float): Maximum distortion of adversarial example compared to origin input.
        n_step (int): Number of attack iterations.
        momentum (float): Gradient momentum.
        rand_init (bool): Whether add random perturbation to origin input before formal attack.
        mean (List[float]): Sequence of means for each channel while normalizing the origin input.
        std (List[float]): Sequence of standard deviations for each channel while normalizing the origin input.
        diversity_p (float): Probability to apply the input diversity transformation. 
        target_size (int): Randomly resized shape in the diversity input layer.
        step_size (Optional[float]): Step size for each attack iteration. 
                                     If is not specified, ``step_size = epsilon / n_step``. Default: ``None``.
        criterion (Optional[_Loss]): Criterion to calculate the loss. Default: ``nn.CrossEntropyLoss``.
        use_eval_mode (bool): Whether use eval mode of the network while running attack. Default: ``False``.
    
    .. _Link:
        https://arxiv.org/abs/1904.02884
    """
    def __init__(self, epsilon: float, n_step: int, momentum: float, 
                 rand_init: bool, mean: List[float], std: List[float], 
                 diversity_p: float, target_size: int, 
                 step_size: Optional[float] = None,
                 criterion: Optional[_Loss] = nn.CrossEntropyLoss(),
                 use_eval_mode: bool = False) -> None:
        step_size = step_size or epsilon / n_step
        super(MDIFGSMAdversary, self).__init__(
                epsilon, n_step, step_size, rand_init, mean, std, criterion, use_eval_mode)
        self.momentum = momentum
        self.diversity_layer = DiversityLayer(target_size, diversity_p)

    def generate_adv(self, net: nn.Module, input: Tensor, target: Tensor, output: Tensor) -> Tensor:
        self.mean = self.mean.to(input.device)
        self.std = self.std.to(input.device)

        wrapper_net = WrapperModel(net, self.mean, self.std)

        input_adv = Variable((input.data.clone() * self.std + self.mean), requires_grad = True)
        input_clone = input_adv.data.clone()

        g = torch.zeros_like(input_adv, device = input.device)

        if self.rand_init:
            eta = input.new(input.size()).uniform_(-self.epsilon, self.epsilon)
            input_adv.data = torch.clamp(input_adv.data + eta, 0., 1.)

        for _ in range(self.n_step):
            output = wrapper_net(self.diversity_layer(input_adv))
            loss = self.criterion(output, Variable(target))
            loss.backward()

            gradient = input_adv.grad.data
            g = self.momentum * g + gradient / torch.norm(gradient, p = 1)

            eta = self.step_size * g.data.sign()
            input_adv = Variable(input_adv.data + eta, requires_grad = True)
            
            eta = torch.clamp(
                input_adv.data - input_clone, -self.epsilon, self.epsilon
            )
            input_adv.data = torch.clamp(input_clone + eta, 0., 1.)

        input_adv.data = (input_adv.data - self.mean) / self.std
        return input_adv.data


class MIFGSMAdversary(MDIFGSMAdversary):
    r"""
    Momentum Iterative Fast Gradient Sign Method (MI-FGSM, `Link`_).

    Args:
        epsilon (float): Maximum distortion of adversarial example compared to origin input.
        n_step (int): Number of attack iterations.
        momentum (float): Gradient momentum.
        rand_init (bool): Whether add random perturbation to origin input before formal attack.
        mean (List[float]): Sequence of means for each channel while normalizing the origin input.
        std (List[float]): Sequence of standard deviations for each channel while normalizing the origin input.
        step_size (Optional[float]): Step size for each attack iteration. 
                                     If is not specified, ``step_size = epsilon / n_step``. Default: ``None``.
        criterion (Optional[_Loss]): Criterion to calculate the loss. Default: ``nn.CrossEntropyLoss``.
        use_eval_mode (bool): Whether use eval mode of the network while running attack. Default: ``False``.
    
    .. _Link:
        https://arxiv.org/abs/1710.06081v3
    """
    def __init__(self, epsilon: float, n_step: int, momentum: float, 
                 rand_init: bool, mean: List[float], std: List[float], 
                 step_size: Optional[float] = None,
                 criterion: Optional[_Loss] = nn.CrossEntropyLoss(),
                 use_eval_mode: bool = False) -> None:
        # do not apply the diversity input transformation
        diversity_p = 0.
        target_size = 0
        super(MIFGSMAdversary, self).__init__(epsilon, n_step, momentum, rand_init, 
                mean, std, diversity_p, target_size, step_size, criterion, use_eval_mode)


class IFGSMAdversary(MIFGSMAdversary):
    r"""
    Iterative Fast Gradient Sign Method (I-FGSM, `Link`_).

    Args:
        epsilon (float): Maximum distortion of adversarial example compared to origin input.
        n_step (int): Number of attack iterations.
        rand_init (bool): Whether add random perturbation to origin input before formal attack.
        mean (List[float]): Sequence of means for each channel while normalizing the origin input.
        std (List[float]): Sequence of standard deviations for each channel while normalizing the origin input.
        step_size (Optional[float]): Step size for each attack iteration. 
                                     If is not specified, ``step_size = epsilon / n_step``. Default: ``None``.
        criterion (Optional[_Loss]): Criterion to calculate the loss. Default: ``nn.CrossEntropyLoss``.
        use_eval_mode (bool): Whether use eval mode of the network while running attack. Default: ``False``.
    
    .. _Link:
        https://arxiv.org/abs/1607.02533
    """
    def __init__(self, epsilon: float, n_step: int, rand_init: bool, 
                 mean: List[float], std: List[float], step_size: Optional[float] = None,
                 criterion: Optional[_Loss] = nn.CrossEntropyLoss(),
                 use_eval_mode: bool = False) -> None:
        momentum = 0.
        super(IFGSMAdversary, self).__init__(epsilon, n_step, momentum, rand_init, 
                mean, std, step_size, criterion, use_eval_mode)
