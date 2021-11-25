from typing import List, Optional

import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

from extorch.nn.utils import WrapperModel
from extorch.adversarial.base import BaseAdversary


class PGDAdversary(BaseAdversary):
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
        super(PGDAdversary, self).__init__(use_eval_mode)
        self.epsilon = epsilon
        self.n_step = n_step
        self.step_size = step_size
        self.rand_init = rand_init
        self.criterion = criterion
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
