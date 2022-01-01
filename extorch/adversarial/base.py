import abc

from torch import nn, Tensor


class BaseAdversary(nn.Module):
    r"""
    Base adversarial adversary.

    Args:
        use_eval_mode (bool): Whether use eval mode while generating adversarial examples. 
                              Default: ``False``.
    """
    def __init__(self, use_eval_mode: bool = False) -> None:
        super(BaseAdversary, self).__init__()
        self.use_eval_mode = use_eval_mode

    def forward(self, net: nn.Module, input: Tensor, target: Tensor, output: Tensor = None) -> Tensor:
        r"""
        Generate adversarial examples.

        Args:
            net (nn.Module): The victim network.
            input (Tensor): Origin input.
            target (Tensor): Label of the input.
            output (Tensor): Origin output. Default: None.

        Returns:
            adv_examples (Tensor): The generated adversarial examples.
        """
        if self.use_eval_mode:
            is_training_stored = net.training
            net.eval()
        else:
            is_training_stored = False

        adv_examples = self.generate_adv(net, input, target, output)

        net.zero_grad()

        if is_training_stored:  # restore the mode
            net.train()
        return adv_examples

    @abc.abstractmethod
    def generate_adv(self, net: nn.Module, input: Tensor, target: Tensor, output: Tensor) -> Tensor:
        r"""
        Adversarial example generation.
        
        Args:
            net (nn.Module): The victim network.
            input (Tensor): Origin input.
            target (Tensor): Label of the input.
            output (Tensor): Origin output.

        Returns:
            adv_examples (Tensor): The generated adversarial examples.
        """
