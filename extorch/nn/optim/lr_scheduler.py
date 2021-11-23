from typing import List

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer


class CosineWithRestarts(_LRScheduler):
    r"""
    Cosine annealing with restarts (`Link`_).
    
    Args:
        optimizer (Optimizer):
        t_max (int): The maximum number of iterations within the first cycle.
        eta_min (float): The minimum learning rate. Default: 0.
        last_epoch (int): The index of the last epoch. This is used when restarting. Default: -1.
        factor (float): The factor by which the cycle length (T_max) increases after 
                        each restart. Default: 1.

    .. _Link:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0., 
                 last_epoch: int = -1, factor: float = 1.) -> None:
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart = 0
        self._cycle_counter = 0
        self._cycle_factor = 1.
        self._updated_cycle_len = t_0
        self._initialized = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        r"""
        Get updated learning rate.

        Returns:
            lrs (List[float]): A list of current learning rates.

        Note:
            We need to check if this is the first time ``self.get_lr()`` was called,
            since ``torch.optim.lr_scheduler._LRScheduler`` will call ``self.get_lr()``
            when first initialized, but the learning rate should remain unchanged
            for the first epoch.
        """
        if not self._initialized:
            self._initialized = True
            return self.base_lrs
        step = self.last_epoch
        self._cycle_counter = step - self._last_restart
        lrs = [
            self.eta_min + ((lr - self.eta_min) / 2) * (
                np.cos(
                    np.pi *
                    (self._cycle_counter % self._updated_cycle_len) /
                    self._updated_cycle_len
                ) + 1
            )
            for lr in self.base_lrs
        ]
        if self._cycle_counter != 0 and self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step
        return lrs
