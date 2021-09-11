from typing import Tuple, List, Union

import torch
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    r"""
    Basic muti-layer perception with relu as the activation function.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        hiddens (Union[Tuple[int], List[int]]): Hidden dimensions.
        dropout (float): Applied dropout rate.

    Examples::
        >>> m = MLP(32, 20, (10, 10, 10), 0.1)
        >>> input = torch.randn(2, 32) # shape [2, 32]
        >>> output = m(input) # shape [2, 20]
    """
    def __init__(self, dim_in: int, dim_out: int, 
                 hiddens: Union[Tuple[int], List[int]], dropout: float = 0.) -> None:
        super(MLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hiddens = hiddens
        self.dropout = dropout

        self.mlp = []
        for hidden_size in hiddens:
            self.mlp.append(nn.Sequential(
                nn.Linear(dim_in, hidden_size),
                nn.ReLU(inplace = False),
                nn.Dropout(p = dropout)))
            dim_in = hidden_size

        self.mlp.append(nn.Linear(dim_in, dim_out))
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, input: Tensor) -> Tensor:
        return self.mlp(input)

