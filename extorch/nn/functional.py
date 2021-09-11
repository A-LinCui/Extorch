from typing import List

import torch
from torch import Tensor
import torch.nn as nn


def average_logits(logits_list: List[Tensor]) -> Tensor:
    r"""
    Aggregates logits from different networks in the average manner,
    and returns the aggregated logits. Used for neural ensemble networks.

    Args:
        logits_list (List[Tensor]): A list of logits from different networks.

    Returns:
        The aggregated logits (Tensor).
    """
    return sum(logtis_list) / len(logits_list)


def soft_voting(logits_list: List[Tensor]) -> Tensor:
    r"""
    Aggregates logits from different sub-networks in the soft-voting manner, 
    and returns the aggregated logits. Used for neural ensemble networks.

    Args:
        logits_list (List[Tensor]): A list of logits from different networks.

    Returns:
        The aggregated logits (Tensor).
    """
    return sum([nn.functional.softmax(logits, dim = 1) for logits in logits_list]) / len(logits_list)


def dec_soft_assignment(input: Tensor, centers: Tensor, alpha: float) -> Tensor:
    r"""
    Soft assignment used by Deep Embedded Clustering (DEC, `Link`_).
    Measure the similarity between embedded point and centroid with the Student’s t-distribution.

    Args:
        input (Tensor): A batch of embedded points. FloatTensor of [batch size, embedding dimension].
        centers (Tensor): The cluster centroids. FloatTensor of [cluster_number, embedding dimension].
        alpha (float): The degrees of freedom of the Student’s tdistribution. Default: 1.0.

    Returns:
        similarity (Tensor): The similarity between embedded point and centroid. FloatTensor [batch size, cluster_number].

    Examples::
        >>> embeddings = torch.ones((2, 10))
        >>> centers = torch.zeros((3, 10))
        >>> similarity = dec_soft_assignment(embeddings, centers, alpha = 1.)

    .. _Link:
        https://arxiv.org/abs/1511.06335
    """
    norm_squared = torch.sum((input.unsqueeze(1) - centers) ** 2, 2)
    numerator = 1.0 / (1.0 + norm_squared / alpha)
    numerator = numerator ** ((alpha + 1.) / 2)
    return numerator / torch.sum(numerator, dim = 1, keepdim = True)
