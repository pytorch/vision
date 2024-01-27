from typing import Optional

import torch
from torch import Tensor

from ..utils import _log_api_usage_once


def poly_loss(
    x: Tensor,
    target: Tensor,
    eps: float = 2.0,
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tensor:
    """Implements the Poly1 loss from `"PolyLoss: A Polynomial Expansion Perspective of Classification Loss
    Functions" <https://arxiv.org/pdf/2204.12511.pdf>`_.

    Args:
        x (Tensor[N, K, ...]): predicted probability
        target (Tensor[N, K, ...]): target probability
        eps (float, optional): epsilon 1 from the paper
        weight (Tensor[K], optional): manual rescaling of each class
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.

    Returns:
        Tensor: loss reduced with `reduction` method
    """
    # Original implementation from https://github.com/frgfm/Holocron/blob/main/holocron/nn/functional.py
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(poly_loss)
    # log(P[class]) = log_softmax(score)[class]
    logpt = F.log_softmax(x, dim=1)

    # Compute pt and logpt only for target classes (the remaining will have a 0 coefficient)
    logpt = logpt.transpose(1, 0).flatten(1).gather(0, target.view(1, -1)).squeeze()
    # Ignore index (set loss contribution to 0)
    valid_idxs = torch.ones(target.view(-1).shape[0], dtype=torch.bool, device=x.device)
    if ignore_index >= 0 and ignore_index < x.shape[1]:
        valid_idxs[target.view(-1) == ignore_index] = False

    # Get P(class)
    loss = -1 * logpt + eps * (1 - logpt.exp())

    # Weight
    if weight is not None:
        # Tensor type
        if weight.type() != x.data.type():
            weight = weight.type_as(x.data)
        logpt = weight.gather(0, target.data.view(-1)) * logpt

    # Loss reduction
    if reduction == "sum":
        loss = loss[valid_idxs].sum()
    elif reduction == "mean":
        loss = loss[valid_idxs].mean()
    elif reduction == "none":
        # if no reduction, reshape tensor like target
        loss = loss.view(*target.shape)
    else:
        raise ValueError(f"invalid value for arg 'reduction': {reduction}")

    return loss
