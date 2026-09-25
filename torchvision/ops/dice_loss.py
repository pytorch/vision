import torch
import torch.nn.functional as F

from ..utils import _log_api_usage_once


# Implementation adapted from https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/dice.html#dice_loss
def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, reduction: str = "none", eps: float = 1e-7) -> torch.Tensor:
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    We compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice\_Loss}(X, Y) = 1 - \frac{2 |X \cap Y|}{|X| + |Y|}

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    Args:
        inputs: (Tensor): A float tensor with rank >= 2 and shape (B, num_classes, N1, .... NK)
                where B is the Batch Size and num_classes is the number of classes.
                The predictions for each example.
        targets: (Tensor): A one-hot tensor with the same shape as inputs.
                The first dimension is the batch size and the second dimension is the
                number of classes.
        eps: (float, optional): Scalar to enforce numerical stability.
        reduction (string, optional): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.

    Return:
        Tensor: Loss tensor with the reduction option applied.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(dice_loss)

    # compute softmax over the classes axis
    p = F.softmax(inputs, dim=1)
    p = p.flatten(start_dim=1)

    targets = targets.flatten(start_dim=1)

    intersection = torch.sum(p * targets, dim=1)
    cardinality = torch.sum(p + targets, dim=1)

    dice_score = 2.0 * intersection / (cardinality + eps)

    loss = 1.0 - dice_score

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )

    return loss
