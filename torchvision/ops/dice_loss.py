import torch
import torch.nn.functional as F

def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, reduction: str = "none", eps: float = 1e-8) -> torch.Tensor:
    """Criterion that computes Sørensen-Dice Coefficient loss.

    We compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X \cap Y|}{|X| + |Y|}

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be thess tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    Args:
        inputs: (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets: (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        eps: (float, optional): Scalar to enforce numerical stabiliy.
        reduction (string, optional): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.

    Return:
        Tensor: Loss tensor with the reduction option applied.
    """
    if not isinstance(inputs, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(inputs)}")

    if not len(inputs.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxNxHxW. Got: {inputs.shape}")

    if not inputs.shape[-2:] == targets.shape[-2:]:
        raise ValueError(f"input and target shapes must be the same. Got: {inputs.shape} and {targets.shape}")

    if not inputs.device == targets.device:
        raise ValueError(f"input and target must be in the same device. Got: {inputs.device} and {targets.device}")

    # compute softmax over the classes axis
    p = F.softmax(inputs, dim=1)

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(p * targets, dims)
    cardinality = torch.sum(p + targets, dims)

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