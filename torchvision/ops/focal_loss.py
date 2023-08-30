import torch
import torch.nn.functional as F

from ..utils import _log_api_usage_once


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(sigmoid_focal_loss)
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * _safe_pow_respecting_ascent_descent_order(1 - p_t, gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


def _safe_pow_respecting_ascent_descent_order(
    input: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    The elements of loss tensor can have either negative and positive signs. Where negative direction
    loss will attribute towards the gradient ascent, the positive will push for decent, both are
    equally valueable when navigating loss curve.

    When we power loss in the context of increasing penality for hard samples and softing the
    blow for easy ones, we just want to increase the "magnitude" of the direction of loss.
    If gamma is = 2,  an ascent i.e. a negative loss can become descent which is wrong. Likewise,
    when gamma is < 1, an ascent can destabilize training because of complex numbers.
    this can go on and get really ugly very quicly, because gamma = 3, an ascent will still be an ascent.
    The right way to power loss term is to power the magnitude in that direction.
    So in safe power, we capture the direction, then power magnitude and apply the direction back.

    Generally this wont be an issue as ascent is rare scenario but can happen.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
    Returns:
        The exponentiated loss tensor
    """
    direction_mask = torch.where(input < 0.0, -1, 1)
    return direction_mask * torch.pow(input.abs(), gamma)
