import torch
import torch.nn.functional as F
from torch import Tensor

from ..utils import _log_api_usage_once


def masks_to_boundaries(masks: Tensor, kernel_size: int = 3) -> Tensor:
    """
    Compute the boundaries around the provided binary masks.

    The boundary of each mask is computed as the difference between the mask and
    its morphological erosion with a square structuring element.

    Args:
        masks (Tensor[N, H, W]): binary masks where N is the number of masks and
            (H, W) are the spatial dimensions.
        kernel_size (int, optional): size of the square structuring element used
            for erosion. It must be a positive odd integer. Default: 3.

    Returns:
        Tensor[N, H, W]: boolean boundary masks.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(masks_to_boundaries)
    if masks.dim() != 3:
        raise ValueError("masks must have shape [N, H, W]")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if masks.numel() == 0:
        return torch.zeros_like(masks, dtype=torch.bool)

    masks_bool = masks.to(dtype=torch.bool)
    masks_float = masks_bool.float().unsqueeze(1)
    weight = torch.ones((1, 1, kernel_size, kernel_size), dtype=masks_float.dtype, device=masks.device)

    eroded_masks = F.conv2d(masks_float, weight, padding=kernel_size // 2).squeeze(1) == kernel_size * kernel_size
    return torch.logical_xor(masks_bool, eroded_masks)
