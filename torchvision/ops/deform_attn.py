import torch
from torch import Tensor
from torchvision.extension import _assert_has_ops

from ..utils import _log_api_usage_once


def deform_attn(
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_loc: Tensor,
    attn_weight: Tensor,
    im2col_step: int = 64,
) -> Tensor:
    r"""
    Performs Deformable Attention, as described in
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection
    <https://arxiv.org/abs/2010.04159>`__.

    Args:
        value (Tensor[batch_size, num_value, num_heads, channels]): input value tensor
        spatial_shapes (Tensor[num_levels, 2]): spatial shapes (H, W) for each feature level
        level_start_index (Tensor[num_levels]): starting index for each feature level in flattened value
        sampling_loc (Tensor[batch_size, num_query, num_heads, num_levels, num_points, 2]):
            sampling locations in normalized coordinates [0, 1]
        attn_weight (Tensor[batch_size, num_query, num_heads, num_levels, num_points]):
            attention weights for each sampling point
        im2col_step (int): step size for im2col operation to reduce memory usage. Default: 64

    Returns:
        Tensor[batch_size, num_query, num_heads * channels]: result of deformable attention

    Examples::
        >>> batch_size, num_query, num_heads, channels = 2, 100, 8, 32
        >>> num_levels, num_points = 4, 4
        >>> # Create value tensor (flattened spatial dimensions across all levels)
        >>> num_value = 1024 + 256 + 64 + 16  # sum of H*W for each level
        >>> value = torch.rand(batch_size, num_value, num_heads, channels, device="cuda")
        >>> # Spatial shapes for 4 feature levels
        >>> spatial_shapes = torch.tensor([[32, 32], [16, 16], [8, 8], [4, 4]], dtype=torch.long, device="cuda")
        >>> # Starting indices for each level in the flattened value tensor
        >>> level_start_index = torch.tensor([0, 1024, 1280, 1344], dtype=torch.long, device="cuda")
        >>> # Sampling locations (normalized coordinates in [0, 1])
        >>> sampling_loc = torch.rand(batch_size, num_query, num_heads, num_levels, num_points, 2, device="cuda")
        >>> # Attention weights (should sum to 1 across num_levels * num_points)
        >>> attn_weight = torch.rand(batch_size, num_query, num_heads, num_levels, num_points, device="cuda")
        >>> attn_weight = attn_weight.softmax(-1).softmax(-2)  # normalize
        >>> out = deform_attn(value, spatial_shapes, level_start_index, sampling_loc, attn_weight)
        >>> print(out.shape)
        >>> # returns
        >>>  torch.Size([2, 100, 256])
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(deform_attn)
    _assert_has_ops()

    return torch.ops.torchvision.deform_attn(
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc,
        attn_weight,
        im2col_step,
    )
