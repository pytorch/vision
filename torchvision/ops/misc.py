from __future__ import division
from collections import OrderedDict
from torch.jit.annotations import Optional, List
from torch import Tensor

"""
helper class that supports empty tensors on some nn functions.

Ideally, add support directly in PyTorch to empty tensors in
those functions.

This can be removed once https://github.com/pytorch/pytorch/issues/12013
is implemented
"""

import math
import torch
from torchvision.ops import _new_empty_tensor
from torch.nn import Module, Conv2d
import torch.nn.functional as F


class ConvTranspose2d(torch.nn.ConvTranspose2d):
    """
    Equivalent to nn.ConvTranspose2d, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    def forward(self, x):
        if x.numel() > 0:
            return self.super_forward(x)
        # get output shape

        output_shape = [
            (i - 1) * d - 2 * p + (di * (k - 1) + 1) + op
            for i, p, di, k, d, op in zip(
                x.shape[-2:],
                list(self.padding),
                list(self.dilation),
                list(self.kernel_size),
                list(self.stride),
                list(self.output_padding),
            )
        ]
        output_shape = [x.shape[0], self.out_channels] + output_shape
        return _new_empty_tensor(x, output_shape)

    def super_forward(self, input, output_size=None):
        # type: (Tensor, Optional[List[int]]) -> Tensor
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)

        return F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class BatchNorm2d(torch.nn.BatchNorm2d):
    """
    Equivalent to nn.BatchNorm2d, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    def forward(self, x):
        if x.numel() > 0:
            return super(BatchNorm2d, self).forward(x)
        # get output shape
        output_shape = x.shape
        return _new_empty_tensor(x, output_shape)


def _check_size_scale_factor(dim, size, scale_factor):
    # type: (int, Optional[List[int]], Optional[float]) -> None
    if size is None and scale_factor is None:
        raise ValueError("either size or scale_factor should be defined")
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor should be defined")
    if scale_factor is not None and isinstance(scale_factor, tuple)\
            and len(scale_factor) != dim:
        raise ValueError(
            "scale_factor shape must match input shape. "
            "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
        )


def _output_size(dim, input, size, scale_factor):
    # type: (int, Tensor, Optional[List[int]], Optional[float]) -> List[int]
    assert dim == 2
    _check_size_scale_factor(dim, size, scale_factor)
    if size is not None:
        return size
    # if dim is not 2 or scale_factor is iterable use _ntuple instead of concat
    assert scale_factor is not None and isinstance(scale_factor, (int, float))
    scale_factors = [scale_factor, scale_factor]
    # math.floor might return float in py2.7
    return [
        int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)
    ]


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners
        )

    output_shape = _output_size(2, input, size, scale_factor)
    output_shape = input.shape[:-2] + output_shape
    return _new_empty_tensor(input, output_shape)


# This is not in nn
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * rv.rsqrt()
        bias = b - rm * scale
        return x * scale + bias
