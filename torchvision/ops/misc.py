from __future__ import division

"""
helper class that supports empty tensors on some nn functions.

Ideally, add support directly in PyTorch to empty tensors in
those functions.

This can be removed once https://github.com/pytorch/pytorch/issues/12013
is implemented
"""

import math
import torch
from torch.nn.modules.utils import _pair
from typing import List


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


# class Conv2d(torch.nn.Conv2d):
#     """
#     Equivalent to nn.Conv2d, but with support for empty batch sizes.
#     This will eventually be supported natively by PyTorch, and this
#     class can go away.
#     """
#     def forward(self, x):
#         if x.numel() > 0:
#             return super(Conv2d, self).forward(x)
#         # get output shape

#         output_shape = [
#             (i + 2 * p - (di * (k - 1) + 1)) // d + 1
#             for i, p, di, k, d in zip(
#                 x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
#             )
#         ]
#         output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
#         return _NewEmptyTensorOp.apply(x, output_shape)


class ConvTranspose2d(torch.nn.ConvTranspose2d):
    """
    Equivalent to nn.ConvTranspose2d, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    def forward(self, x):
        if x.numel() > 0:
            return super(ConvTranspose2d, self).forward(x)
        # get output shape

        output_shape = [
            (i - 1) * d - 2 * p + (di * (k - 1) + 1) + op
            for i, p, di, k, d, op in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
                self.output_padding,
            )
        ]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


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
        return _NewEmptyTensorOp.apply(x, output_shape)

def _check_size_scale_factor(dim, size, scale_factor):
    if size is None and scale_factor is None:
        raise ValueError("either size or scale_factor should be defined")
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor should be defined")
    if (
        scale_factor is not None and
        isinstance(scale_factor, tuple) and
        len(scale_factor) != dim
    ):
        raise ValueError(
            "scale_factor shape must match input shape. "
            "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
        )

def _output_size(input, dim, size, scale_factor):
    # type: (Tensor, int, Optional[List[int]], Optional[float])
    # _check_size_scale_factor(dim)
    if size is not None:
        return size
    scale_factors = (2., 2.)
    # math.floor might return float in py2.7
    out_size : List[int] = []
    for i in range(dim):
        out_size.append(int(math.floor(input.size(i + 2) * scale_factors[i])))

    return out_size


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # type: (Tensor, Optional[List[int]], Optional[float], str, bool)
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    # if input.numel() > 0:
    return torch.nn.functional.interpolate(
        input, size, scale_factor, mode, align_corners
    )

    # output_shape = _output_size(input, 2, size, scale_factor)
    # output_shape = input.shape[:-2] + output_shape
    # return _NewEmptyTensorOp.apply(input, output_shape)


# This is not in nn
class FrozenBatchNorm2d(torch.jit.ScriptModule):
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

    @torch.jit.script_method
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
