import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from torchvision.extension import _assert_has_ops

from ..utils import _log_api_usage_once


def deform_conv2d(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    mask: Optional[Tensor] = None,
) -> Tensor:
    r"""
    Performs Deformable Convolution v2, described in
    `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168>`__ if :attr:`mask` is not ``None`` and
    Performs Deformable Convolution, described in
    `Deformable Convolutional Networks
    <https://arxiv.org/abs/1703.06211>`__ if :attr:`mask` is ``None``.

    Args:
        input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]):
            offsets to be applied for each position in the convolution kernel.
        weight (Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]): convolution weights,
            split into groups of size (in_channels // groups)
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int, int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int, int]): height/width of padding of zeroes around
            each image. Default: 0
        dilation (int or Tuple[int, int]): the spacing between kernel elements. Default: 1
        mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]):
            masks to be applied for each position in the convolution kernel. Default: None

    Returns:
        Tensor[batch_sz, out_channels, out_h, out_w]: result of convolution

    Examples::
        >>> input = torch.rand(4, 3, 10, 10)
        >>> kh, kw = 3, 3
        >>> weight = torch.rand(5, 3, kh, kw)
        >>> # offset and mask should have the same spatial size as the output
        >>> # of the convolution. In this case, for an input of 10, stride of 1
        >>> # and kernel size of 3, without padding, the output size is 8.
        >>> # For the simplest case, 1 channel 3x3 kernel, at each valid (consider padding) input feature position:
        >>> # [[p0 p1 p2],
        >>> #  [p3 p4 p5],
        >>> #  [p6 p7 p8]]
        >>> # The 18 corresponding offsets will be:
        >>> # [p0_offset_h, p0_offset_w, p1_offset_h, p1_offset_w, p2_offset_h, ..., p8_offset_w]
        >>> # ``_h`` means pixel offset in the height direction (i.e. p0 -> p3 -> p6),
        >>> # ``_w`` means pixel offset in the width direction (i.e. p0 -> p1 -> p2).
        >>> offset = torch.rand(4, 2 * kh * kw, 8, 8)
        >>> mask = torch.rand(4, kh * kw, 8, 8)
        >>> out = deform_conv2d(input, offset, weight, mask=mask)
        >>> print(out.shape)
        >>> # returns
        >>>  torch.Size([4, 5, 8, 8])

    More complex examples::

        h = w = 3
        # batch_size, num_channels, out_height, out_width
        x = torch.arange(h * w * 3, dtype=torch.float32).reshape(1, 3, h, w)

        # to show the effect of offset more intuitively, only the case of kh=kw=1 is considered here
        # and we use the same offset for each local neighborhood in the single channel
        offset = torch.FloatTensor(
            [  # create our predefined offset with offset_groups = 3
                0, -1,  # sample the left pixel of the centroid pixel
                0, 1,  # sample the right pixel of the centroid pixel
                -1, 0,  # sample the top pixel of the centroid pixel
            ]  # here, we divide the input channels into offset_groups groups with different offsets.
        ).reshape(1, 2 * 3 * 1 * 1, 1, 1)
        # so we repeat the offset to the whole space: batch_size, 2 * offset_groups * kh * kw, out_height, out_width
        offset = offset.repeat(1, 1, h, w)

        weight = torch.FloatTensor(
            [
                [1, 0, 0],  # only extract the first channel of the input tensor
                [0, 1, 0],  # only extract the second channel of the input tensor
                [1, 1, 0],  # add the first and the second channels of the input tensor
                [0, 0, 1],  # only extract the third channel of the input tensor
                [0, 1, 0],  # only extract the second channel of the input tensor
            ]
        ).reshape(5, 3, 1, 1)
        deconv_shift = deform_conv2d(x, offset=offset, weight=weight)
        print(deconv_shift)

        tensor([[[[ 0.,  0.,  1.],  # offset=(0, -1) the first channel of the input tensor
                [ 0.,  3.,  4.],  # output hw indices (1, 2) => (1, 2-1) => input indices (1, 1)
                [ 0.,  6.,  7.]], # output hw indices (2, 1) => (2, 1-1) => input indices (2, 0)

                [[10., 11.,  0.],  # offset=(0, 1) the second channel of the input tensor
                [13., 14.,  0.],  # output hw indices (1, 1) => (1, 1+1) => input indices (1, 2)
                [16., 17.,  0.]], # output hw indices (2, 0) => (2, 0+1) => input indices (2, 1)

                [[10., 11.,  1.],  # offset=[(0, -1), (0, 1)], accumulate the first and second channels after being sampled with an offset.
                [13., 17.,  4.],
                [16., 23.,  7.]],

                [[ 0.,  0.,  0.],  # offset=(-1, 0) the third channel of the input tensor
                [18., 19., 20.],  # output hw indices (1, 1) => (1-1, 1) => input indices (0, 1)
                [21., 22., 23.]], # output hw indices (2, 2) => (2-1, 2) => input indices (1, 2)

                [[10., 11.,  0.],  # offset=(0, 1) the second channel of the input tensor
                [13., 14.,  0.],  # output hw indices (1, 1) => (1, 1+1) => input indices (1, 2)
                [16., 17.,  0.]]]])  # output hw indices (2, 0) => (2, 0+1) => input indices (2, 1)
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(deform_conv2d)
    _assert_has_ops()
    out_channels = weight.shape[0]

    use_mask = mask is not None

    if mask is None:
        mask = torch.zeros((input.shape[0], 1), device=input.device, dtype=input.dtype)

    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    weights_h, weights_w = weight.shape[-2:]
    _, n_in_channels, _, _ = input.shape

    n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)
    n_weight_grps = n_in_channels // weight.shape[1]

    if n_offset_grps == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of 2 * weight.size[2] * weight.size[3].\n"
            f"Got offset.shape[1]={offset.shape[1]}, while 2 * weight.size[2] * weight.size[3]={2 * weights_h * weights_w}"
        )

    return torch.ops.torchvision.deform_conv2d(
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,
    )


class DeformConv2d(nn.Module):
    """
    See :func:`deform_conv2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        _log_api_usage_once(self)

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = Parameter(
            torch.empty(out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1])
        )

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, offset: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
            offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]):
                offsets to be applied for each position in the convolution kernel.
            mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]):
                masks to be applied for each position in the convolution kernel.
        """
        return deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"{self.in_channels}"
            f", {self.out_channels}"
            f", kernel_size={self.kernel_size}"
            f", stride={self.stride}"
        )
        s += f", padding={self.padding}" if self.padding != (0, 0) else ""
        s += f", dilation={self.dilation}" if self.dilation != (1, 1) else ""
        s += f", groups={self.groups}" if self.groups != 1 else ""
        s += ", bias=False" if self.bias is None else ""
        s += ")"

        return s
