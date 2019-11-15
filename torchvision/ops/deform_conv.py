from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn.modules.utils import _pair
from torch.jit.annotations import List


def deform_conv(input, offset, weight, stride=(1, 1), pad=(0, 0), dilation=(1, 1), n_parallel_imgs=64):
    # type: (Tensor, Tensor, Tensor, Tuple[int, int], Tuple[int, int], Tuple[int, int], int) -> Tensor
    """
    Performs Deformable Convolution, described in Deformable Convolutional Networks

    Arguments:
        input (Tensor[batch_sz, in_channels, in_h, in_w]): input tensor
        offset (Tensor[batch_sz, 2 * n_offset_grps * weight_h * weight_w, out_h, out_w])
        weight (Tensor[out_channels, in_channels // n_weight_grps, weight_h, weight_w]):
            convolution weights, with n_weight_grps different connection groups
        stride (int or Tuple[int, int]): distance between convolution centers
        pad (int or Tuple[int, int]): height/width of padding of zeroes around each image
        dilation (int or Tuple[int, int]): point distance in convolution grid
        n_parallel_imgs (int): Number of images to be processed at once; does not change
            behavior, only used for performance purposes

    Returns:
        output (Tensor[batch_sz, out_channels, out_h, out_w]): result of convolution
    """

    stride_h, stride_w = stride
    pad_h, pad_w = pad
    dil_h, dil_w = dilation
    weights_h, weights_w = weight.shape[-2:]
    _, n_in_channels, in_h, in_w = input.shape

    n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)
    n_weight_grps = n_in_channels // weight.shape[1]

    return torch.ops.torchvision.deform_conv(
        input,
        offset,
        weight,
        *stride,
        *pad,
        *dilation,
        n_weight_grps,
        n_offset_grps,
        n_parallel_imgs)


class DeformConv(nn.Module):
    """
    See deform_conv
    """
    def __init__(self, stride=1, pad=0, dilation=1, n_parallel_imgs=64):
        super(DeformConv, self).__init__()
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.dilation = _pair(dilation)
        self.n_parallel_imgs = n_parallel_imgs

    def forward(self, input, offset, weight):
        return deform_conv(input, offset, weight, stride=self.stride, pad=self.pad,
                           dilation=self.dilation, n_parallel_imgs=self.n_parallel_imgs)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ')'
        return tmpstr
