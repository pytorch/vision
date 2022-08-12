# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Callable, List, Sequence, Type
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.misc import Conv3dNormActivation


class SepConv3d(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(SepConv3d, self).__init__(
            Conv3dNormActivation(
                in_planes,
                out_planes,
                kernel_size=(1, kernel_size, kernel_size),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
                norm_layer=partial(nn.BatchNorm3d, eps=1e-3, momentum=1e-3, affine=True),
            ),
            Conv3dNormActivation(
                out_planes,
                out_planes,
                kernel_size=(kernel_size, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                bias=False,
                norm_layer=partial(nn.BatchNorm3d, eps=1e-3, momentum=1e-3, affine=True),
            ),
        )


class SepInceptionBlock3D(nn.Module):
    """Separable Inception block for S3D model

    Args:
        in_planes (int): dimension of input
        branch_layers (List[List[int]]): list of list of output dimensions for each layer in each branch.
            Must be in the format ``[[b0_out], [b1_mid, b1_out], [b2_mid, b2_out], [b3_out]]``.
            E.g.: ``[[5], [6,7], [8,9], [10]]`` means the 0th branch has output dim 5, the 1st branch has 
            a hidden output of dim 6 before the final output of dim 7, etc.
    """
    def __init__(self, in_planes, branch_layers):
        super().__init__()
        [b0_out], [b1_mid, b1_out], [b2_mid, b2_out], [b3_out] = branch_layers
        
        self.branch0 = Conv3dNormActivation(in_planes, b0_out, kernel_size=1, stride=1, norm_layer=partial(nn.BatchNorm3d, eps=1e-3, momentum=1e-3))
        self.branch1 = nn.Sequential(
            Conv3dNormActivation(in_planes, b1_mid, kernel_size=1, stride=1, norm_layer=partial(nn.BatchNorm3d, eps=1e-3, momentum=1e-3)),
            SepConv3d(b1_mid, b1_out, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            Conv3dNormActivation(in_planes, b2_mid, kernel_size=1, stride=1, norm_layer=partial(nn.BatchNorm3d, eps=1e-3, momentum=1e-3)),
            SepConv3d(b2_mid, b2_out, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            Conv3dNormActivation(in_planes, b3_out, kernel_size=1, stride=1, norm_layer=partial(nn.BatchNorm3d, eps=1e-3, momentum=1e-3)),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class S3D(nn.Module):
    """S3D is a video classification model that improves over I3D in speed by replacing
        3D convolutions with a spatial 2D convolution followed by a temporal 1D convolution.
        Paper: https://arxiv.org/abs/1712.04851
        Code: https://github.com/kylemin/S3D

    Args:
        num_class (int): number of classes for the classification task

    Inputs:
        x (Tensor): batch of videos with dimensions (batch, channel, time, height, width)
    """

    def __init__(
        self, 
        num_classes: int = 400,
    ):
        super().__init__()
        self.base = nn.Sequential(
            SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            Conv3dNormActivation(64, 64, kernel_size=1, stride=1, norm_layer=partial(nn.BatchNorm3d, eps=1e-3, momentum=1e-3)),
            SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            SepInceptionBlock3D(192, [[64], [96, 128], [16, 32], [32]]),
            SepInceptionBlock3D(256, [[128], [128, 192], [32, 96], [64]]),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            SepInceptionBlock3D(480, [[192], [96, 208], [16, 48], [64]]),
            SepInceptionBlock3D(512, [[160], [112, 224], [24, 64], [64]]),
            SepInceptionBlock3D(512, [[128], [128, 256], [24, 64], [64]]), 
            SepInceptionBlock3D(512, [[112], [144, 288], [32, 64], [64]]),
            SepInceptionBlock3D(528, [[256], [160, 320], [32, 128], [128]]),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
            SepInceptionBlock3D(832, [[256], [160, 320], [32, 128], [128]]),
            SepInceptionBlock3D(832, [[384], [192, 384], [48, 128], [128]]),
        )
        self.fc = nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        y = self.base(x)
        y = F.avg_pool3d(y, (2, y.size(3), y.size(4)), stride=1)
        y = self.fc(y)
        y = y.view(y.size(0), y.size(1), y.size(2))
        logits = torch.mean(y, 2)

        return logits
