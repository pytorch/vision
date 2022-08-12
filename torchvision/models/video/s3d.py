# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.misc import Conv3dNormActivation

from .._api import register_model, WeightsEnum


__all__ = [
    "S3D",
    "S3D_Weights",
    "s3d",
]


class SepConv3d(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, stride: int, padding: int = 0):
        super().__init__(
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
    """Separable Inception block for S3D model.

    Args:
        in_planes (int): dimension of input
        branch_layers (List[List[int]]): list of list of output dimensions for each layer in each branch.
            Must be in the format ``[[b0_out], [b1_mid, b1_out], [b2_mid, b2_out], [b3_out]]``.
            E.g.: ``[[5], [6,7], [8,9], [10]]`` means the 0th branch has output dim 5, the 1st branch has
            a hidden output of dim 6 before the final output of dim 7, etc.
    """

    def __init__(self, in_planes: int, branch_layers: List[List[int]]):
        super().__init__()
        [b0_out], [b1_mid, b1_out], [b2_mid, b2_out], [b3_out] = branch_layers

        self.branch0 = Conv3dNormActivation(
            in_planes, b0_out, kernel_size=1, stride=1, norm_layer=partial(nn.BatchNorm3d, eps=1e-3, momentum=1e-3)
        )
        self.branch1 = nn.Sequential(
            Conv3dNormActivation(
                in_planes, b1_mid, kernel_size=1, stride=1, norm_layer=partial(nn.BatchNorm3d, eps=1e-3, momentum=1e-3)
            ),
            SepConv3d(b1_mid, b1_out, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            Conv3dNormActivation(
                in_planes, b2_mid, kernel_size=1, stride=1, norm_layer=partial(nn.BatchNorm3d, eps=1e-3, momentum=1e-3)
            ),
            SepConv3d(b2_mid, b2_out, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            Conv3dNormActivation(
                in_planes, b3_out, kernel_size=1, stride=1, norm_layer=partial(nn.BatchNorm3d, eps=1e-3, momentum=1e-3)
            ),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class S3D(nn.Module):
    """S3D main class.

    Args:
        num_class (int): number of classes for the classification task.

    Inputs:
        x (Tensor): batch of videos with dimensions (batch, channel, time, height, width)
    """

    def __init__(
        self,
        num_classes: int = 400,
    ) -> None:
        super().__init__()
        self.base = nn.Sequential(
            SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            Conv3dNormActivation(
                64, 64, kernel_size=1, stride=1, norm_layer=partial(nn.BatchNorm3d, eps=1e-3, momentum=1e-3)
            ),
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


class S3D_Weights(WeightsEnum):
    pass


@register_model()
def s3d(*, weights: Optional[S3D_Weights] = None, progress: bool = True, **kwargs: Any) -> S3D:
    """Construct Separable 3D CNN model.

    Reference: `Rethinking Spatiotemporal Feature Learning <https://arxiv.org/abs/1712.04851>`__.

    Args:
        weights (:class:`~torchvision.models.video.S3D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.S3D_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.S3D`` base class.
            Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/s3d.py>`_
            for more details about this class.

    """
    weights = S3D_Weights.verify(weights)

    model = S3D(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
