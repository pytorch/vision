import numpy as np
import math
import torch

from collections import OrderedDict
from enum import Enum, auto
from typing import Any, Callable, List, Optional, Tuple
from torch import nn, Tensor

from .._internally_replaced_utils import load_state_dict_from_url
from torchvision.models.mobilenetv2 import _make_divisible


model_urls = {
    # TODO(kazhang): add pretrained weights
    "regnet_y_400m": "",
}


class _SqueezeExcitation(nn.Module):
    """
    Squeeze and excitation layer from
    `"Squeeze-and-Excitation Networks" <https://arxiv.org/pdf/1709.01507>`_.
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: Optional[int] = 16,
        reduced_channels: Optional[int] = None,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Either reduction_ratio is defined, or out_channels is defined,
        # neither both nor none of them
        assert bool(reduction_ratio) != bool(reduced_channels)

        if activation is None:
            activation = nn.ReLU()

        reduced_channels = (
            in_channels // reduction_ratio if reduced_channels is None else reduced_channels
        )
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, stride=1, bias=True),
            activation,
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_squeezed = self.avgpool(x)
        x_excited = self.excitation(x_squeezed)
        x_scaled = x * x_excited
        return x_scaled


class BasicTransform(nn.Sequential):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
    ) -> None:
        super().__init__()

        self.a = nn.Sequential(
            nn.Conv2d(width_in, width_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum),
            activation,
            nn.Conv2d(width_out, width_out, 3, stride=1, padding=1, bias=False),
        )

        self.final_bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.depth = 2


class ResStemCifar(nn.Sequential):
    """ResNet stem for CIFAR: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(width_in, width_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum),
            activation,
        )
        self.depth = 2


class ResStemIN(nn.Sequential):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(width_in, width_out, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum),
            activation,
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.depth = 3


class SimpleStemIN(nn.Sequential):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(width_in, width_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum),
            activation,
        )
        self.depth = 2


class VanillaBlock(nn.Sequential):
    """Vanilla block: [3x3 conv, BN, Relu] x2."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.a = nn.Sequential(
            nn.Conv2d(width_in, width_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum),
            activation,
        )

        self.b = nn.Sequential(
            nn.Conv2d(width_out, width_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum),
            activation,
        )

        self.depth = 2


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.proj_block = (width_in != width_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(
                width_in, width_out, 1, stride=stride, padding=0, bias=False
            )
            self.bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.f = BasicTransform(
            width_in, width_out, stride, bn_epsilon, bn_momentum, activation
        )
        self.activation = activation

        # The projection and transform happen in parallel,
        # and ReLU is not counted with respect to depth
        self.depth = self.f.depth

    def forward(self, x: Tensor) -> Tensor:
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)

        return self.activation(x)


class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
    ) -> None:
        super().__init__()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        self.a = nn.Sequential(
            nn.Conv2d(width_in, w_b, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(w_b, eps=bn_epsilon, momentum=bn_momentum),
            activation,
        )

        self.b = nn.Sequential(
            nn.Conv2d(w_b, w_b, 3, stride=stride, padding=1, groups=g, bias=False),
            nn.BatchNorm2d(w_b, eps=bn_epsilon, momentum=bn_momentum),
            activation,
        )

        if se_ratio:
            # The SE reduction ratio is defined with respect to the
            # beginning of the block
            width_se_out = int(round(se_ratio * width_in))
            self.se = _SqueezeExcitation(
                in_channels=w_b,
                reduction_ratio=None,
                reduced_channels=width_se_out,
                activation=activation,
            )

        self.c = nn.Conv2d(w_b, width_out, 1, stride=1, padding=0, bias=False)
        self.final_bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.depth = 3 if not se_ratio else 4


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()

        # Use skip connection with projection if shape changes
        self.proj_block = (width_in != width_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(
                width_in, width_out, 1, stride=stride, padding=0, bias=False
            )
            self.bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            bn_epsilon,
            bn_momentum,
            activation,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )
        self.activation = activation

        # The projection and transform happen in parallel,
        # and activation is not counted with respect to depth
        self.depth = self.f.depth

    def forward(self, x: Tensor) -> Tensor:
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class ResBottleneckLinearBlock(nn.Module):
    """Residual linear bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
        group_width: int = 1,
        bottleneck_multiplier: float = 4.0,
        se_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.has_skip = (width_in == width_out) and (stride == 1)
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            bn_epsilon,
            bn_momentum,
            activation,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )

        self.depth = self.f.depth

    def forward(self, x: Tensor) -> Tensor:
        return x + self.f(x) if self.has_skip else self.f(x)


class AnyStage(nn.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        block_constructor: Callable[..., nn.Module],
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float] = None,
        stage_index: int = 0,
    ) -> None:
        super().__init__()
        self.stage_depth = 0

        for i in range(depth):
            block = block_constructor(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                bn_epsilon,
                bn_momentum,
                activation,
                group_width,
                bottleneck_multiplier,
                se_ratio,
            )

            self.stage_depth += block.depth
            self.add_module(f"block{stage_index}-{i}", block)


class RegNetParams:
    def __init__(
        self,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        stem_type: Callable[..., nn.Module] = SimpleStemIN,
        stem_width: int = 32,
        block_type: Callable[..., nn.Module] = ResBottleneckBlock,
        activation: Callable[..., nn.Module] = nn.ReLU,
        use_se: bool = True,
        se_ratio: float = 0.25,
        bn_epsilon: float = 1e-05,
        bn_momentum: float = 0.1,
        num_classes: int = 1000,
    ) -> None:
        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")
        self.depth = depth
        self.w_0 = w_0
        self.w_a = w_a
        self.w_m = w_m
        self.group_width = group_width
        self.bottleneck_multiplier = bottleneck_multiplier
        self.stem_type = stem_type
        self.block_type = block_type
        self.activation = activation
        self.stem_width = stem_width
        self.use_se = use_se
        self.se_ratio = se_ratio if use_se else None
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum
        self.num_classes = num_classes

    def get_expanded_params(self):
        """
        Programatically compute all the per-block settings,
        given the RegNet parameters.

        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - `w_a` is the width progression slope
        - `w_0` is the initial width
        - `w_m` is the width stepping in the log space

        In other terms
        `log(block_width) = log(w_0) + w_m * block_capacity`,
        with `bock_capacity` ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.

        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage.
        """

        QUANT = 8
        STRIDE = 2

        # Compute the block widths. Each stage has one unique block width
        widths_cont = np.arange(self.depth) * self.w_a + self.w_0
        block_capacity = np.round(np.log(widths_cont / self.w_0) / np.log(self.w_m))
        block_widths = (
            np.round(np.divide(self.w_0 * np.power(self.w_m, block_capacity), QUANT))
            * QUANT
        )
        num_stages = len(np.unique(block_widths))
        block_widths = block_widths.astype(int).tolist()

        # Convert to per stage parameters
        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = np.diff([d for d, t in enumerate(splits) if t]).tolist()

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [self.bottleneck_multiplier] * num_stages
        group_widths = [self.group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, group_widths = self._adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return zip(
            stage_widths, strides, stage_depths, group_widths, bottleneck_multipliers
        )

    @staticmethod
    def _adjust_widths_groups_compatibilty(
            stage_widths: List[int], bottleneck_ratios: List[float],
            group_widths: List[int]) -> Tuple[List[int], List[int]]:
        """
        Adjusts the compatibility of widths and groups,
        depending on the bottleneck ratio.
        """
        # Compute all widths for the current settings
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

        # Compute the adjusted widths so that stage and group widths fit
        ws_bot = [_make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min


class RegNet(nn.Module):
    def __init__(self, params: RegNetParams) -> None:
        super().__init__()

        activation = params.activation(inplace=True)

        # Ad hoc stem
        self.stem = params.stem_type(
            3,  # width_in
            params.stem_width,
            params.bn_epsilon,
            params.bn_momentum,
            activation,
        )

        current_width = params.stem_width

        self.trunk_depth = 0

        blocks = []
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(params.get_expanded_params()):
            blocks.append(
                (
                    f"block{i+1}",
                    AnyStage(
                        current_width,
                        width_out,
                        stride,
                        depth,
                        params.block_type,
                        params.bn_epsilon,
                        params.bn_momentum,
                        activation,
                        group_width,
                        bottleneck_multiplier,
                        params.se_ratio,
                        stage_index=i + 1,
                    ),
                )
            )

            self.trunk_depth += blocks[-1][1].stage_depth

            current_width = width_out

        self.trunk_output = nn.Sequential(OrderedDict(blocks))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=current_width, out_features=params.num_classes)

        # Init weights and good to go
        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.trunk_output(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x

    def _init_weights(self) -> None:
        # Performs ResNet-style weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()


def _regnet(arch: str, params: RegNetParams, pretrained: bool, progress: bool, **kwargs: Any) -> RegNet:
    model = RegNet(params)
    if pretrained:
        if arch not in model_urls:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def regnet_y_400mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    params = RegNetParams(depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, **kwargs)
    return _regnet("regnet_y_400mf", params, pretrained, progress, **kwargs)
