# Modified from
# https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/models/anynet.py
# https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/models/regnet.py


import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import nn, Tensor

from .._internally_replaced_utils import load_state_dict_from_url
from ..ops.misc import ConvNormActivation, SqueezeExcitation
from ..utils import _log_api_usage_once
from ._utils import _make_divisible


__all__ = [
    "RegNet",
    "regnet_y_400mf",
    "regnet_y_800mf",
    "regnet_y_1_6gf",
    "regnet_y_3_2gf",
    "regnet_y_8gf",
    "regnet_y_16gf",
    "regnet_y_32gf",
    "regnet_y_128gf",
    "regnet_x_400mf",
    "regnet_x_800mf",
    "regnet_x_1_6gf",
    "regnet_x_3_2gf",
    "regnet_x_8gf",
    "regnet_x_16gf",
    "regnet_x_32gf",
]


model_urls = {
    "regnet_y_400mf": "https://download.pytorch.org/models/regnet_y_400mf-c65dace8.pth",
    "regnet_y_800mf": "https://download.pytorch.org/models/regnet_y_800mf-1b27b58c.pth",
    "regnet_y_1_6gf": "https://download.pytorch.org/models/regnet_y_1_6gf-b11a554e.pth",
    "regnet_y_3_2gf": "https://download.pytorch.org/models/regnet_y_3_2gf-b5a9779c.pth",
    "regnet_y_8gf": "https://download.pytorch.org/models/regnet_y_8gf-d0d0e4a8.pth",
    "regnet_y_16gf": "https://download.pytorch.org/models/regnet_y_16gf-9e6ed7dd.pth",
    "regnet_y_32gf": "https://download.pytorch.org/models/regnet_y_32gf-4dee3f7a.pth",
    "regnet_x_400mf": "https://download.pytorch.org/models/regnet_x_400mf-adf1edd5.pth",
    "regnet_x_800mf": "https://download.pytorch.org/models/regnet_x_800mf-ad17e45c.pth",
    "regnet_x_1_6gf": "https://download.pytorch.org/models/regnet_x_1_6gf-e3633e7f.pth",
    "regnet_x_3_2gf": "https://download.pytorch.org/models/regnet_x_3_2gf-f342aeae.pth",
    "regnet_x_8gf": "https://download.pytorch.org/models/regnet_x_8gf-03ceed89.pth",
    "regnet_x_16gf": "https://download.pytorch.org/models/regnet_x_16gf-2007eb11.pth",
    "regnet_x_32gf": "https://download.pytorch.org/models/regnet_x_32gf-9d47f8d0.pth",
}


class SimpleStemIN(ConvNormActivation):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__(
            width_in, width_out, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=activation_layer
        )


class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
    ) -> None:
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        layers["a"] = ConvNormActivation(
            width_in, w_b, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=activation_layer
        )
        layers["b"] = ConvNormActivation(
            w_b, w_b, kernel_size=3, stride=stride, groups=g, norm_layer=norm_layer, activation_layer=activation_layer
        )

        if se_ratio:
            # The SE reduction ratio is defined with respect to the
            # beginning of the block
            width_se_out = int(round(se_ratio * width_in))
            layers["se"] = SqueezeExcitation(
                input_channels=w_b,
                squeeze_channels=width_se_out,
                activation=activation_layer,
            )

        layers["c"] = ConvNormActivation(
            w_b, width_out, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=None
        )
        super().__init__(layers)


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()

        # Use skip connection with projection if shape changes
        self.proj = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            self.proj = ConvNormActivation(
                width_in, width_out, kernel_size=1, stride=stride, norm_layer=norm_layer, activation_layer=None
            )
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )
        self.activation = activation_layer(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class AnyStage(nn.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        block_constructor: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float] = None,
        stage_index: int = 0,
    ) -> None:
        super().__init__()

        for i in range(depth):
            block = block_constructor(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                norm_layer,
                activation_layer,
                group_width,
                bottleneck_multiplier,
                se_ratio,
            )

            self.add_module(f"block{stage_index}-{i}", block)


class BlockParams:
    def __init__(
        self,
        depths: List[int],
        widths: List[int],
        group_widths: List[int],
        bottleneck_multipliers: List[float],
        strides: List[int],
        se_ratio: Optional[float] = None,
    ) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(
        cls,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> "BlockParams":
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

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")
        # Compute the block widths. Each stage has one unique block width
        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (torch.round(torch.divide(w_0 * torch.pow(w_m, block_capacity), QUANT)) * QUANT).int().tolist()
        num_stages = len(set(block_widths))

        # Convert to per stage parameters
        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = torch.diff(torch.tensor([d for d, t in enumerate(splits) if t])).int().tolist()

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def _get_expanded_params(self):
        return zip(self.widths, self.strides, self.depths, self.group_widths, self.bottleneck_multipliers)

    @staticmethod
    def _adjust_widths_groups_compatibilty(
        stage_widths: List[int], bottleneck_ratios: List[float], group_widths: List[int]
    ) -> Tuple[List[int], List[int]]:
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
    def __init__(
        self,
        block_params: BlockParams,
        num_classes: int = 1000,
        stem_width: int = 32,
        stem_type: Optional[Callable[..., nn.Module]] = None,
        block_type: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU

        # Ad hoc stem
        self.stem = stem_type(
            3,  # width_in
            stem_width,
            norm_layer,
            activation,
        )

        current_width = stem_width

        blocks = []
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(block_params._get_expanded_params()):
            blocks.append(
                (
                    f"block{i+1}",
                    AnyStage(
                        current_width,
                        width_out,
                        stride,
                        depth,
                        block_type,
                        norm_layer,
                        activation,
                        group_width,
                        bottleneck_multiplier,
                        block_params.se_ratio,
                        stage_index=i + 1,
                    ),
                )
            )

            current_width = width_out

        self.trunk_output = nn.Sequential(OrderedDict(blocks))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=current_width, out_features=num_classes)

        # Performs ResNet-style weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.trunk_output(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


def _regnet(arch: str, block_params: BlockParams, pretrained: bool, progress: bool, **kwargs: Any) -> RegNet:
    norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
    model = RegNet(block_params, norm_layer=norm_layer, **kwargs)
    if pretrained:
        if arch not in model_urls:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def regnet_y_400mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_400MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25, **kwargs)
    return _regnet("regnet_y_400mf", params, pretrained, progress, **kwargs)


def regnet_y_800mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_800MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=14, w_0=56, w_a=38.84, w_m=2.4, group_width=16, se_ratio=0.25, **kwargs)
    return _regnet("regnet_y_800mf", params, pretrained, progress, **kwargs)


def regnet_y_1_6gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_1.6GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=27, w_0=48, w_a=20.71, w_m=2.65, group_width=24, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_1_6gf", params, pretrained, progress, **kwargs)


def regnet_y_3_2gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_3.2GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=21, w_0=80, w_a=42.63, w_m=2.66, group_width=24, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_3_2gf", params, pretrained, progress, **kwargs)


def regnet_y_8gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_8GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=17, w_0=192, w_a=76.82, w_m=2.19, group_width=56, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_8gf", params, pretrained, progress, **kwargs)


def regnet_y_16gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_16GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=18, w_0=200, w_a=106.23, w_m=2.48, group_width=112, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_16gf", params, pretrained, progress, **kwargs)


def regnet_y_32gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_32GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=20, w_0=232, w_a=115.89, w_m=2.53, group_width=232, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_32gf", params, pretrained, progress, **kwargs)


def regnet_y_128gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_128GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.
    NOTE: Pretrained weights are not available for this model.
    """
    params = BlockParams.from_init_params(
        depth=27, w_0=456, w_a=160.83, w_m=2.52, group_width=264, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_128gf", params, pretrained, progress, **kwargs)


def regnet_x_400mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_400MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16, **kwargs)
    return _regnet("regnet_x_400mf", params, pretrained, progress, **kwargs)


def regnet_x_800mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_800MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=16, w_0=56, w_a=35.73, w_m=2.28, group_width=16, **kwargs)
    return _regnet("regnet_x_800mf", params, pretrained, progress, **kwargs)


def regnet_x_1_6gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_1.6GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24, **kwargs)
    return _regnet("regnet_x_1_6gf", params, pretrained, progress, **kwargs)


def regnet_x_3_2gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_3.2GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=25, w_0=88, w_a=26.31, w_m=2.25, group_width=48, **kwargs)
    return _regnet("regnet_x_3_2gf", params, pretrained, progress, **kwargs)


def regnet_x_8gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_8GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=23, w_0=80, w_a=49.56, w_m=2.88, group_width=120, **kwargs)
    return _regnet("regnet_x_8gf", params, pretrained, progress, **kwargs)


def regnet_x_16gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_16GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=22, w_0=216, w_a=55.59, w_m=2.1, group_width=128, **kwargs)
    return _regnet("regnet_x_16gf", params, pretrained, progress, **kwargs)


def regnet_x_32gf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetX_32GF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=23, w_0=320, w_a=69.86, w_m=2.0, group_width=168, **kwargs)
    return _regnet("regnet_x_32gf", params, pretrained, progress, **kwargs)


# TODO(kazhang): Add RegNetZ_500MF and RegNetZ_4GF
