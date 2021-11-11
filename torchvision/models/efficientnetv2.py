import copy
import math
from functools import partial
from typing import Any, Callable, Optional, List, Sequence

import torch
from torch import nn, Tensor

from ._utils import _make_divisible
from .._internally_replaced_utils import load_state_dict_from_url
from ..ops import StochasticDepth
from .efficientnet import MBConv, ConvNormActivation


__all__ = [
    "EfficientNetV2", 
    "efficientnet_v2_s", # 384
    "efficientnet_v2_m", # 480 
    "efficientnet_v2_l", # 480 
]


model_urls = {
    # Weights ported from https://github.com/rwightman/pytorch-image-models/
    "efficientnet_v2_s": "",
    'efficientnet_v2_m': "",
    'efficientnet_v2_l': ""
}


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(
        self,
        block_type: str,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
    ) -> None:
        self.block_type = block_type
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "block_type={block_type}"
        s += "expand_ratio={expand_ratio}"
        s += ", kernel={kernel}"
        s += ", stride={stride}"
        s += ", input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class FusedMBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = None,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        if se_layer:
            # squeeze and excitation
            squeeze_channels = max(1, cnf.input_channels // 4)
            layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            ConvNormActivation(
                expanded_channels,
                cnf.out_channels,
                kernel_size=1 if expanded_channels != cnf.input_channels else cnf.kernel,
                stride=1 if expanded_channels != cnf.input_channels else cnf.stride,
                norm_layer=norm_layer,
                activation_layer=None if expanded_channels != cnf.input_channels else activation_layer,
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNetV2(nn.Module):
    def __init__(
        self,
        block_setting: List[MBConvConfig],
        dropout: float,
        lastconv_output_channels: int = 1280,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        """
        EfficientNetV2 main class
        Args:
            block_setting (List): Network structure
            dropout (float): The droupout probability
            lastconv_output_channels (int): the output channels of last conv layer
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (
            isinstance(block_setting, Sequence)
            and all([isinstance(s, MBConvConfig) for s in block_setting])
        ):
            raise TypeError("The block_setting should be List[MBConvConfig]")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
            )
        )

        # building blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            block = MBConv if cnf.block_type == 'MB' else FusedMBConv
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = block_setting[-1].out_channels
        if lastconv_output_channels is None:
            lastconv_output_channels = 4 * lastconv_input_channels
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _efficientnet_v2(
    arch: str,
    block_setting,
    dropout: float,
    lastconv_output_channels: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> EfficientNetV2:
    
    model = EfficientNetV2(block_setting, dropout, lastconv_output_channels=lastconv_output_channels, **kwargs)
    if pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def efficientnet_v2_s(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNetV2:
    """
    Constructs a EfficientNetV2-S architecture from
    `"EfficientNetV2: Smaller Models and Faster Training" <https://arxiv.org/abs/2104.00298>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    block_setting = [
        MBConvConfig('FusedMB', 1, 3, 1, 24, 24, 2),
        MBConvConfig('FusedMB', 4, 3, 2, 24, 48, 4),
        MBConvConfig('FusedMB', 4, 3, 2, 48, 64, 4),
        MBConvConfig('MB', 4, 3, 2, 64, 128, 6),
        MBConvConfig('MB', 6, 3, 1, 128, 160, 9),
        MBConvConfig('MB', 6, 3, 2, 160, 256, 15)
    ]
    return _efficientnet_v2("efficientnet_v2_s", block_setting, 0., 1280, pretrained, progress, **kwargs)


def efficientnet_v2_m(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNetV2:
    """
    Constructs a EfficientNetV2-M architecture from
    `"EfficientNetV2: Smaller Models and Faster Training" <https://arxiv.org/abs/2104.00298>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    block_setting = [
        MBConvConfig('FusedMB', 1, 3, 1, 24, 24, 3),
        MBConvConfig('FusedMB', 4, 3, 2, 24, 48, 5),
        MBConvConfig('FusedMB', 4, 3, 2, 48, 80, 5),
        MBConvConfig('MB', 4, 3, 2, 80, 160, 7),
        MBConvConfig('MB', 6, 3, 1, 160, 176, 14),
        MBConvConfig('MB', 6, 3, 2, 176, 304, 18),
        MBConvConfig('MB', 6, 3, 1, 304, 512, 5)
    ]
    return _efficientnet_v2("efficientnet_v2_m", block_setting, 0.2, 1280, pretrained, progress, **kwargs)


def efficientnet_v2_l(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNetV2:
    """
    Constructs a EfficientNetV2-L architecture from
    `"EfficientNetV2: Smaller Models and Faster Training" <https://arxiv.org/abs/2104.00298>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    block_setting = [
        MBConvConfig('FusedMB', 1, 3, 1, 32, 32, 4),
        MBConvConfig('FusedMB', 4, 3, 2, 32, 64, 7),
        MBConvConfig('FusedMB', 4, 3, 2, 64, 96, 7),
        MBConvConfig('MB', 4, 3, 2, 96, 192, 10),
        MBConvConfig('MB', 6, 3, 1, 192, 224, 19),
        MBConvConfig('MB', 6, 3, 2, 224, 384, 25),
        MBConvConfig('MB', 6, 3, 1, 384, 640, 7),
    ]
    return _efficientnet_v2("efficientnet_v2_l", block_setting, 0.5, 1280, pretrained, progress, **kwargs)
