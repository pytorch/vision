import torch

from functools import partial
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Callable, List, Optional

from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.mobilenetv2 import _make_divisible, ConvBNActivation


__all__ = ["MobileNetV3", "mobilenet_v3_large", "mobilenet_v3_small"]


# TODO: add pretrained
model_urls = {
    "mobilenet_v3_large_1_0": None,
    "mobilenet_v3_small_1_0": None,
}


class _InplaceActivation(nn.Module):

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'inplace=True' if self.inplace else ''


class Identity(_InplaceActivation):

    def forward(self, input: Tensor) -> Tensor:
        return input


def hard_sigmoid(x: Tensor, inplace: bool = False) -> Tensor:
    return F.relu6(x + 3.0, inplace=inplace) / 6.0


class HardSigmoid(_InplaceActivation):

    def forward(self, input: Tensor) -> Tensor:
        return hard_sigmoid(input, inplace=self.inplace)


def hard_swish(x: Tensor, inplace: bool = False) -> Tensor:
    return x * hard_sigmoid(x, inplace=inplace)


class HardSwish(_InplaceActivation):

    def forward(self, input: Tensor) -> Tensor:
        return hard_swish(input, inplace=self.inplace)


class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def forward(self, input: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = hard_sigmoid(scale, inplace=True)
        return scale * input


class InvertedResidualConfig:

    def __init__(self, input_channels: int, kernel: int, expanded_channels: int, output_channels: int, use_se: bool,
                 activation: str, stride: int, width_mult: float):
        self.input_channels = _make_divisible(input_channels * width_mult, 8)
        self.kernel = kernel
        self.expanded_channels = _make_divisible(expanded_channels * width_mult, 8)
        self.output_channels = _make_divisible(output_channels * width_mult, 8)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride


class InvertedResidual(nn.Module):

    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module]):
        super().__init__()
        assert cnf.stride in [1, 2]

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.output_channels

        layers = []
        activation_layer = HardSwish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                       stride=cnf.stride, groups=cnf.expanded_channels, norm_layer=norm_layer,
                                       activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_channels))

        # project
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.output_channels, kernel_size=1, norm_layer=norm_layer,
                                       activation_layer=Identity))

        self.block = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(nn.Module):

    def __init__(
            self,
            inverted_residual_setting: List[InvertedResidualConfig],
            last_channel: int,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers = [ConvBNActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                                   activation_layer=HardSwish)]

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].output_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvBNActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                       norm_layer=norm_layer, activation_layer=HardSwish))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            HardSwish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _mobilenet_v3(
    arch: str,
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
):
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def mobilenet_v3_large(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    width_mult = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    lastchannel_conf = lambda c: _make_divisible(c * width_mult, 8)

    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160, True, "HS", 2),
        bneck_conf(160, 5, 960, 160, True, "HS", 1),
        bneck_conf(160, 5, 960, 160, True, "HS", 1),
    ]
    last_channel = lastchannel_conf(1280)

    return _mobilenet_v3("mobilenet_v3_large_1_0", inverted_residual_setting, last_channel, pretrained, progress,
                         **kwargs)


def mobilenet_v3_small(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    width_mult = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    lastchannel_conf = lambda c: _make_divisible(c * width_mult, 8)

    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, True, "RE", 2),
        bneck_conf(16, 3, 72, 24, False, "RE", 2),
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96, True, "HS", 2),
        bneck_conf(96, 5, 576, 96, True, "HS", 1),
        bneck_conf(96, 5, 576, 96, True, "HS", 1),
    ]
    last_channel = lastchannel_conf(1024)

    return _mobilenet_v3("mobilenet_v3_small_1_0", inverted_residual_setting, last_channel, pretrained, progress,
                         **kwargs)
