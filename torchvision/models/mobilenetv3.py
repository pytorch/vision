from functools import partial
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Callable, List, Optional


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class _InplaceActivation(nn.Module):

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'inplace=True' if self.inplace else ''


def hard_sigmoid(x: Tensor, inplace: bool = False) -> Tensor:
    return F.relu6(x + 3.0, inplace=inplace) / 6.0


class HardSigmoid(_InplaceActivation):

    def forward(self, input: Tensor) -> Tensor:
        return hard_sigmoid(input, inplace=self.inplace)


def hard_swish(x: Tensor, inplace: bool = False) -> Tensor:
    return x * hard_swish(x, inplace=inplace)


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
        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.extend([
                nn.Conv2d(cnf.input_channels, cnf.expanded_channels, 1, bias=False),
                norm_layer(cnf.expanded_channels),
                HardSwish(inplace=True) if cnf.use_hs else nn.ReLU(inplace=True),
            ])

        # depthwise
        layers.extend([
            nn.Conv2d(cnf.expanded_channels, cnf.expanded_channels, cnf.kernel, stride=cnf.stride,
                      padding=(cnf.kernel - 1) // 2, groups=cnf.expanded_channels, bias=False),
            norm_layer(cnf.expanded_channels),
            HardSwish(inplace=True) if cnf.use_hs else nn.ReLU(inplace=True),
        ])
        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_channels))

        # project
        layers.extend([
            nn.Conv2d(cnf.expanded_channels, cnf.output_channels, 1, bias=False),
            norm_layer(cnf.expanded_channels),
        ])

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
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers = [
            nn.Conv2d(3, firstconv_output_channels, 3, stride=2, padding=1, bias=False),
            norm_layer(firstconv_output_channels),
            HardSwish(inplace=True),
        ]

        pass
        # TODO: initialize weights


def mobilenetv3(mode: str = "large", width_mult: float = 1.0):
    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    if mode == "large":
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
        last_channel = 1280
    else:
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
        last_channel = 1024

    last_channel = _make_divisible(last_channel * width_mult, 8)

    return MobileNetV3(inverted_residual_setting, last_channel)
