from functools import partial
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Callable, List, Optional


def _make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
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
        squeeze_channels = _make_divisible(input_channels // squeeze_factor)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def forward(self, input: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = hard_sigmoid(scale, inplace=True)
        return scale * input


class InvertedResidual(nn.Module):

    def __init__(self, input_channels: int, kernel: int, expanded_channels: int, output_channels: int,
                 use_se: bool, use_hs: bool, stride: int, norm_layer: Callable[..., nn.Module]):
        super().__init__()
        assert stride in [1, 2]

        self.use_res_connect = stride == 1 and input_channels == output_channels

        layers: List[nn.Module] = []
        # expand
        if expanded_channels != input_channels:
            layers.extend([
                nn.Conv2d(input_channels, expanded_channels, 1, bias=False),
                norm_layer(expanded_channels),
                HardSwish(inplace=True) if use_hs else nn.ReLU(inplace=True),
            ])

        # depthwise
        layers.extend([
            nn.Conv2d(expanded_channels, expanded_channels, kernel, stride=stride, padding=(kernel - 1) // 2,
                      groups=expanded_channels, bias=False),
            norm_layer(expanded_channels),
            HardSwish(inplace=True) if use_hs else nn.ReLU(inplace=True),
        ])
        if use_se:
            layers.append(SqueezeExcitation(expanded_channels))

        # project
        layers.extend([
            nn.Conv2d(expanded_channels, output_channels, 1, bias=False),
            norm_layer(expanded_channels),
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
            inverted_residual_setting: List[List[int]],
            last_channel: int,
            num_classes: int = 1000,
            blocks: Optional[List[Callable[..., nn.Module]]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()

        if blocks is None:
            blocks = [SqueezeExcitation, InvertedResidual]
        se_layer, bottleneck_layer = blocks

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = [

        ]



        pass
        # TODO: initialize weights


def mobilenetv3(mode: str = "large", width_mult: float = 1.0):
    if mode == "large":
        inverted_residual_setting = [
            # in, kernel, exp, out, use_se, use_hs, stride
            [16, 3, 16, 16, 0, 0, 1],
            [16, 3, 64, 24, 0, 0, 2],
            [24, 3, 72, 24, 0, 0, 1],
            [24, 5, 72, 40, 1, 0, 2],
            [40, 5, 120, 40, 1, 0, 1],
            [40, 5, 120, 40, 1, 0, 1],
            [40, 3, 240, 80, 0, 1, 2],
            [80, 3, 200, 80, 0, 1, 1],
            [80, 3, 184, 80, 0, 1, 1],
            [80, 3, 184, 80, 0, 1, 1],
            [80, 3, 480, 112, 1, 1, 1],
            [112, 3, 672, 112, 1, 1, 1],
            [112, 5, 672, 160, 1, 1, 2],
            [160, 5, 960, 160, 1, 1, 1],
            [160, 5, 960, 160, 1, 1, 1],
        ]
        last_channel = 1280
    else:
        inverted_residual_setting = [
            # in, kernel, exp, out, use_se, use_hs, stride
            [16, 3, 16, 16, 1, 0, 2],
            [16, 3, 72, 24, 0, 0, 2],
            [24, 3, 88, 24, 0, 0, 1],
            [24, 5, 96, 40, 1, 1, 2],
            [40, 5, 240, 40, 1, 1, 1],
            [40, 5, 240, 40, 1, 1, 1],
            [40, 5, 120, 48, 1, 1, 1],
            [48, 5, 144, 48, 1, 1, 1],
            [48, 5, 288, 96, 1, 1, 2],
            [96, 5, 576, 96, 1, 1, 1],
            [96, 5, 576, 96, 1, 1, 1],
        ]
        last_channel = 1024

    # apply multipler on: in, exp, out columns
    for row in inverted_residual_setting:
        for id in (0, 2, 3):
            row[id] = _make_divisible(row[id] * width_mult)
    last_channel = _make_divisible(last_channel * width_mult)

    return MobileNetV3(inverted_residual_setting, last_channel)
