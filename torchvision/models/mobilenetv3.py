from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional


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

    def __init__(self, input_channels: int, squeeze_factor: int):
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


class InvertedResidualBottleneck(nn.Module):

    def __init__(self, input_channels: int, kernel: int, expanded_channels: int, output_channels: int,
                 se_block: bool, activation: str, stride: int):
        super().__init__()
        self.shortcut = stride == 1 and input_channels == output_channels

        self.block = nn.Sequential()
        if expanded_channels != input_channels:
            self.block.add_module("expand_conv", nn.Conv2d(input_channels, expanded_channels, 1, bias=False))
            self._add_bn_act("expand", expanded_channels, activation)

        self.block.add_module("depthwise_conv", nn.Conv2d(expanded_channels, expanded_channels, kernel, stride=stride,
                                                          padding=(kernel - 1) // 2, groups=expanded_channels,
                                                          bias=False))
        self._add_bn_act("depthwise", expanded_channels, activation)

        if se_block:
            self.block.add_module("squeeze_excitation", SqueezeExcitation(expanded_channels, 4))

        self.block.add_module("project_conv", nn.Conv2d(expanded_channels, output_channels, 1, bias=False))
        self._add_bn_act("project", expanded_channels, None)

    def _add_bn_act(self, block_name: str, channels: int, activation: Optional[str]):
        self.block.add_module("{}_bn".format(block_name), nn.BatchNorm2d(channels, momentum=0.01, eps=0.001))
        if activation == "RE":
            self.block.add_module("{}_act".format(block_name), nn.ReLU(inplace=True))
        elif activation == "HS":
            self.block.add_module("{}_act".format(block_name), HardSwish(inplace=True))

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.shortcut:
            result += input
        return result


class MobileNetV3(nn.Module):

    def __init__(self,

                 num_classes: int = 1000):


        #TODO: initialize weights
