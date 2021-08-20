import torch

from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Callable, List, Optional

from .._internally_replaced_utils import load_state_dict_from_url
from torchvision.ops import stochastic_depth

# TODO: refactor this to a common place?
from torchvision.models.mobilenetv2 import ConvBNActivation, _make_divisible
from torchvision.models.mobilenetv3 import SqueezeExcitation


__all__ = ["EfficientNet"]


model_urls = {
    "efficientnet_b0": "",  # TODO: Add weights
}


class MBConvConfig:
    def __init__(self,
                 kernel: int, stride: int, dilation: int,
                 input_channels: int, out_channels: int, expand_ratio: float,
                 width_mult: float) -> None:
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.expanded_channels = self.adjust_channels(input_channels, expand_ratio * width_mult)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)


class MBConv(nn.Module):
    def __init__(self, cnf: MBConvConfig, norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = SqueezeExcitation) -> None:
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                       stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
                                       norm_layer=norm_layer, activation_layer=activation_layer))

        # squeeze and excitation
        layers.append(se_layer(cnf.expanded_channels, min_value=1, activation_fn=F.sigmoid))

        # project
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor, drop_rate: float = 0.0) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = stochastic_depth(result, drop_rate, "row", training=self.training)
            result += input
        return result


class EfficientNet(nn.Module):
    pass


def efficientnet_b0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    pass
