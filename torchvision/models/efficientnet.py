import torch

from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Optional

from .._internally_replaced_utils import load_state_dict_from_url

# TODO: refactor this to a common place?
from torchvision.models.mobilenetv2 import ConvBNActivation, _make_divisible
from torchvision.models.mobilenetv3 import SqueezeExcitation


__all__ = []


model_urls = {}


class MBConvConfig:
    # TODO: Add dilation for supporting detection and segmentation pipelines
    def __init__(self,
                 kernel: int, stride: int,
                 input_channels: int, out_channels: int, expand_ratio: float, se_ratio: float,
                 skip: bool, width_mult: float):
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.expanded_channels = self.adjust_channels(input_channels, expand_ratio * width_mult)
        self.se_channels = self.adjust_channels(input_channels, se_ratio * width_mult, 1)
        self.skip = skip

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None):
        return _make_divisible(channels * width_mult, 8, min_value)


class MBConv(nn.Module):
    pass


class EfficientNet(nn.Module):
    pass


def efficientnet_b0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    pass
