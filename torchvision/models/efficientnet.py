import torch

from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any

from .._internally_replaced_utils import load_state_dict_from_url

# TODO: refactor this to a common place?
from torchvision.models.mobilenetv2 import ConvBNActivation
from torchvision.models.mobilenetv3 import SqueezeExcitation


class MBConvConfig:
    pass


class MBConv(nn.Module):
    pass


class EfficientNet(nn.Module):
    pass


def efficientnet_b0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    pass
