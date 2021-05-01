import torch

from functools import partial
from torch import nn, Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple

from .ssd import SSDScoringHead
from ..mobilenetv3 import ConvBNActivation


__all__ = []

model_urls = {
    # TODO: add weights
}


def _prediction_block(in_channels: int, out_channels: int, kernel_size: int,
                      norm_layer: Callable[..., nn.Module]) -> nn.Sequential:
    return nn.Sequential(
        # 3x3 depthwise with stride 1 and padding 1
        ConvBNActivation(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels,
                         norm_layer=norm_layer, activation_layer=nn.ReLU),

        # 1x1 projetion to output channels
        nn.Conv2d(in_channels, out_channels, 1)
    )


def _extra_block(in_channels: int, out_channels: int, norm_layer: Callable[..., nn.Module]) -> nn.Sequential:
    activation = nn.ReLU
    intermediate_channels = out_channels // 2
    return nn.Sequential(
        # 1x1 projection to half output channels
        ConvBNActivation(in_channels, intermediate_channels, kernel_size=1,
                         norm_layer=norm_layer, activation_layer=activation),

        # 3x3 depthwise with stride 2 and padding 1
        ConvBNActivation(intermediate_channels, intermediate_channels, kernel_size=3, stride=2,
                         groups=intermediate_channels, norm_layer=norm_layer, activation_layer=activation),

        # 1x1 projetion to output channels
        ConvBNActivation(intermediate_channels, out_channels, kernel_size=1,
                         norm_layer=norm_layer, activation_layer=activation),
    )


def _normal_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.03)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)


class SSDLiteHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int,
                 norm_layer: Callable[..., nn.Module]):
        super().__init__()
        self.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer)
        self.regression_head = SSDLiteRegressionHead(in_channels, num_anchors, norm_layer)

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        return {
            'bbox_regression': self.regression_head(x),
            'cls_logits': self.classification_head(x),
        }


class SSDLiteClassificationHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int,
                 norm_layer: Callable[..., nn.Module]):
        cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            cls_logits.append(_prediction_block(channels, num_classes * anchors, 3, norm_layer))
        _normal_init(cls_logits)
        super().__init__(cls_logits, num_classes)


class SSDLiteRegressionHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], norm_layer: Callable[..., nn.Module]):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(_prediction_block(channels, 4 * anchors, 3, norm_layer))
        _normal_init(bbox_reg)
        super().__init__(bbox_reg, 4)


class SSDLiteFeatureExtractorMobileNet(nn.Module):
    def __init__(self, backbone: nn.Module, norm_layer: Callable[..., nn.Module], width_mult: float = 1.0,
                 min_depth: int = 16):
        super().__init__()

        self.features = None  # TODO: fix this

        get_depth = lambda d: max(min_depth, int(d * width_mult))
        extra = nn.ModuleList([
            _extra_block(backbone[-1].out_channels, get_depth(512), norm_layer),
            _extra_block(get_depth(512), get_depth(256), norm_layer),
            _extra_block(get_depth(256), get_depth(256), norm_layer),
            _extra_block(get_depth(256), get_depth(128), norm_layer),
        ])
        _normal_init(extra)

        self.extra = extra

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        pass # TODO: fix this


def _mobilenet_extractor(backbone_name: str, progress: bool, pretrained: bool, trainable_layers: int,
                         norm_layer: Callable[..., nn.Module]):
    pass


def ssd320_mobilenet_v3_large(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                              pretrained_backbone: bool = True, trainable_backbone_layers: Optional[int] = None,
                              norm_layer: Optional[Callable[..., nn.Module]] = None,
                              **kwargs: Any):

    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    pass # TODO: add this
