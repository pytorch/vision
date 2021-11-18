import warnings
from typing import Any, Optional, Union

from torchvision.prototype.transforms import CocoEval
from torchvision.transforms.functional import InterpolationMode

from ....models.detection.faster_rcnn import (
    _mobilenet_extractor,
    _resnet_fpn_extractor,
    _validate_trainable_layers,
    AnchorGenerator,
    FasterRCNN,
    misc_nn_ops,
    overwrite_eps,
)
from .._api import Weights, WeightEntry
from .._meta import _COCO_CATEGORIES
from ..mobilenetv3 import MobileNetV3LargeWeights, mobilenet_v3_large
from ..resnet import ResNet50Weights, resnet50


__all__ = [
    "FasterRCNN",
    "FasterRCNNResNet50FPNWeights",
    "FasterRCNNMobileNetV3LargeFPNWeights",
    "FasterRCNNMobileNetV3Large320FPNWeights",
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
]


_COMMON_META = {
    "categories": _COCO_CATEGORIES,
    "interpolation": InterpolationMode.BILINEAR,
}


class FasterRCNNResNet50FPNWeights(Weights):
    Coco_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
        transforms=CocoEval,
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/detection#faster-r-cnn-resnet-50-fpn",
            "map": 37.0,
        },
    )


class FasterRCNNMobileNetV3LargeFPNWeights(Weights):
    Coco_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth",
        transforms=CocoEval,
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/detection#faster-r-cnn-mobilenetv3-large-fpn",
            "map": 32.8,
        },
    )


class FasterRCNNMobileNetV3Large320FPNWeights(Weights):
    Coco_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth",
        transforms=CocoEval,
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/detection#faster-r-cnn-mobilenetv3-large-320-fpn",
            "map": 22.8,
        },
    )


def fasterrcnn_resnet50_fpn(
    weights: Optional[FasterRCNNResNet50FPNWeights] = None,
    weights_backbone: Optional[ResNet50Weights] = None,
    progress: bool = True,
    num_classes: int = 91,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> FasterRCNN:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = FasterRCNNResNet50FPNWeights.Coco_RefV1 if kwargs.pop("pretrained") else None
    weights = FasterRCNNResNet50FPNWeights.verify(weights)
    if "pretrained_backbone" in kwargs:
        warnings.warn("The parameter pretrained_backbone is deprecated, please use weights_backbone instead.")
        weights_backbone = ResNet50Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained_backbone") else None
    weights_backbone = ResNet50Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = len(weights.meta["categories"])

    trainable_backbone_layers = _validate_trainable_layers(
        weights is not None or weights_backbone is not None, trainable_backbone_layers, 5, 3
    )

    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes=num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
        if weights == FasterRCNNResNet50FPNWeights.Coco_RefV1:
            overwrite_eps(model, 0.0)

    return model


def _fasterrcnn_mobilenet_v3_large_fpn(
    weights: Optional[Union[FasterRCNNMobileNetV3LargeFPNWeights, FasterRCNNMobileNetV3Large320FPNWeights]] = None,
    weights_backbone: Optional[MobileNetV3LargeWeights] = None,
    progress: bool = True,
    num_classes: int = 91,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> FasterRCNN:
    if weights is not None:
        weights_backbone = None
        num_classes = len(weights.meta["categories"])

    trainable_backbone_layers = _validate_trainable_layers(
        weights is not None or weights_backbone is not None, trainable_backbone_layers, 6, 3
    )

    backbone = mobilenet_v3_large(weights=weights_backbone, progress=progress, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    backbone = _mobilenet_extractor(backbone, True, trainable_backbone_layers)
    anchor_sizes = (
        (
            32,
            64,
            128,
            256,
            512,
        ),
    ) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    model = FasterRCNN(
        backbone, num_classes, rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios), **kwargs
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def fasterrcnn_mobilenet_v3_large_fpn(
    weights: Optional[FasterRCNNMobileNetV3LargeFPNWeights] = None,
    weights_backbone: Optional[MobileNetV3LargeWeights] = None,
    progress: bool = True,
    num_classes: int = 91,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> FasterRCNN:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = FasterRCNNMobileNetV3LargeFPNWeights.Coco_RefV1 if kwargs.pop("pretrained") else None
    weights = FasterRCNNMobileNetV3LargeFPNWeights.verify(weights)
    if "pretrained_backbone" in kwargs:
        warnings.warn("The parameter pretrained_backbone is deprecated, please use weights_backbone instead.")
        weights_backbone = MobileNetV3LargeWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained_backbone") else None
    weights_backbone = MobileNetV3LargeWeights.verify(weights_backbone)

    defaults = {
        "rpn_score_thresh": 0.05,
    }

    kwargs = {**defaults, **kwargs}
    return _fasterrcnn_mobilenet_v3_large_fpn(
        weights,
        weights_backbone,
        progress,
        num_classes,
        trainable_backbone_layers,
        **kwargs,
    )


def fasterrcnn_mobilenet_v3_large_320_fpn(
    weights: Optional[FasterRCNNMobileNetV3Large320FPNWeights] = None,
    weights_backbone: Optional[MobileNetV3LargeWeights] = None,
    progress: bool = True,
    num_classes: int = 91,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> FasterRCNN:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = FasterRCNNMobileNetV3Large320FPNWeights.Coco_RefV1 if kwargs.pop("pretrained") else None
    weights = FasterRCNNMobileNetV3Large320FPNWeights.verify(weights)
    if "pretrained_backbone" in kwargs:
        warnings.warn("The parameter pretrained_backbone is deprecated, please use weights_backbone instead.")
        weights_backbone = MobileNetV3LargeWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained_backbone") else None
    weights_backbone = MobileNetV3LargeWeights.verify(weights_backbone)

    defaults = {
        "min_size": 320,
        "max_size": 640,
        "rpn_pre_nms_top_n_test": 150,
        "rpn_post_nms_top_n_test": 150,
        "rpn_score_thresh": 0.05,
    }

    kwargs = {**defaults, **kwargs}
    return _fasterrcnn_mobilenet_v3_large_fpn(
        weights,
        weights_backbone,
        progress,
        num_classes,
        trainable_backbone_layers,
        **kwargs,
    )
