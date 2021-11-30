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
from .._api import WeightsEnum, Weights
from .._meta import _COCO_CATEGORIES
from .._utils import handle_legacy_interface, _ovewrite_value_param
from ..mobilenetv3 import MobileNet_V3_Large_Weights, mobilenet_v3_large
from ..resnet import ResNet50_Weights, resnet50


__all__ = [
    "FasterRCNN",
    "FasterRCNN_ResNet50_FPN_Weights",
    "FasterRCNN_MobileNet_V3_Large_FPN_Weights",
    "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights",
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
]


_COMMON_META = {
    "categories": _COCO_CATEGORIES,
    "interpolation": InterpolationMode.BILINEAR,
}


class FasterRCNN_ResNet50_FPN_Weights(WeightsEnum):
    Coco_V1 = Weights(
        url="https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
        transforms=CocoEval,
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/detection#faster-r-cnn-resnet-50-fpn",
            "map": 37.0,
        },
    )
    default = Coco_V1


class FasterRCNN_MobileNet_V3_Large_FPN_Weights(WeightsEnum):
    Coco_V1 = Weights(
        url="https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth",
        transforms=CocoEval,
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/detection#faster-r-cnn-mobilenetv3-large-fpn",
            "map": 32.8,
        },
    )
    default = Coco_V1


class FasterRCNN_MobileNet_V3_Large_320_FPN_Weights(WeightsEnum):
    Coco_V1 = Weights(
        url="https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth",
        transforms=CocoEval,
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/detection#faster-r-cnn-mobilenetv3-large-320-fpn",
            "map": 22.8,
        },
    )
    default = Coco_V1


@handle_legacy_interface(
    weights=("pretrained", FasterRCNN_ResNet50_FPN_Weights.Coco_V1),
    weights_backbone=("pretrained_backbone", ResNet50_Weights.ImageNet1K_V1),
)
def fasterrcnn_resnet50_fpn(
    *,
    weights: Optional[FasterRCNN_ResNet50_FPN_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[ResNet50_Weights] = None,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> FasterRCNN:
    weights = FasterRCNN_ResNet50_FPN_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    trainable_backbone_layers = _validate_trainable_layers(
        weights is not None or weights_backbone is not None, trainable_backbone_layers, 5, 3
    )

    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes=num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
        if weights == FasterRCNN_ResNet50_FPN_Weights.Coco_V1:
            overwrite_eps(model, 0.0)

    return model


def _fasterrcnn_mobilenet_v3_large_fpn(
    *,
    weights: Optional[Union[FasterRCNN_MobileNet_V3_Large_FPN_Weights, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights]],
    progress: bool,
    num_classes: Optional[int],
    weights_backbone: Optional[MobileNet_V3_Large_Weights],
    trainable_backbone_layers: Optional[int],
    **kwargs: Any,
) -> FasterRCNN:
    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

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


@handle_legacy_interface(
    weights=("pretrained", FasterRCNN_MobileNet_V3_Large_FPN_Weights.Coco_V1),
    weights_backbone=("pretrained_backbone", MobileNet_V3_Large_Weights.ImageNet1K_V1),
)
def fasterrcnn_mobilenet_v3_large_fpn(
    *,
    weights: Optional[FasterRCNN_MobileNet_V3_Large_FPN_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[MobileNet_V3_Large_Weights] = None,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> FasterRCNN:
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)

    defaults = {
        "rpn_score_thresh": 0.05,
    }

    kwargs = {**defaults, **kwargs}
    return _fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        progress=progress,
        num_classes=num_classes,
        weights_backbone=weights_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs,
    )


@handle_legacy_interface(
    weights=("pretrained", FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.Coco_V1),
    weights_backbone=("pretrained_backbone", MobileNet_V3_Large_Weights.ImageNet1K_V1),
)
def fasterrcnn_mobilenet_v3_large_320_fpn(
    *,
    weights: Optional[FasterRCNN_MobileNet_V3_Large_320_FPN_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[MobileNet_V3_Large_Weights] = None,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> FasterRCNN:

    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)

    defaults = {
        "min_size": 320,
        "max_size": 640,
        "rpn_pre_nms_top_n_test": 150,
        "rpn_post_nms_top_n_test": 150,
        "rpn_score_thresh": 0.05,
    }

    kwargs = {**defaults, **kwargs}
    return _fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        progress=progress,
        num_classes=num_classes,
        weights_backbone=weights_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs,
    )
