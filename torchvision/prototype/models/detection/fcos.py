from typing import Any, Optional

from torch import nn
from torchvision.prototype.transforms import ObjectDetectionEval
from torchvision.transforms.functional import InterpolationMode

from ....models.detection.fcos import (
    _resnet_fpn_extractor,
    _validate_trainable_layers,
    FCOS,
    LastLevelP6P7,
    misc_nn_ops,
)
from .._api import WeightsEnum, Weights
from .._meta import _COCO_CATEGORIES
from .._utils import handle_legacy_interface, _ovewrite_value_param
from ..resnet import ResNet50_Weights, resnet50


__all__ = [
    "FCOS",
    "FCOS_ResNet50_FPN_Weights",
    "fcos_resnet50_fpn",
]


class FCOS_ResNet50_FPN_Weights(WeightsEnum):
    COCO_V1 = Weights(
        url="https://download.pytorch.org/models/fcos_resnet50_fpn_coco-99b0c9b7.pth",
        transforms=ObjectDetectionEval,
        meta={
            "task": "image_object_detection",
            "architecture": "FCOS",
            "publication_year": 2019,
            "num_params": 32269600,
            "categories": _COCO_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/detection#fcos-resnet-50-fpn",
            "map": 39.2,
        },
    )
    DEFAULT = COCO_V1


@handle_legacy_interface(
    weights=("pretrained", FCOS_ResNet50_FPN_Weights.COCO_V1),
    weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1),
)
def fcos_resnet50_fpn(
    *,
    weights: Optional[FCOS_ResNet50_FPN_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[ResNet50_Weights] = None,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> FCOS:
    weights = FCOS_ResNet50_FPN_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(
        backbone, trainable_backbone_layers, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
    )
    model = FCOS(backbone, num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
