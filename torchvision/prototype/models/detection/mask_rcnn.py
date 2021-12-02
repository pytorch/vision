from typing import Any, Optional

from torchvision.prototype.transforms import CocoEval
from torchvision.transforms.functional import InterpolationMode

from ....models.detection.mask_rcnn import (
    _resnet_fpn_extractor,
    _validate_trainable_layers,
    MaskRCNN,
    misc_nn_ops,
    overwrite_eps,
)
from .._api import WeightsEnum, Weights
from .._meta import _COCO_CATEGORIES
from .._utils import _deprecated_param, _deprecated_positional, _ovewrite_value_param
from ..resnet import ResNet50_Weights, resnet50


__all__ = [
    "MaskRCNN",
    "MaskRCNN_ResNet50_FPN_Weights",
    "maskrcnn_resnet50_fpn",
]


class MaskRCNN_ResNet50_FPN_Weights(WeightsEnum):
    Coco_V1 = Weights(
        url="https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
        transforms=CocoEval,
        meta={
            "categories": _COCO_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/detection#mask-r-cnn",
            "box_map": 37.9,
            "mask_map": 34.6,
        },
    )
    default = Coco_V1


def maskrcnn_resnet50_fpn(
    weights: Optional[MaskRCNN_ResNet50_FPN_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[ResNet50_Weights] = None,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> MaskRCNN:
    if type(weights) == bool and weights:
        _deprecated_positional(kwargs, "pretrained", "weights", True)
    if "pretrained" in kwargs:
        weights = _deprecated_param(kwargs, "pretrained", "weights", MaskRCNN_ResNet50_FPN_Weights.Coco_V1)
    weights = MaskRCNN_ResNet50_FPN_Weights.verify(weights)
    if type(weights_backbone) == bool and weights_backbone:
        _deprecated_positional(kwargs, "pretrained_backbone", "weights_backbone", True)
    if "pretrained_backbone" in kwargs:
        weights_backbone = _deprecated_param(
            kwargs, "pretrained_backbone", "weights_backbone", ResNet50_Weights.ImageNet1K_V1
        )
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
    model = MaskRCNN(backbone, num_classes=num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
        if weights == MaskRCNN_ResNet50_FPN_Weights.Coco_V1:
            overwrite_eps(model, 0.0)

    return model
