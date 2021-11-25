from typing import Any, Optional

from torchvision.prototype.transforms import CocoEval
from torchvision.transforms.functional import InterpolationMode

from ....models.detection.retinanet import (
    _resnet_fpn_extractor,
    _validate_trainable_layers,
    RetinaNet,
    LastLevelP6P7,
    misc_nn_ops,
    overwrite_eps,
)
from .._api import Weights, WeightEntry
from .._meta import _COCO_CATEGORIES
from .._utils import _deprecated_param, _ovewrite_value_param
from ..resnet import ResNet50Weights, resnet50


__all__ = [
    "RetinaNet",
    "RetinaNetResNet50FPNWeights",
    "retinanet_resnet50_fpn",
]


class RetinaNetResNet50FPNWeights(Weights):
    Coco_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth",
        transforms=CocoEval,
        meta={
            "categories": _COCO_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/detection#retinanet",
            "map": 36.4,
        },
    )


def retinanet_resnet50_fpn(
    weights: Optional[RetinaNetResNet50FPNWeights] = None,
    weights_backbone: Optional[ResNet50Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> RetinaNet:
    if "pretrained" in kwargs:
        weights = _deprecated_param(kwargs, "pretrained", "weights", RetinaNetResNet50FPNWeights.Coco_RefV1)
    weights = RetinaNetResNet50FPNWeights.verify(weights)
    if "pretrained_backbone" in kwargs:
        weights_backbone = _deprecated_param(
            kwargs, "pretrained_backbone", "weights_backbone", ResNet50Weights.ImageNet1K_RefV1
        )
    weights_backbone = ResNet50Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    trainable_backbone_layers = _validate_trainable_layers(
        weights is not None or weights_backbone is not None, trainable_backbone_layers, 5, 3
    )

    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = _resnet_fpn_extractor(
        backbone, trainable_backbone_layers, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
    )
    model = RetinaNet(backbone, num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
        if weights == RetinaNetResNet50FPNWeights.Coco_RefV1:
            overwrite_eps(model, 0.0)

    return model
