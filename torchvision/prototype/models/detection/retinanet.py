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
from .._api import WeightsEnum, Weights
from .._meta import _COCO_CATEGORIES
from .._utils import handle_legacy_interface, _ovewrite_value_param
from ..resnet import ResNet50_Weights, resnet50


__all__ = [
    "RetinaNet",
    "RetinaNet_ResNet50_FPN_Weights",
    "retinanet_resnet50_fpn",
]


class RetinaNet_ResNet50_FPN_Weights(WeightsEnum):
    Coco_V1 = Weights(
        url="https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth",
        transforms=CocoEval,
        meta={
            "categories": _COCO_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/detection#retinanet",
            "map": 36.4,
        },
    )
    default = Coco_V1


@handle_legacy_interface(
    weights=("pretrained", RetinaNet_ResNet50_FPN_Weights.Coco_V1),
    weights_backbone=("pretrained_backbone", ResNet50_Weights.ImageNet1K_V1),
)
def retinanet_resnet50_fpn(
    *,
    weights: Optional[RetinaNet_ResNet50_FPN_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[ResNet50_Weights] = None,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> RetinaNet:
    weights = RetinaNet_ResNet50_FPN_Weights.verify(weights)
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
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = _resnet_fpn_extractor(
        backbone, trainable_backbone_layers, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
    )
    model = RetinaNet(backbone, num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
        if weights == RetinaNet_ResNet50_FPN_Weights.Coco_V1:
            overwrite_eps(model, 0.0)

    return model
