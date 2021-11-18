import warnings
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
from .._api import Weights, WeightEntry
from .._meta import _COCO_CATEGORIES
from ..resnet import ResNet50Weights, resnet50


__all__ = [
    "MaskRCNN",
    "MaskRCNNResNet50FPNWeights",
    "maskrcnn_resnet50_fpn",
]


class MaskRCNNResNet50FPNWeights(Weights):
    Coco_RefV1 = WeightEntry(
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


def maskrcnn_resnet50_fpn(
    weights: Optional[MaskRCNNResNet50FPNWeights] = None,
    weights_backbone: Optional[ResNet50Weights] = None,
    progress: bool = True,
    num_classes: int = 91,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> MaskRCNN:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = MaskRCNNResNet50FPNWeights.Coco_RefV1 if kwargs.pop("pretrained") else None
    weights = MaskRCNNResNet50FPNWeights.verify(weights)
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
    model = MaskRCNN(backbone, num_classes=num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
        if weights == MaskRCNNResNet50FPNWeights.Coco_RefV1:
            overwrite_eps(model, 0.0)

    return model
