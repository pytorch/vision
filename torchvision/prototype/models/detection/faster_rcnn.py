import warnings
from typing import Any, Optional

from ....models.detection.faster_rcnn import (
    _validate_trainable_layers,
    _resnet_fpn_extractor,
    FasterRCNN,
    misc_nn_ops,
    overwrite_eps,
)
from ...transforms.presets import CocoEval
from .._api import Weights, WeightEntry
from .._meta import _COCO_CATEGORIES
from ..resnet import ResNet50Weights, resnet50


__all__ = ["FasterRCNN", "FasterRCNNResNet50FPNWeights", "fasterrcnn_resnet50_fpn"]


class FasterRCNNResNet50FPNWeights(Weights):
    Coco_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
        transforms=CocoEval,
        meta={
            "categories": _COCO_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/detection#faster-r-cnn-resnet-50-fpn",
            "map": 37.0,
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
        warnings.warn("The argument pretrained is deprecated, please use weights instead.")
        weights = FasterRCNNResNet50FPNWeights.Coco_RefV1 if kwargs.pop("pretrained") else None
    weights = FasterRCNNResNet50FPNWeights.verify(weights)
    if "pretrained_backbone" in kwargs:
        warnings.warn("The argument pretrained_backbone is deprecated, please use weights_backbone instead.")
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
        model.load_state_dict(weights.state_dict(progress=progress))
        if weights == FasterRCNNResNet50FPNWeights.Coco_RefV1:
            overwrite_eps(model, 0.0)

    return model
