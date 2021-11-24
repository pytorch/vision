import warnings
from typing import Any, Optional

from torchvision.prototype.transforms import CocoEval
from torchvision.transforms.functional import InterpolationMode

from ....models.detection.ssd import (
    _validate_trainable_layers,
    _vgg_extractor,
    DefaultBoxGenerator,
    SSD,
)
from .._api import Weights, WeightEntry
from .._meta import _COCO_CATEGORIES
from ..vgg import VGG16Weights, vgg16


__all__ = [
    "SSD300VGG16Weights",
    "ssd300_vgg16",
]


class SSD300VGG16Weights(Weights):
    Coco_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth",
        transforms=CocoEval,
        meta={
            "size": (300, 300),
            "categories": _COCO_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/detection#ssd300-vgg16",
            "map": 25.1,
        },
    )


def ssd300_vgg16(
    weights: Optional[SSD300VGG16Weights] = None,
    weights_backbone: Optional[VGG16Weights] = None,
    progress: bool = True,
    num_classes: int = 91,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> SSD:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = SSD300VGG16Weights.Coco_RefV1 if kwargs.pop("pretrained") else None
    weights = SSD300VGG16Weights.verify(weights)
    if "pretrained_backbone" in kwargs:
        warnings.warn("The parameter pretrained_backbone is deprecated, please use weights_backbone instead.")
        weights_backbone = VGG16Weights.ImageNet1K_Features if kwargs.pop("pretrained_backbone") else None
    weights_backbone = VGG16Weights.verify(weights_backbone)

    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the parameter.")

    if weights is not None:
        weights_backbone = None
        num_classes = len(weights.meta["categories"])

    trainable_backbone_layers = _validate_trainable_layers(
        weights is not None or weights_backbone is not None, trainable_backbone_layers, 5, 4
    )

    # Use custom backbones more appropriate for SSD
    backbone = vgg16(weights=weights_backbone, progress=progress)
    backbone = _vgg_extractor(backbone, False, trainable_backbone_layers)
    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
        steps=[8, 16, 32, 64, 100, 300],
    )

    defaults = {
        # Rescale the input in a way compatible to the backbone
        "image_mean": [0.48235, 0.45882, 0.40784],
        "image_std": [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],  # undo the 0-1 scaling of toTensor
    }
    kwargs: Any = {**defaults, **kwargs}
    model = SSD(backbone, anchor_generator, (300, 300), num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
