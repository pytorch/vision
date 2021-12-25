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
from .._api import WeightsEnum, Weights
from .._meta import _COCO_CATEGORIES
from .._utils import handle_legacy_interface, _ovewrite_value_param
from ..vgg import VGG16_Weights, vgg16


__all__ = [
    "SSD300_VGG16_Weights",
    "ssd300_vgg16",
]


class SSD300_VGG16_Weights(WeightsEnum):
    Coco_V1 = Weights(
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
    default = Coco_V1


@handle_legacy_interface(
    weights=("pretrained", SSD300_VGG16_Weights.Coco_V1),
    weights_backbone=("pretrained_backbone", VGG16_Weights.ImageNet1K_Features),
)
def ssd300_vgg16(
    *,
    weights: Optional[SSD300_VGG16_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[VGG16_Weights] = None,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> SSD:
    weights = SSD300_VGG16_Weights.verify(weights)
    weights_backbone = VGG16_Weights.verify(weights_backbone)

    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the parameter.")

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

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
