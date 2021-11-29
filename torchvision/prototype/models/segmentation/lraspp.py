from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import VocEval
from torchvision.transforms.functional import InterpolationMode

from ....models.segmentation.lraspp import LRASPP, _lraspp_mobilenetv3
from .._api import WeightsEnum, Weights
from .._meta import _VOC_CATEGORIES
from .._utils import _deprecated_param, _deprecated_positional, _ovewrite_value_param
from ..mobilenetv3 import MobileNetV3LargeWeights, mobilenet_v3_large


__all__ = ["LRASPP", "LRASPPMobileNetV3LargeWeights", "lraspp_mobilenet_v3_large"]


class LRASPPMobileNetV3LargeWeights(WeightsEnum):
    CocoWithVocLabels_V1 = Weights(
        url="https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth",
        transforms=partial(VocEval, resize_size=520),
        meta={
            "categories": _VOC_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#lraspp_mobilenet_v3_large",
            "mIoU": 57.9,
            "acc": 91.2,
        },
        default=True,
    )


def lraspp_mobilenet_v3_large(
    weights: Optional[LRASPPMobileNetV3LargeWeights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[MobileNetV3LargeWeights] = None,
    **kwargs: Any,
) -> LRASPP:
    if kwargs.pop("aux_loss", False):
        raise NotImplementedError("This model does not use auxiliary loss")

    if type(weights) == bool and weights:
        _deprecated_positional(kwargs, "pretrained", "weights", True)
    if "pretrained" in kwargs:
        weights = _deprecated_param(kwargs, "pretrained", "weights", LRASPPMobileNetV3LargeWeights.CocoWithVocLabels_V1)
    weights = LRASPPMobileNetV3LargeWeights.verify(weights)
    if type(weights_backbone) == bool and weights_backbone:
        _deprecated_positional(kwargs, "pretrained_backbone", "weights_backbone", True)
    if "pretrained_backbone" in kwargs:
        weights_backbone = _deprecated_param(
            kwargs, "pretrained_backbone", "weights_backbone", MobileNetV3LargeWeights.ImageNet1K_V1
        )
    weights_backbone = MobileNetV3LargeWeights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 21

    backbone = mobilenet_v3_large(weights=weights_backbone, dilated=True)
    model = _lraspp_mobilenetv3(backbone, num_classes)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
