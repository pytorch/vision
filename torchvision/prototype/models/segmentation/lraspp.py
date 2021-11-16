import warnings
from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import VocEval
from torchvision.transforms.functional import InterpolationMode

from ....models.segmentation.lraspp import LRASPP, _lraspp_mobilenetv3
from .._api import Weights, WeightEntry
from .._meta import _VOC_CATEGORIES
from ..mobilenetv3 import MobileNetV3LargeWeights, mobilenet_v3_large


__all__ = ["LRASPP", "LRASPPMobileNetV3LargeWeights", "lraspp_mobilenet_v3_large"]


class LRASPPMobileNetV3LargeWeights(Weights):
    CocoWithVocLabels_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth",
        transforms=partial(VocEval, resize_size=520),
        meta={
            "categories": _VOC_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#lraspp_mobilenet_v3_large",
            "mIoU": 57.9,
            "acc": 91.2,
        },
    )


def lraspp_mobilenet_v3_large(
    weights: Optional[LRASPPMobileNetV3LargeWeights] = None,
    weights_backbone: Optional[MobileNetV3LargeWeights] = None,
    progress: bool = True,
    num_classes: int = 21,
    **kwargs: Any,
) -> LRASPP:
    if kwargs.pop("aux_loss", False):
        raise NotImplementedError("This model does not use auxiliary loss")

    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = LRASPPMobileNetV3LargeWeights.CocoWithVocLabels_RefV1 if kwargs.pop("pretrained") else None
    weights = LRASPPMobileNetV3LargeWeights.verify(weights)
    if "pretrained_backbone" in kwargs:
        warnings.warn("The parameter pretrained_backbone is deprecated, please use weights_backbone instead.")
        weights_backbone = MobileNetV3LargeWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained_backbone") else None
    weights_backbone = MobileNetV3LargeWeights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = len(weights.meta["categories"])

    backbone = mobilenet_v3_large(weights=weights_backbone, dilated=True)
    model = _lraspp_mobilenetv3(backbone, num_classes)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
