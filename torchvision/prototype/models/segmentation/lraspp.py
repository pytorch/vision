from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import VocEval
from torchvision.transforms.functional import InterpolationMode

from ....models.segmentation.lraspp import LRASPP, _lraspp_mobilenetv3
from .._api import WeightsEnum, Weights
from .._meta import _VOC_CATEGORIES
from .._utils import handle_legacy_interface, _ovewrite_value_param
from ..mobilenetv3 import MobileNet_V3_Large_Weights, mobilenet_v3_large


__all__ = ["LRASPP", "LRASPP_MobileNet_V3_Large_Weights", "lraspp_mobilenet_v3_large"]


class LRASPP_MobileNet_V3_Large_Weights(WeightsEnum):
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
    )
    default = CocoWithVocLabels_V1


@handle_legacy_interface(
    weights=("pretrained", LRASPP_MobileNet_V3_Large_Weights.CocoWithVocLabels_V1),
    weights_backbone=("pretrained_backbone", MobileNet_V3_Large_Weights.ImageNet1K_V1),
)
def lraspp_mobilenet_v3_large(
    *,
    weights: Optional[LRASPP_MobileNet_V3_Large_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[MobileNet_V3_Large_Weights] = None,
    **kwargs: Any,
) -> LRASPP:
    if kwargs.pop("aux_loss", False):
        raise NotImplementedError("This model does not use auxiliary loss")

    weights = LRASPP_MobileNet_V3_Large_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)

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
