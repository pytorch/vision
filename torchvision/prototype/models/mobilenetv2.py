from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import ImageClassificationEval
from torchvision.transforms.functional import InterpolationMode

from ...models.mobilenetv2 import MobileNetV2
from ._api import WeightsEnum, Weights
from ._meta import _IMAGENET_CATEGORIES
from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = ["MobileNetV2", "MobileNet_V2_Weights", "mobilenet_v2"]


_COMMON_META = {
    "task": "image_classification",
    "architecture": "MobileNetV2",
    "publication_year": 2018,
    "num_params": 3504872,
    "size": (224, 224),
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
    "interpolation": InterpolationMode.BILINEAR,
}


class MobileNet_V2_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
        transforms=partial(ImageClassificationEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv2",
            "acc@1": 71.878,
            "acc@5": 90.286,
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth",
        transforms=partial(ImageClassificationEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning",
            "acc@1": 72.154,
            "acc@5": 90.822,
        },
    )
    DEFAULT = IMAGENET1K_V2


@handle_legacy_interface(weights=("pretrained", MobileNet_V2_Weights.IMAGENET1K_V1))
def mobilenet_v2(
    *, weights: Optional[MobileNet_V2_Weights] = None, progress: bool = True, **kwargs: Any
) -> MobileNetV2:
    weights = MobileNet_V2_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MobileNetV2(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
