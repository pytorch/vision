from functools import partial
from typing import Any, Optional, List

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.mobilenetv3 import MobileNetV3, _mobilenet_v3_conf, InvertedResidualConfig
from ._api import WeightsEnum, Weights
from ._meta import _IMAGENET_CATEGORIES
from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = [
    "MobileNetV3",
    "MobileNet_V3_Large_Weights",
    "MobileNet_V3_Small_Weights",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
]


def _mobilenet_v3(
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> MobileNetV3:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "size": (224, 224),
    "categories": _IMAGENET_CATEGORIES,
    "interpolation": InterpolationMode.BILINEAR,
}


class MobileNet_V3_Large_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv3-large--small",
            "acc@1": 74.042,
            "acc@5": 91.340,
        },
    )
    ImageNet1K_V2 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning",
            "acc@1": 75.274,
            "acc@5": 92.566,
        },
    )
    default = ImageNet1K_V2


class MobileNet_V3_Small_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv3-large--small",
            "acc@1": 67.668,
            "acc@5": 87.402,
        },
    )
    default = ImageNet1K_V1


@handle_legacy_interface(weights=("pretrained", MobileNet_V3_Large_Weights.ImageNet1K_V1))
def mobilenet_v3_large(
    *, weights: Optional[MobileNet_V3_Large_Weights] = None, progress: bool = True, **kwargs: Any
) -> MobileNetV3:
    weights = MobileNet_V3_Large_Weights.verify(weights)

    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large", **kwargs)
    return _mobilenet_v3(inverted_residual_setting, last_channel, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", MobileNet_V3_Small_Weights.ImageNet1K_V1))
def mobilenet_v3_small(
    *, weights: Optional[MobileNet_V3_Small_Weights] = None, progress: bool = True, **kwargs: Any
) -> MobileNetV3:
    weights = MobileNet_V3_Small_Weights.verify(weights)

    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", **kwargs)
    return _mobilenet_v3(inverted_residual_setting, last_channel, weights, progress, **kwargs)
