from functools import partial
from typing import Any, Optional, List

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.mobilenetv3 import MobileNetV3, _mobilenet_v3_conf, InvertedResidualConfig
from ._api import WeightsEnum, Weights
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _deprecated_param, _deprecated_positional, _ovewrite_named_param


__all__ = [
    "MobileNetV3",
    "MobileNetV3LargeWeights",
    "MobileNetV3SmallWeights",
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


class MobileNetV3LargeWeights(WeightsEnum):
    ImageNet1K_RefV1 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv3-large--small",
            "acc@1": 74.042,
            "acc@5": 91.340,
        },
        default=False,
    )
    ImageNet1K_RefV2 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning",
            "acc@1": 75.274,
            "acc@5": 92.566,
        },
        default=True,
    )


class MobileNetV3SmallWeights(WeightsEnum):
    ImageNet1K_RefV1 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv3-large--small",
            "acc@1": 67.668,
            "acc@5": 87.402,
        },
        default=True,
    )


def mobilenet_v3_large(
    weights: Optional[MobileNetV3LargeWeights] = None, progress: bool = True, **kwargs: Any
) -> MobileNetV3:
    if type(weights) == bool and weights:
        _deprecated_positional(kwargs, "pretrained", "weights", True)
    if "pretrained" in kwargs:
        weights = _deprecated_param(kwargs, "pretrained", "weights", MobileNetV3LargeWeights.ImageNet1K_RefV1)
    weights = MobileNetV3LargeWeights.verify(weights)

    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large", **kwargs)
    return _mobilenet_v3(inverted_residual_setting, last_channel, weights, progress, **kwargs)


def mobilenet_v3_small(
    weights: Optional[MobileNetV3SmallWeights] = None, progress: bool = True, **kwargs: Any
) -> MobileNetV3:
    if type(weights) == bool and weights:
        _deprecated_positional(kwargs, "pretrained", "weights", True)
    if "pretrained" in kwargs:
        weights = _deprecated_param(kwargs, "pretrained", "weights", MobileNetV3SmallWeights.ImageNet1K_RefV1)
    weights = MobileNetV3SmallWeights.verify(weights)

    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", **kwargs)
    return _mobilenet_v3(inverted_residual_setting, last_channel, weights, progress, **kwargs)
