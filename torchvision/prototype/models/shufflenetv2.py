import warnings
from functools import partial
from typing import Any, Optional

from torchvision.transforms.functional import InterpolationMode

from ...models.shufflenetv2 import ShuffleNetV2
from ..transforms.presets import ImageNetEval
from ._api import Weights, WeightEntry
from ._meta import _IMAGENET_CATEGORIES


__all__ = [
    "ShuffleNetV2",
    "ShuffleNetV2_x0_5Weights",
    "ShuffleNetV2_x1_0Weights",
    "ShuffleNetV2_x1_5Weights",
    "ShuffleNetV2_x2_0Weights",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
]


def _shufflenetv2(
    weights: Optional[Weights],
    progress: bool,
    *args: Any,
    **kwargs: Any,
) -> ShuffleNetV2:
    if weights is not None:
        kwargs["num_classes"] = len(weights.meta["categories"])

    model = ShuffleNetV2(*args, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.state_dict(progress=progress))

    return model


_common_meta = {"size": (224, 224), "categories": _IMAGENET_CATEGORIES, "interpolation": InterpolationMode.BILINEAR}


class ShuffleNetV2_x0_5Weights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_common_meta,
            "recipe": "",
            "acc@1": 69.362,
            "acc@5": 88.316,
        },
    )


class ShuffleNetV2_x1_0Weights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_common_meta,
            "recipe": "",
            "acc@1": 60.552,
            "acc@5": 81.746,
        },
    )


class ShuffleNetV2_x1_5Weights(Weights):
    pass


class ShuffleNetV2_x2_0Weights(Weights):
    pass


def shufflenet_v2_x0_5(
    weights: Optional[ShuffleNetV2_x0_5Weights] = None, progress: bool = True, **kwargs: Any
) -> ShuffleNetV2:
    if "pretrained" in kwargs:
        warnings.warn("The argument pretrained is deprecated, please use weights instead.")
        weights = ShuffleNetV2_x0_5Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = ShuffleNetV2_x0_5Weights.verify(weights)

    return _shufflenetv2(weights, progress, [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_0(
    weights: Optional[ShuffleNetV2_x1_0Weights] = None, progress: bool = True, **kwargs: Any
) -> ShuffleNetV2:
    if "pretrained" in kwargs:
        warnings.warn("The argument pretrained is deprecated, please use weights instead.")
        weights = ShuffleNetV2_x1_0Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = ShuffleNetV2_x1_0Weights.verify(weights)

    return _shufflenetv2(weights, progress, [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


def shufflenet_v2_x1_5(
    weights: Optional[ShuffleNetV2_x1_5Weights] = None, progress: bool = True, **kwargs: Any
) -> ShuffleNetV2:
    if "pretrained" in kwargs:
        warnings.warn("The argument pretrained is deprecated, please use weights instead.")
        weights = ShuffleNetV2_x1_5Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = ShuffleNetV2_x1_5Weights.verify(weights)

    return _shufflenetv2(weights, progress, [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenet_v2_x2_0(
    weights: Optional[ShuffleNetV2_x2_0Weights] = None, progress: bool = True, **kwargs: Any
) -> ShuffleNetV2:
    if "pretrained" in kwargs:
        warnings.warn("The argument pretrained is deprecated, please use weights instead.")
        weights = ShuffleNetV2_x2_0Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = ShuffleNetV2_x2_0Weights.verify(weights)

    return _shufflenetv2(weights, progress, [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)
