from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.shufflenetv2 import ShuffleNetV2
from ._api import WeightsEnum, Weights
from ._meta import _IMAGENET_CATEGORIES
from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = [
    "ShuffleNetV2",
    "ShuffleNet_V2_X0_5_Weights",
    "ShuffleNet_V2_X1_0_Weights",
    "ShuffleNet_V2_X1_5_Weights",
    "ShuffleNet_V2_X2_0_Weights",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
]


def _shufflenetv2(
    weights: Optional[WeightsEnum],
    progress: bool,
    *args: Any,
    **kwargs: Any,
) -> ShuffleNetV2:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ShuffleNetV2(*args, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "size": (224, 224),
    "categories": _IMAGENET_CATEGORIES,
    "interpolation": InterpolationMode.BILINEAR,
    "recipe": "https://github.com/barrh/Shufflenet-v2-Pytorch/tree/v0.1.0",
}


class ShuffleNet_V2_X0_5_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "acc@1": 69.362,
            "acc@5": 88.316,
        },
    )
    default = ImageNet1K_V1


class ShuffleNet_V2_X1_0_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "acc@1": 60.552,
            "acc@5": 81.746,
        },
    )
    default = ImageNet1K_V1


class ShuffleNet_V2_X1_5_Weights(WeightsEnum):
    pass


class ShuffleNet_V2_X2_0_Weights(WeightsEnum):
    pass


@handle_legacy_interface(weights=("pretrained", ShuffleNet_V2_X0_5_Weights.ImageNet1K_V1))
def shufflenet_v2_x0_5(
    *, weights: Optional[ShuffleNet_V2_X0_5_Weights] = None, progress: bool = True, **kwargs: Any
) -> ShuffleNetV2:
    weights = ShuffleNet_V2_X0_5_Weights.verify(weights)

    return _shufflenetv2(weights, progress, [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


@handle_legacy_interface(weights=("pretrained", ShuffleNet_V2_X1_0_Weights.ImageNet1K_V1))
def shufflenet_v2_x1_0(
    *, weights: Optional[ShuffleNet_V2_X1_0_Weights] = None, progress: bool = True, **kwargs: Any
) -> ShuffleNetV2:
    weights = ShuffleNet_V2_X1_0_Weights.verify(weights)

    return _shufflenetv2(weights, progress, [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


@handle_legacy_interface(weights=("pretrained", None))
def shufflenet_v2_x1_5(
    *, weights: Optional[ShuffleNet_V2_X1_5_Weights] = None, progress: bool = True, **kwargs: Any
) -> ShuffleNetV2:
    weights = ShuffleNet_V2_X1_5_Weights.verify(weights)

    return _shufflenetv2(weights, progress, [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


@handle_legacy_interface(weights=("pretrained", None))
def shufflenet_v2_x2_0(
    *, weights: Optional[ShuffleNet_V2_X2_0_Weights] = None, progress: bool = True, **kwargs: Any
) -> ShuffleNetV2:
    weights = ShuffleNet_V2_X2_0_Weights.verify(weights)

    return _shufflenetv2(weights, progress, [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)
