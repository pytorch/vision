import re
from functools import partial
from typing import Any, Optional, Tuple

import torch.nn as nn
from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.densenet import DenseNet
from ._api import WeightsEnum, Weights
from ._meta import _IMAGENET_CATEGORIES
from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = [
    "DenseNet",
    "DenseNet121_Weights",
    "DenseNet161_Weights",
    "DenseNet169_Weights",
    "DenseNet201_Weights",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
]


def _load_state_dict(model: nn.Module, weights: WeightsEnum, progress: bool) -> None:
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = weights.get_state_dict(progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> DenseNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)

    if weights is not None:
        _load_state_dict(model=model, weights=weights, progress=progress)

    return model


_COMMON_META = {
    "size": (224, 224),
    "categories": _IMAGENET_CATEGORIES,
    "interpolation": InterpolationMode.BILINEAR,
    "recipe": "https://github.com/pytorch/vision/pull/116",
}


class DenseNet121_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/densenet121-a639ec97.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "acc@1": 74.434,
            "acc@5": 91.972,
        },
    )
    default = ImageNet1K_V1


class DenseNet161_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/densenet161-8d451a50.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "acc@1": 77.138,
            "acc@5": 93.560,
        },
    )
    default = ImageNet1K_V1


class DenseNet169_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/densenet169-b2777c0a.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "acc@1": 75.600,
            "acc@5": 92.806,
        },
    )
    default = ImageNet1K_V1


class DenseNet201_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/densenet201-c1103571.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "acc@1": 76.896,
            "acc@5": 93.370,
        },
    )
    default = ImageNet1K_V1


@handle_legacy_interface(weights=("pretrained", DenseNet121_Weights.ImageNet1K_V1))
def densenet121(*, weights: Optional[DenseNet121_Weights] = None, progress: bool = True, **kwargs: Any) -> DenseNet:
    weights = DenseNet121_Weights.verify(weights)

    return _densenet(32, (6, 12, 24, 16), 64, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", DenseNet161_Weights.ImageNet1K_V1))
def densenet161(*, weights: Optional[DenseNet161_Weights] = None, progress: bool = True, **kwargs: Any) -> DenseNet:
    weights = DenseNet161_Weights.verify(weights)

    return _densenet(48, (6, 12, 36, 24), 96, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", DenseNet169_Weights.ImageNet1K_V1))
def densenet169(*, weights: Optional[DenseNet169_Weights] = None, progress: bool = True, **kwargs: Any) -> DenseNet:
    weights = DenseNet169_Weights.verify(weights)

    return _densenet(32, (6, 12, 32, 32), 64, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", DenseNet201_Weights.ImageNet1K_V1))
def densenet201(*, weights: Optional[DenseNet201_Weights] = None, progress: bool = True, **kwargs: Any) -> DenseNet:
    weights = DenseNet201_Weights.verify(weights)

    return _densenet(32, (6, 12, 48, 32), 64, weights, progress, **kwargs)
