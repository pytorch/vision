from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import ImageClassificationEval
from torchvision.transforms.functional import InterpolationMode

from ...models.squeezenet import SqueezeNet
from ._api import WeightsEnum, Weights
from ._meta import _IMAGENET_CATEGORIES
from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = ["SqueezeNet", "SqueezeNet1_0_Weights", "SqueezeNet1_1_Weights", "squeezenet1_0", "squeezenet1_1"]


_COMMON_META = {
    "task": "image_classification",
    "architecture": "SqueezeNet",
    "publication_year": 2016,
    "size": (224, 224),
    "categories": _IMAGENET_CATEGORIES,
    "interpolation": InterpolationMode.BILINEAR,
    "recipe": "https://github.com/pytorch/vision/pull/49#issuecomment-277560717",
}


class SqueezeNet1_0_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth",
        transforms=partial(ImageClassificationEval, crop_size=224),
        meta={
            **_COMMON_META,
            "min_size": (21, 21),
            "num_params": 1248424,
            "acc@1": 58.092,
            "acc@5": 80.420,
        },
    )
    DEFAULT = IMAGENET1K_V1


class SqueezeNet1_1_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
        transforms=partial(ImageClassificationEval, crop_size=224),
        meta={
            **_COMMON_META,
            "min_size": (17, 17),
            "num_params": 1235496,
            "acc@1": 58.178,
            "acc@5": 80.624,
        },
    )
    DEFAULT = IMAGENET1K_V1


@handle_legacy_interface(weights=("pretrained", SqueezeNet1_0_Weights.IMAGENET1K_V1))
def squeezenet1_0(
    *, weights: Optional[SqueezeNet1_0_Weights] = None, progress: bool = True, **kwargs: Any
) -> SqueezeNet:
    weights = SqueezeNet1_0_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = SqueezeNet("1_0", **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


@handle_legacy_interface(weights=("pretrained", SqueezeNet1_1_Weights.IMAGENET1K_V1))
def squeezenet1_1(
    *, weights: Optional[SqueezeNet1_1_Weights] = None, progress: bool = True, **kwargs: Any
) -> SqueezeNet:
    weights = SqueezeNet1_1_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = SqueezeNet("1_1", **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
