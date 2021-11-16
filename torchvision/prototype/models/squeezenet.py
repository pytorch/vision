import warnings
from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.squeezenet import SqueezeNet
from ._api import Weights, WeightEntry
from ._meta import _IMAGENET_CATEGORIES


__all__ = ["SqueezeNet", "SqueezeNet1_0Weights", "SqueezeNet1_1Weights", "squeezenet1_0", "squeezenet1_1"]


_COMMON_META = {
    "size": (224, 224),
    "categories": _IMAGENET_CATEGORIES,
    "interpolation": InterpolationMode.BILINEAR,
    "recipe": "https://github.com/pytorch/vision/pull/49#issuecomment-277560717",
}


class SqueezeNet1_0Weights(Weights):
    ImageNet1K_Community = WeightEntry(
        url="https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "acc@1": 58.092,
            "acc@5": 80.420,
        },
    )


class SqueezeNet1_1Weights(Weights):
    ImageNet1K_Community = WeightEntry(
        url="https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "acc@1": 58.178,
            "acc@5": 80.624,
        },
    )


def squeezenet1_0(weights: Optional[SqueezeNet1_0Weights] = None, progress: bool = True, **kwargs: Any) -> SqueezeNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = SqueezeNet1_0Weights.ImageNet1K_Community if kwargs.pop("pretrained") else None
    weights = SqueezeNet1_0Weights.verify(weights)
    if weights is not None:
        kwargs["num_classes"] = len(weights.meta["categories"])

    model = SqueezeNet("1_0", **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def squeezenet1_1(weights: Optional[SqueezeNet1_1Weights] = None, progress: bool = True, **kwargs: Any) -> SqueezeNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = SqueezeNet1_1Weights.ImageNet1K_Community if kwargs.pop("pretrained") else None
    weights = SqueezeNet1_1Weights.verify(weights)
    if weights is not None:
        kwargs["num_classes"] = len(weights.meta["categories"])

    model = SqueezeNet("1_1", **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
