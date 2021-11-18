import warnings
from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.mnasnet import MNASNet
from ._api import Weights, WeightEntry
from ._meta import _IMAGENET_CATEGORIES


__all__ = [
    "MNASNet",
    "MNASNet0_5Weights",
    "MNASNet0_75Weights",
    "MNASNet1_0Weights",
    "MNASNet1_3Weights",
    "mnasnet0_5",
    "mnasnet0_75",
    "mnasnet1_0",
    "mnasnet1_3",
]


_COMMON_META = {
    "size": (224, 224),
    "categories": _IMAGENET_CATEGORIES,
    "interpolation": InterpolationMode.BILINEAR,
    "recipe": "https://github.com/1e100/mnasnet_trainer",
}


class MNASNet0_5Weights(Weights):
    ImageNet1K_Community = WeightEntry(
        url="https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "acc@1": 67.734,
            "acc@5": 87.490,
        },
    )


class MNASNet0_75Weights(Weights):
    # If a default model is added here the corresponding changes need to be done in mnasnet0_75
    pass


class MNASNet1_0Weights(Weights):
    ImageNet1K_Community = WeightEntry(
        url="https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "acc@1": 73.456,
            "acc@5": 91.510,
        },
    )


class MNASNet1_3Weights(Weights):
    # If a default model is added here the corresponding changes need to be done in mnasnet1_3
    pass


def _mnasnet(alpha: float, weights: Optional[Weights], progress: bool, **kwargs: Any) -> MNASNet:
    if weights is not None:
        kwargs["num_classes"] = len(weights.meta["categories"])

    model = MNASNet(alpha, **kwargs)

    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def mnasnet0_5(weights: Optional[MNASNet0_5Weights] = None, progress: bool = True, **kwargs: Any) -> MNASNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = MNASNet0_5Weights.ImageNet1K_Community if kwargs.pop("pretrained") else None

    weights = MNASNet0_5Weights.verify(weights)

    return _mnasnet(0.5, weights, progress, **kwargs)


def mnasnet0_75(weights: Optional[MNASNet0_75Weights] = None, progress: bool = True, **kwargs: Any) -> MNASNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        if kwargs.pop("pretrained"):
            raise ValueError("No checkpoint is available for model type mnasnet0_75")

    weights = MNASNet0_75Weights.verify(weights)

    return _mnasnet(0.75, weights, progress, **kwargs)


def mnasnet1_0(weights: Optional[MNASNet1_0Weights] = None, progress: bool = True, **kwargs: Any) -> MNASNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = MNASNet1_0Weights.ImageNet1K_Community if kwargs.pop("pretrained") else None
    weights = MNASNet1_0Weights.verify(weights)

    return _mnasnet(1.0, weights, progress, **kwargs)


def mnasnet1_3(weights: Optional[MNASNet1_3Weights] = None, progress: bool = True, **kwargs: Any) -> MNASNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        if kwargs.pop("pretrained"):
            raise ValueError("No checkpoint is available for model type mnasnet1_3")

    weights = MNASNet1_3Weights.verify(weights)

    return _mnasnet(1.3, weights, progress, **kwargs)
