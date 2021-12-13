from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.mnasnet import MNASNet
from ._api import WeightsEnum, Weights
from ._meta import _IMAGENET_CATEGORIES
from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = [
    "MNASNet",
    "MNASNet0_5_Weights",
    "MNASNet0_75_Weights",
    "MNASNet1_0_Weights",
    "MNASNet1_3_Weights",
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


class MNASNet0_5_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "acc@1": 67.734,
            "acc@5": 87.490,
        },
    )
    default = ImageNet1K_V1


class MNASNet0_75_Weights(WeightsEnum):
    # If a default model is added here the corresponding changes need to be done in mnasnet0_75
    pass


class MNASNet1_0_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "acc@1": 73.456,
            "acc@5": 91.510,
        },
    )
    default = ImageNet1K_V1


class MNASNet1_3_Weights(WeightsEnum):
    # If a default model is added here the corresponding changes need to be done in mnasnet1_3
    pass


def _mnasnet(alpha: float, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> MNASNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MNASNet(alpha, **kwargs)

    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


@handle_legacy_interface(weights=("pretrained", MNASNet0_5_Weights.ImageNet1K_V1))
def mnasnet0_5(*, weights: Optional[MNASNet0_5_Weights] = None, progress: bool = True, **kwargs: Any) -> MNASNet:
    weights = MNASNet0_5_Weights.verify(weights)

    return _mnasnet(0.5, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", None))
def mnasnet0_75(*, weights: Optional[MNASNet0_75_Weights] = None, progress: bool = True, **kwargs: Any) -> MNASNet:
    weights = MNASNet0_75_Weights.verify(weights)

    return _mnasnet(0.75, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", MNASNet1_0_Weights.ImageNet1K_V1))
def mnasnet1_0(*, weights: Optional[MNASNet1_0_Weights] = None, progress: bool = True, **kwargs: Any) -> MNASNet:
    weights = MNASNet1_0_Weights.verify(weights)

    return _mnasnet(1.0, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", None))
def mnasnet1_3(*, weights: Optional[MNASNet1_3_Weights] = None, progress: bool = True, **kwargs: Any) -> MNASNet:
    weights = MNASNet1_3_Weights.verify(weights)

    return _mnasnet(1.3, weights, progress, **kwargs)
