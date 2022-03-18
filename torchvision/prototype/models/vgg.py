from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import ImageClassificationEval
from torchvision.transforms.functional import InterpolationMode

from ...models.vgg import VGG, make_layers, cfgs
from ._api import WeightsEnum, Weights
from ._meta import _IMAGENET_CATEGORIES
from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = [
    "VGG",
    "VGG11_Weights",
    "VGG11_BN_Weights",
    "VGG13_Weights",
    "VGG13_BN_Weights",
    "VGG16_Weights",
    "VGG16_BN_Weights",
    "VGG19_Weights",
    "VGG19_BN_Weights",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
]


def _vgg(cfg: str, batch_norm: bool, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> VGG:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    return model


_COMMON_META = {
    "task": "image_classification",
    "architecture": "VGG",
    "publication_year": 2014,
    "size": (224, 224),
    "min_size": (32, 32),
    "categories": _IMAGENET_CATEGORIES,
    "interpolation": InterpolationMode.BILINEAR,
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
}


class VGG11_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg11-8a719046.pth",
        transforms=partial(ImageClassificationEval, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 132863336,
            "acc@1": 69.020,
            "acc@5": 88.628,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG11_BN_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
        transforms=partial(ImageClassificationEval, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 132868840,
            "acc@1": 70.370,
            "acc@5": 89.810,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG13_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg13-19584684.pth",
        transforms=partial(ImageClassificationEval, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 133047848,
            "acc@1": 69.928,
            "acc@5": 89.246,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG13_BN_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
        transforms=partial(ImageClassificationEval, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 133053736,
            "acc@1": 71.586,
            "acc@5": 90.374,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg16-397923af.pth",
        transforms=partial(ImageClassificationEval, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 138357544,
            "acc@1": 71.592,
            "acc@5": 90.382,
        },
    )
    # We port the features of a VGG16 backbone trained by amdegroot because unlike the one on TorchVision, it uses the
    # same input standardization method as the paper. Only the `features` weights have proper values, those on the
    # `classifier` module are filled with nans.
    IMAGENET1K_FEATURES = Weights(
        url="https://download.pytorch.org/models/vgg16_features-amdegroot-88682ab5.pth",
        transforms=partial(
            ImageClassificationEval,
            crop_size=224,
            mean=(0.48235, 0.45882, 0.40784),
            std=(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0),
        ),
        meta={
            **_COMMON_META,
            "num_params": 138357544,
            "categories": None,
            "recipe": "https://github.com/amdegroot/ssd.pytorch#training-ssd",
            "acc@1": float("nan"),
            "acc@5": float("nan"),
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG16_BN_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
        transforms=partial(ImageClassificationEval, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 138365992,
            "acc@1": 73.360,
            "acc@5": 91.516,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG19_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
        transforms=partial(ImageClassificationEval, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 143667240,
            "acc@1": 72.376,
            "acc@5": 90.876,
        },
    )
    DEFAULT = IMAGENET1K_V1


class VGG19_BN_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
        transforms=partial(ImageClassificationEval, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 143678248,
            "acc@1": 74.218,
            "acc@5": 91.842,
        },
    )
    DEFAULT = IMAGENET1K_V1


@handle_legacy_interface(weights=("pretrained", VGG11_Weights.IMAGENET1K_V1))
def vgg11(*, weights: Optional[VGG11_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    weights = VGG11_Weights.verify(weights)

    return _vgg("A", False, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", VGG11_BN_Weights.IMAGENET1K_V1))
def vgg11_bn(*, weights: Optional[VGG11_BN_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    weights = VGG11_BN_Weights.verify(weights)

    return _vgg("A", True, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", VGG13_Weights.IMAGENET1K_V1))
def vgg13(*, weights: Optional[VGG13_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    weights = VGG13_Weights.verify(weights)

    return _vgg("B", False, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", VGG13_BN_Weights.IMAGENET1K_V1))
def vgg13_bn(*, weights: Optional[VGG13_BN_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    weights = VGG13_BN_Weights.verify(weights)

    return _vgg("B", True, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", VGG16_Weights.IMAGENET1K_V1))
def vgg16(*, weights: Optional[VGG16_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    weights = VGG16_Weights.verify(weights)

    return _vgg("D", False, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", VGG16_BN_Weights.IMAGENET1K_V1))
def vgg16_bn(*, weights: Optional[VGG16_BN_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    weights = VGG16_BN_Weights.verify(weights)

    return _vgg("D", True, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", VGG19_Weights.IMAGENET1K_V1))
def vgg19(*, weights: Optional[VGG19_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    weights = VGG19_Weights.verify(weights)

    return _vgg("E", False, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", VGG19_BN_Weights.IMAGENET1K_V1))
def vgg19_bn(*, weights: Optional[VGG19_BN_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    weights = VGG19_BN_Weights.verify(weights)

    return _vgg("E", True, weights, progress, **kwargs)
