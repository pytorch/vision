import warnings
from functools import partial
from typing import Any, Optional

from torch import nn
from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.efficientnet import EfficientNet, MBConvConfig
from ._api import Weights, WeightEntry
from ._meta import _IMAGENET_CATEGORIES


__all__ = [
    "EfficientNet",
    "EfficientNetB0Weights",
    "EfficientNetB1Weights",
    "EfficientNetB2Weights",
    "EfficientNetB3Weights",
    "EfficientNetB4Weights",
    "EfficientNetB5Weights",
    "EfficientNetB6Weights",
    "EfficientNetB7Weights",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
]


def _efficientnet(
    width_mult: float,
    depth_mult: float,
    dropout: float,
    weights: Optional[Weights],
    progress: bool,
    **kwargs: Any,
) -> EfficientNet:
    if weights is not None:
        kwargs["num_classes"] = len(weights.meta["categories"])

    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]

    model = EfficientNet(inverted_residual_setting, dropout, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "categories": _IMAGENET_CATEGORIES,
    "interpolation": InterpolationMode.BICUBIC,
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#efficientnet",
}


class EfficientNetB0Weights(Weights):
    ImageNet1K_TimmV1 = WeightEntry(
        url="https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=256, interpolation=InterpolationMode.BICUBIC),
        meta={
            **_COMMON_META,
            "size": (224, 224),
            "acc@1": 77.692,
            "acc@5": 93.532,
        },
    )


class EfficientNetB1Weights(Weights):
    ImageNet1K_TimmV1 = WeightEntry(
        url="https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
        transforms=partial(ImageNetEval, crop_size=240, resize_size=256, interpolation=InterpolationMode.BICUBIC),
        meta={
            **_COMMON_META,
            "size": (240, 240),
            "acc@1": 78.642,
            "acc@5": 94.186,
        },
    )


class EfficientNetB2Weights(Weights):
    ImageNet1K_TimmV1 = WeightEntry(
        url="https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
        transforms=partial(ImageNetEval, crop_size=288, resize_size=288, interpolation=InterpolationMode.BICUBIC),
        meta={
            **_COMMON_META,
            "size": (288, 288),
            "acc@1": 80.608,
            "acc@5": 95.310,
        },
    )


class EfficientNetB3Weights(Weights):
    ImageNet1K_TimmV1 = WeightEntry(
        url="https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth",
        transforms=partial(ImageNetEval, crop_size=300, resize_size=320, interpolation=InterpolationMode.BICUBIC),
        meta={
            **_COMMON_META,
            "size": (300, 300),
            "acc@1": 82.008,
            "acc@5": 96.054,
        },
    )


class EfficientNetB4Weights(Weights):
    ImageNet1K_TimmV1 = WeightEntry(
        url="https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth",
        transforms=partial(ImageNetEval, crop_size=380, resize_size=384, interpolation=InterpolationMode.BICUBIC),
        meta={
            **_COMMON_META,
            "size": (380, 380),
            "acc@1": 83.384,
            "acc@5": 96.594,
        },
    )


class EfficientNetB5Weights(Weights):
    ImageNet1K_TFV1 = WeightEntry(
        url="https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth",
        transforms=partial(ImageNetEval, crop_size=456, resize_size=456, interpolation=InterpolationMode.BICUBIC),
        meta={
            **_COMMON_META,
            "size": (456, 456),
            "acc@1": 83.444,
            "acc@5": 96.628,
        },
    )


class EfficientNetB6Weights(Weights):
    ImageNet1K_TFV1 = WeightEntry(
        url="https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth",
        transforms=partial(ImageNetEval, crop_size=528, resize_size=528, interpolation=InterpolationMode.BICUBIC),
        meta={
            **_COMMON_META,
            "size": (528, 528),
            "acc@1": 84.008,
            "acc@5": 96.916,
        },
    )


class EfficientNetB7Weights(Weights):
    ImageNet1K_TFV1 = WeightEntry(
        url="https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth",
        transforms=partial(ImageNetEval, crop_size=600, resize_size=600, interpolation=InterpolationMode.BICUBIC),
        meta={
            **_COMMON_META,
            "size": (600, 600),
            "acc@1": 84.122,
            "acc@5": 96.908,
        },
    )


def efficientnet_b0(
    weights: Optional[EfficientNetB0Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = EfficientNetB0Weights.ImageNet1K_TimmV1 if kwargs.pop("pretrained") else None
    weights = EfficientNetB0Weights.verify(weights)
    return _efficientnet(width_mult=1.0, depth_mult=1.0, dropout=0.2, weights=weights, progress=progress, **kwargs)


def efficientnet_b1(
    weights: Optional[EfficientNetB1Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = EfficientNetB1Weights.ImageNet1K_TimmV1 if kwargs.pop("pretrained") else None
    weights = EfficientNetB1Weights.verify(weights)
    return _efficientnet(width_mult=1.0, depth_mult=1.1, dropout=0.2, weights=weights, progress=progress, **kwargs)


def efficientnet_b2(
    weights: Optional[EfficientNetB2Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = EfficientNetB2Weights.ImageNet1K_TimmV1 if kwargs.pop("pretrained") else None
    weights = EfficientNetB2Weights.verify(weights)
    return _efficientnet(width_mult=1.1, depth_mult=1.2, dropout=0.3, weights=weights, progress=progress, **kwargs)


def efficientnet_b3(
    weights: Optional[EfficientNetB3Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = EfficientNetB3Weights.ImageNet1K_TimmV1 if kwargs.pop("pretrained") else None
    weights = EfficientNetB3Weights.verify(weights)
    return _efficientnet(width_mult=1.2, depth_mult=1.4, dropout=0.3, weights=weights, progress=progress, **kwargs)


def efficientnet_b4(
    weights: Optional[EfficientNetB4Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = EfficientNetB4Weights.ImageNet1K_TimmV1 if kwargs.pop("pretrained") else None
    weights = EfficientNetB4Weights.verify(weights)
    return _efficientnet(width_mult=1.4, depth_mult=1.8, dropout=0.4, weights=weights, progress=progress, **kwargs)


def efficientnet_b5(
    weights: Optional[EfficientNetB5Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = EfficientNetB5Weights.ImageNet1K_TFV1 if kwargs.pop("pretrained") else None
    weights = EfficientNetB5Weights.verify(weights)
    return _efficientnet(
        width_mult=1.6,
        depth_mult=2.2,
        dropout=0.4,
        weights=weights,
        progress=progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_b6(
    weights: Optional[EfficientNetB6Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = EfficientNetB6Weights.ImageNet1K_TFV1 if kwargs.pop("pretrained") else None
    weights = EfficientNetB6Weights.verify(weights)
    return _efficientnet(
        width_mult=1.8,
        depth_mult=2.6,
        dropout=0.5,
        weights=weights,
        progress=progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_b7(
    weights: Optional[EfficientNetB7Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = EfficientNetB7Weights.ImageNet1K_TFV1 if kwargs.pop("pretrained") else None
    weights = EfficientNetB7Weights.verify(weights)
    return _efficientnet(
        width_mult=2.0,
        depth_mult=3.1,
        dropout=0.5,
        weights=weights,
        progress=progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )
