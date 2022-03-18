from functools import partial
from typing import Any, Optional, Sequence, Union

from torch import nn
from torchvision.prototype.transforms import ImageClassificationEval
from torchvision.transforms.functional import InterpolationMode

from ...models.efficientnet import EfficientNet, MBConvConfig, FusedMBConvConfig, _efficientnet_conf
from ._api import WeightsEnum, Weights
from ._meta import _IMAGENET_CATEGORIES
from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = [
    "EfficientNet",
    "EfficientNet_B0_Weights",
    "EfficientNet_B1_Weights",
    "EfficientNet_B2_Weights",
    "EfficientNet_B3_Weights",
    "EfficientNet_B4_Weights",
    "EfficientNet_B5_Weights",
    "EfficientNet_B6_Weights",
    "EfficientNet_B7_Weights",
    "EfficientNet_V2_S_Weights",
    "EfficientNet_V2_M_Weights",
    "EfficientNet_V2_L_Weights",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
]


def _efficientnet(
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
    dropout: float,
    last_channel: Optional[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> EfficientNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = EfficientNet(inverted_residual_setting, dropout, last_channel=last_channel, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "task": "image_classification",
    "categories": _IMAGENET_CATEGORIES,
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#efficientnet",
}


_COMMON_META_V1 = {
    **_COMMON_META,
    "architecture": "EfficientNet",
    "publication_year": 2019,
    "interpolation": InterpolationMode.BICUBIC,
    "min_size": (1, 1),
}


_COMMON_META_V2 = {
    **_COMMON_META,
    "architecture": "EfficientNetV2",
    "publication_year": 2021,
    "interpolation": InterpolationMode.BILINEAR,
    "min_size": (33, 33),
}


class EfficientNet_B0_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
        transforms=partial(
            ImageClassificationEval, crop_size=224, resize_size=256, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META_V1,
            "num_params": 5288548,
            "size": (224, 224),
            "acc@1": 77.692,
            "acc@5": 93.532,
        },
    )
    DEFAULT = IMAGENET1K_V1


class EfficientNet_B1_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
        transforms=partial(
            ImageClassificationEval, crop_size=240, resize_size=256, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META_V1,
            "num_params": 7794184,
            "size": (240, 240),
            "acc@1": 78.642,
            "acc@5": 94.186,
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth",
        transforms=partial(
            ImageClassificationEval, crop_size=240, resize_size=255, interpolation=InterpolationMode.BILINEAR
        ),
        meta={
            **_COMMON_META_V1,
            "num_params": 7794184,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-lr-wd-crop-tuning",
            "interpolation": InterpolationMode.BILINEAR,
            "size": (240, 240),
            "acc@1": 79.838,
            "acc@5": 94.934,
        },
    )
    DEFAULT = IMAGENET1K_V2


class EfficientNet_B2_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
        transforms=partial(
            ImageClassificationEval, crop_size=288, resize_size=288, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META_V1,
            "num_params": 9109994,
            "size": (288, 288),
            "acc@1": 80.608,
            "acc@5": 95.310,
        },
    )
    DEFAULT = IMAGENET1K_V1


class EfficientNet_B3_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth",
        transforms=partial(
            ImageClassificationEval, crop_size=300, resize_size=320, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META_V1,
            "num_params": 12233232,
            "size": (300, 300),
            "acc@1": 82.008,
            "acc@5": 96.054,
        },
    )
    DEFAULT = IMAGENET1K_V1


class EfficientNet_B4_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth",
        transforms=partial(
            ImageClassificationEval, crop_size=380, resize_size=384, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META_V1,
            "num_params": 19341616,
            "size": (380, 380),
            "acc@1": 83.384,
            "acc@5": 96.594,
        },
    )
    DEFAULT = IMAGENET1K_V1


class EfficientNet_B5_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth",
        transforms=partial(
            ImageClassificationEval, crop_size=456, resize_size=456, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META_V1,
            "num_params": 30389784,
            "size": (456, 456),
            "acc@1": 83.444,
            "acc@5": 96.628,
        },
    )
    DEFAULT = IMAGENET1K_V1


class EfficientNet_B6_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth",
        transforms=partial(
            ImageClassificationEval, crop_size=528, resize_size=528, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META_V1,
            "num_params": 43040704,
            "size": (528, 528),
            "acc@1": 84.008,
            "acc@5": 96.916,
        },
    )
    DEFAULT = IMAGENET1K_V1


class EfficientNet_B7_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth",
        transforms=partial(
            ImageClassificationEval, crop_size=600, resize_size=600, interpolation=InterpolationMode.BICUBIC
        ),
        meta={
            **_COMMON_META_V1,
            "num_params": 66347960,
            "size": (600, 600),
            "acc@1": 84.122,
            "acc@5": 96.908,
        },
    )
    DEFAULT = IMAGENET1K_V1


class EfficientNet_V2_S_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth",
        transforms=partial(
            ImageClassificationEval,
            crop_size=384,
            resize_size=384,
            interpolation=InterpolationMode.BILINEAR,
        ),
        meta={
            **_COMMON_META_V2,
            "num_params": 21458488,
            "size": (384, 384),
            "acc@1": 84.228,
            "acc@5": 96.878,
        },
    )
    DEFAULT = IMAGENET1K_V1


class EfficientNet_V2_M_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth",
        transforms=partial(
            ImageClassificationEval,
            crop_size=480,
            resize_size=480,
            interpolation=InterpolationMode.BILINEAR,
        ),
        meta={
            **_COMMON_META_V2,
            "num_params": 54139356,
            "size": (480, 480),
            "acc@1": 85.112,
            "acc@5": 97.156,
        },
    )
    DEFAULT = IMAGENET1K_V1


class EfficientNet_V2_L_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/efficientnet_v2_l-59c71312.pth",
        transforms=partial(
            ImageClassificationEval,
            crop_size=480,
            resize_size=480,
            interpolation=InterpolationMode.BICUBIC,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
        meta={
            **_COMMON_META_V2,
            "num_params": 118515272,
            "size": (480, 480),
            "acc@1": 85.808,
            "acc@5": 97.788,
        },
    )
    DEFAULT = IMAGENET1K_V1


@handle_legacy_interface(weights=("pretrained", EfficientNet_B0_Weights.IMAGENET1K_V1))
def efficientnet_b0(
    *, weights: Optional[EfficientNet_B0_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_B0_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
    return _efficientnet(inverted_residual_setting, 0.2, last_channel, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", EfficientNet_B1_Weights.IMAGENET1K_V1))
def efficientnet_b1(
    *, weights: Optional[EfficientNet_B1_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_B1_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b1", width_mult=1.0, depth_mult=1.1)
    return _efficientnet(inverted_residual_setting, 0.2, last_channel, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", EfficientNet_B2_Weights.IMAGENET1K_V1))
def efficientnet_b2(
    *, weights: Optional[EfficientNet_B2_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_B2_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b2", width_mult=1.1, depth_mult=1.2)
    return _efficientnet(inverted_residual_setting, 0.3, last_channel, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", EfficientNet_B3_Weights.IMAGENET1K_V1))
def efficientnet_b3(
    *, weights: Optional[EfficientNet_B3_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_B3_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b3", width_mult=1.2, depth_mult=1.4)
    return _efficientnet(inverted_residual_setting, 0.3, last_channel, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", EfficientNet_B4_Weights.IMAGENET1K_V1))
def efficientnet_b4(
    *, weights: Optional[EfficientNet_B4_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_B4_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b4", width_mult=1.4, depth_mult=1.8)
    return _efficientnet(inverted_residual_setting, 0.4, last_channel, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", EfficientNet_B5_Weights.IMAGENET1K_V1))
def efficientnet_b5(
    *, weights: Optional[EfficientNet_B5_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_B5_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b5", width_mult=1.6, depth_mult=2.2)
    return _efficientnet(
        inverted_residual_setting,
        0.4,
        last_channel,
        weights,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


@handle_legacy_interface(weights=("pretrained", EfficientNet_B6_Weights.IMAGENET1K_V1))
def efficientnet_b6(
    *, weights: Optional[EfficientNet_B6_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_B6_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b6", width_mult=1.8, depth_mult=2.6)
    return _efficientnet(
        inverted_residual_setting,
        0.5,
        last_channel,
        weights,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


@handle_legacy_interface(weights=("pretrained", EfficientNet_B7_Weights.IMAGENET1K_V1))
def efficientnet_b7(
    *, weights: Optional[EfficientNet_B7_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_B7_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b7", width_mult=2.0, depth_mult=3.1)
    return _efficientnet(
        inverted_residual_setting,
        0.5,
        last_channel,
        weights,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


@handle_legacy_interface(weights=("pretrained", EfficientNet_V2_S_Weights.IMAGENET1K_V1))
def efficientnet_v2_s(
    *, weights: Optional[EfficientNet_V2_S_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_V2_S_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
    return _efficientnet(
        inverted_residual_setting,
        0.2,
        last_channel,
        weights,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
        **kwargs,
    )


@handle_legacy_interface(weights=("pretrained", EfficientNet_V2_M_Weights.IMAGENET1K_V1))
def efficientnet_v2_m(
    *, weights: Optional[EfficientNet_V2_M_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_V2_M_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")
    return _efficientnet(
        inverted_residual_setting,
        0.3,
        last_channel,
        weights,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
        **kwargs,
    )


@handle_legacy_interface(weights=("pretrained", EfficientNet_V2_L_Weights.IMAGENET1K_V1))
def efficientnet_v2_l(
    *, weights: Optional[EfficientNet_V2_L_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_V2_L_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_l")
    return _efficientnet(
        inverted_residual_setting,
        0.4,
        last_channel,
        weights,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
        **kwargs,
    )
