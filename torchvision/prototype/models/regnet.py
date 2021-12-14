from functools import partial
from typing import Any, Optional

from torch import nn
from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.regnet import RegNet, BlockParams
from ._api import WeightsEnum, Weights
from ._meta import _IMAGENET_CATEGORIES
from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = [
    "RegNet",
    "RegNet_Y_400MF_Weights",
    "RegNet_Y_800MF_Weights",
    "RegNet_Y_1_6GF_Weights",
    "RegNet_Y_3_2GF_Weights",
    "RegNet_Y_8GF_Weights",
    "RegNet_Y_16GF_Weights",
    "RegNet_Y_32GF_Weights",
    "RegNet_X_400MF_Weights",
    "RegNet_X_800MF_Weights",
    "RegNet_X_1_6GF_Weights",
    "RegNet_X_3_2GF_Weights",
    "RegNet_X_8GF_Weights",
    "RegNet_X_16GF_Weights",
    "RegNet_X_32GF_Weights",
    "regnet_y_400mf",
    "regnet_y_800mf",
    "regnet_y_1_6gf",
    "regnet_y_3_2gf",
    "regnet_y_8gf",
    "regnet_y_16gf",
    "regnet_y_32gf",
    "regnet_x_400mf",
    "regnet_x_800mf",
    "regnet_x_1_6gf",
    "regnet_x_3_2gf",
    "regnet_x_8gf",
    "regnet_x_16gf",
    "regnet_x_32gf",
]

_COMMON_META = {"size": (224, 224), "categories": _IMAGENET_CATEGORIES, "interpolation": InterpolationMode.BILINEAR}


def _regnet(
    block_params: BlockParams,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> RegNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
    model = RegNet(block_params, norm_layer=norm_layer, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


class RegNet_Y_400MF_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/regnet_y_400mf-c65dace8.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#small-models",
            "acc@1": 74.046,
            "acc@5": 91.716,
        },
    )
    default = ImageNet1K_V1


class RegNet_Y_800MF_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/regnet_y_800mf-1b27b58c.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#small-models",
            "acc@1": 76.420,
            "acc@5": 93.136,
        },
    )
    default = ImageNet1K_V1


class RegNet_Y_1_6GF_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/regnet_y_1_6gf-b11a554e.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#small-models",
            "acc@1": 77.950,
            "acc@5": 93.966,
        },
    )
    default = ImageNet1K_V1


class RegNet_Y_3_2GF_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/regnet_y_3_2gf-b5a9779c.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#medium-models",
            "acc@1": 78.948,
            "acc@5": 94.576,
        },
    )
    default = ImageNet1K_V1


class RegNet_Y_8GF_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/regnet_y_8gf-d0d0e4a8.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#medium-models",
            "acc@1": 80.032,
            "acc@5": 95.048,
        },
    )
    default = ImageNet1K_V1


class RegNet_Y_16GF_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/regnet_y_16gf-9e6ed7dd.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#large-models",
            "acc@1": 80.424,
            "acc@5": 95.240,
        },
    )
    default = ImageNet1K_V1


class RegNet_Y_32GF_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/regnet_y_32gf-4dee3f7a.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#large-models",
            "acc@1": 80.878,
            "acc@5": 95.340,
        },
    )
    default = ImageNet1K_V1


class RegNet_X_400MF_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/regnet_x_400mf-adf1edd5.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#small-models",
            "acc@1": 72.834,
            "acc@5": 90.950,
        },
    )
    default = ImageNet1K_V1


class RegNet_X_800MF_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/regnet_x_800mf-ad17e45c.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#small-models",
            "acc@1": 75.212,
            "acc@5": 92.348,
        },
    )
    default = ImageNet1K_V1


class RegNet_X_1_6GF_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/regnet_x_1_6gf-e3633e7f.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#small-models",
            "acc@1": 77.040,
            "acc@5": 93.440,
        },
    )
    default = ImageNet1K_V1


class RegNet_X_3_2GF_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/regnet_x_3_2gf-f342aeae.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#medium-models",
            "acc@1": 78.364,
            "acc@5": 93.992,
        },
    )
    default = ImageNet1K_V1


class RegNet_X_8GF_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/regnet_x_8gf-03ceed89.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#medium-models",
            "acc@1": 79.344,
            "acc@5": 94.686,
        },
    )
    default = ImageNet1K_V1


class RegNet_X_16GF_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/regnet_x_16gf-2007eb11.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#medium-models",
            "acc@1": 80.058,
            "acc@5": 94.944,
        },
    )
    default = ImageNet1K_V1


class RegNet_X_32GF_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/regnet_x_32gf-9d47f8d0.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#large-models",
            "acc@1": 80.622,
            "acc@5": 95.248,
        },
    )
    default = ImageNet1K_V1


@handle_legacy_interface(weights=("pretrained", RegNet_Y_400MF_Weights.ImageNet1K_V1))
def regnet_y_400mf(*, weights: Optional[RegNet_Y_400MF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    weights = RegNet_Y_400MF_Weights.verify(weights)

    params = BlockParams.from_init_params(depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25, **kwargs)
    return _regnet(params, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", RegNet_Y_800MF_Weights.ImageNet1K_V1))
def regnet_y_800mf(*, weights: Optional[RegNet_Y_800MF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    weights = RegNet_Y_800MF_Weights.verify(weights)

    params = BlockParams.from_init_params(depth=14, w_0=56, w_a=38.84, w_m=2.4, group_width=16, se_ratio=0.25, **kwargs)
    return _regnet(params, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", RegNet_Y_1_6GF_Weights.ImageNet1K_V1))
def regnet_y_1_6gf(*, weights: Optional[RegNet_Y_1_6GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    weights = RegNet_Y_1_6GF_Weights.verify(weights)

    params = BlockParams.from_init_params(
        depth=27, w_0=48, w_a=20.71, w_m=2.65, group_width=24, se_ratio=0.25, **kwargs
    )
    return _regnet(params, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", RegNet_Y_3_2GF_Weights.ImageNet1K_V1))
def regnet_y_3_2gf(*, weights: Optional[RegNet_Y_3_2GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    weights = RegNet_Y_3_2GF_Weights.verify(weights)

    params = BlockParams.from_init_params(
        depth=21, w_0=80, w_a=42.63, w_m=2.66, group_width=24, se_ratio=0.25, **kwargs
    )
    return _regnet(params, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", RegNet_Y_8GF_Weights.ImageNet1K_V1))
def regnet_y_8gf(*, weights: Optional[RegNet_Y_8GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    weights = RegNet_Y_8GF_Weights.verify(weights)

    params = BlockParams.from_init_params(
        depth=17, w_0=192, w_a=76.82, w_m=2.19, group_width=56, se_ratio=0.25, **kwargs
    )
    return _regnet(params, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", RegNet_Y_16GF_Weights.ImageNet1K_V1))
def regnet_y_16gf(*, weights: Optional[RegNet_Y_16GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    weights = RegNet_Y_16GF_Weights.verify(weights)

    params = BlockParams.from_init_params(
        depth=18, w_0=200, w_a=106.23, w_m=2.48, group_width=112, se_ratio=0.25, **kwargs
    )
    return _regnet(params, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", RegNet_Y_32GF_Weights.ImageNet1K_V1))
def regnet_y_32gf(*, weights: Optional[RegNet_Y_32GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    weights = RegNet_Y_32GF_Weights.verify(weights)

    params = BlockParams.from_init_params(
        depth=20, w_0=232, w_a=115.89, w_m=2.53, group_width=232, se_ratio=0.25, **kwargs
    )
    return _regnet(params, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", RegNet_X_400MF_Weights.ImageNet1K_V1))
def regnet_x_400mf(*, weights: Optional[RegNet_X_400MF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    weights = RegNet_X_400MF_Weights.verify(weights)

    params = BlockParams.from_init_params(depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16, **kwargs)
    return _regnet(params, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", RegNet_X_800MF_Weights.ImageNet1K_V1))
def regnet_x_800mf(*, weights: Optional[RegNet_X_800MF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    weights = RegNet_X_800MF_Weights.verify(weights)

    params = BlockParams.from_init_params(depth=16, w_0=56, w_a=35.73, w_m=2.28, group_width=16, **kwargs)
    return _regnet(params, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", RegNet_X_1_6GF_Weights.ImageNet1K_V1))
def regnet_x_1_6gf(*, weights: Optional[RegNet_X_1_6GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    weights = RegNet_X_1_6GF_Weights.verify(weights)

    params = BlockParams.from_init_params(depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24, **kwargs)
    return _regnet(params, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", RegNet_X_3_2GF_Weights.ImageNet1K_V1))
def regnet_x_3_2gf(*, weights: Optional[RegNet_X_3_2GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    weights = RegNet_X_3_2GF_Weights.verify(weights)

    params = BlockParams.from_init_params(depth=25, w_0=88, w_a=26.31, w_m=2.25, group_width=48, **kwargs)
    return _regnet(params, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", RegNet_X_8GF_Weights.ImageNet1K_V1))
def regnet_x_8gf(*, weights: Optional[RegNet_X_8GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    weights = RegNet_X_8GF_Weights.verify(weights)

    params = BlockParams.from_init_params(depth=23, w_0=80, w_a=49.56, w_m=2.88, group_width=120, **kwargs)
    return _regnet(params, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", RegNet_X_16GF_Weights.ImageNet1K_V1))
def regnet_x_16gf(*, weights: Optional[RegNet_X_16GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    weights = RegNet_X_16GF_Weights.verify(weights)

    params = BlockParams.from_init_params(depth=22, w_0=216, w_a=55.59, w_m=2.1, group_width=128, **kwargs)
    return _regnet(params, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", RegNet_X_32GF_Weights.ImageNet1K_V1))
def regnet_x_32gf(*, weights: Optional[RegNet_X_32GF_Weights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    weights = RegNet_X_32GF_Weights.verify(weights)

    params = BlockParams.from_init_params(depth=23, w_0=320, w_a=69.86, w_m=2.0, group_width=168, **kwargs)
    return _regnet(params, weights, progress, **kwargs)
