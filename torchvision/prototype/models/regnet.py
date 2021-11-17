import warnings
from functools import partial
from typing import Any, Optional

from torch import nn
from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.regnet import RegNet, BlockParams
from ._api import Weights, WeightEntry
from ._meta import _IMAGENET_CATEGORIES


__all__ = [
    "RegNet",
    "RegNet_y_400mfWeights",
    "RegNet_y_800mfWeights",
    "RegNet_y_1_6gfWeights",
    "RegNet_y_3_2gfWeights",
    "RegNet_y_8gfWeights",
    "RegNet_y_16gfWeights",
    "RegNet_y_32gfWeights",
    "RegNet_x_400mfWeights",
    "RegNet_x_800mfWeights",
    "RegNet_x_1_6gfWeights",
    "RegNet_x_3_2gfWeights",
    "RegNet_x_8gfWeights",
    "RegNet_x_16gfWeights",
    "RegNet_x_32gfWeights",
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
    weights: Optional[Weights],
    progress: bool,
    **kwargs: Any,
) -> RegNet:
    if weights is not None:
        kwargs["num_classes"] = len(weights.meta["categories"])

    norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
    model = RegNet(block_params, norm_layer=norm_layer, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


class RegNet_y_400mfWeights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/regnet_y_400mf-c65dace8.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#small-models",
            "acc@1": 74.046,
            "acc@5": 91.716,
        },
    )


class RegNet_y_800mfWeights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/regnet_y_800mf-1b27b58c.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#small-models",
            "acc@1": 76.420,
            "acc@5": 93.136,
        },
    )


class RegNet_y_1_6gfWeights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/regnet_y_1_6gf-b11a554e.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#small-models",
            "acc@1": 77.950,
            "acc@5": 93.966,
        },
    )


class RegNet_y_3_2gfWeights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/regnet_y_3_2gf-b5a9779c.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#medium-models",
            "acc@1": 78.948,
            "acc@5": 94.576,
        },
    )


class RegNet_y_8gfWeights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/regnet_y_8gf-d0d0e4a8.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#medium-models",
            "acc@1": 80.032,
            "acc@5": 95.048,
        },
    )


class RegNet_y_16gfWeights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/regnet_y_16gf-9e6ed7dd.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#large-models",
            "acc@1": 80.424,
            "acc@5": 95.240,
        },
    )


class RegNet_y_32gfWeights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/regnet_y_32gf-4dee3f7a.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#large-models",
            "acc@1": 80.878,
            "acc@5": 95.340,
        },
    )


class RegNet_x_400mfWeights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/regnet_x_400mf-adf1edd5.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#small-models",
            "acc@1": 72.834,
            "acc@5": 90.950,
        },
    )


class RegNet_x_800mfWeights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/regnet_x_800mf-ad17e45c.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#small-models",
            "acc@1": 75.212,
            "acc@5": 92.348,
        },
    )


class RegNet_x_1_6gfWeights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/regnet_x_1_6gf-e3633e7f.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#small-models",
            "acc@1": 77.040,
            "acc@5": 93.440,
        },
    )


class RegNet_x_3_2gfWeights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/regnet_x_3_2gf-f342aeae.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#medium-models",
            "acc@1": 78.364,
            "acc@5": 93.992,
        },
    )


class RegNet_x_8gfWeights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/regnet_x_8gf-03ceed89.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#medium-models",
            "acc@1": 79.344,
            "acc@5": 94.686,
        },
    )


class RegNet_x_16gfWeights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/regnet_x_16gf-2007eb11.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#medium-models",
            "acc@1": 80.058,
            "acc@5": 94.944,
        },
    )


class RegNet_x_32gfWeights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/regnet_x_32gf-9d47f8d0.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#large-models",
            "acc@1": 80.622,
            "acc@5": 95.248,
        },
    )


def regnet_y_400mf(weights: Optional[RegNet_y_400mfWeights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = RegNet_y_400mfWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = RegNet_y_400mfWeights.verify(weights)

    params = BlockParams.from_init_params(depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25, **kwargs)
    return _regnet(params, weights, progress, **kwargs)


def regnet_y_800mf(weights: Optional[RegNet_y_800mfWeights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = RegNet_y_800mfWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = RegNet_y_800mfWeights.verify(weights)

    params = BlockParams.from_init_params(depth=14, w_0=56, w_a=38.84, w_m=2.4, group_width=16, se_ratio=0.25, **kwargs)
    return _regnet(params, weights, progress, **kwargs)


def regnet_y_1_6gf(weights: Optional[RegNet_y_1_6gfWeights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = RegNet_y_1_6gfWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = RegNet_y_1_6gfWeights.verify(weights)

    params = BlockParams.from_init_params(
        depth=27, w_0=48, w_a=20.71, w_m=2.65, group_width=24, se_ratio=0.25, **kwargs
    )
    return _regnet(params, weights, progress, **kwargs)


def regnet_y_3_2gf(weights: Optional[RegNet_y_3_2gfWeights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = RegNet_y_3_2gfWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = RegNet_y_3_2gfWeights.verify(weights)
    params = BlockParams.from_init_params(
        depth=21, w_0=80, w_a=42.63, w_m=2.66, group_width=24, se_ratio=0.25, **kwargs
    )
    return _regnet(params, weights, progress, **kwargs)


def regnet_y_8gf(weights: Optional[RegNet_y_8gfWeights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = RegNet_y_8gfWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = RegNet_y_8gfWeights.verify(weights)
    params = BlockParams.from_init_params(
        depth=17, w_0=192, w_a=76.82, w_m=2.19, group_width=56, se_ratio=0.25, **kwargs
    )
    return _regnet(params, weights, progress, **kwargs)


def regnet_y_16gf(weights: Optional[RegNet_y_16gfWeights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = RegNet_y_16gfWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = RegNet_y_16gfWeights.verify(weights)
    params = BlockParams.from_init_params(
        depth=18, w_0=200, w_a=106.23, w_m=2.48, group_width=112, se_ratio=0.25, **kwargs
    )
    return _regnet(params, weights, progress, **kwargs)


def regnet_y_32gf(weights: Optional[RegNet_y_32gfWeights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = RegNet_y_32gfWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = RegNet_y_32gfWeights.verify(weights)
    params = BlockParams.from_init_params(
        depth=20, w_0=232, w_a=115.89, w_m=2.53, group_width=232, se_ratio=0.25, **kwargs
    )
    return _regnet(params, weights, progress, **kwargs)


def regnet_x_400mf(weights: Optional[RegNet_x_400mfWeights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = RegNet_x_400mfWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = RegNet_x_400mfWeights.verify(weights)
    params = BlockParams.from_init_params(depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16, **kwargs)

    return _regnet(params, weights, progress, **kwargs)


def regnet_x_800mf(weights: Optional[RegNet_x_800mfWeights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = RegNet_x_800mfWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = RegNet_x_800mfWeights.verify(weights)
    params = BlockParams.from_init_params(depth=16, w_0=56, w_a=35.73, w_m=2.28, group_width=16, **kwargs)

    return _regnet(params, weights, progress, **kwargs)


def regnet_x_1_6gf(weights: Optional[RegNet_x_1_6gfWeights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = RegNet_x_1_6gfWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = RegNet_x_1_6gfWeights.verify(weights)
    params = BlockParams.from_init_params(depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24, **kwargs)

    return _regnet(params, weights, progress, **kwargs)


def regnet_x_3_2gf(weights: Optional[RegNet_x_3_2gfWeights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = RegNet_x_3_2gfWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = RegNet_x_3_2gfWeights.verify(weights)
    params = BlockParams.from_init_params(depth=25, w_0=88, w_a=26.31, w_m=2.25, group_width=48, **kwargs)

    return _regnet(params, weights, progress, **kwargs)


def regnet_x_8gf(weights: Optional[RegNet_x_8gfWeights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = RegNet_x_8gfWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = RegNet_x_8gfWeights.verify(weights)
    params = BlockParams.from_init_params(depth=23, w_0=80, w_a=49.56, w_m=2.88, group_width=120, **kwargs)

    return _regnet(params, weights, progress, **kwargs)


def regnet_x_16gf(weights: Optional[RegNet_x_16gfWeights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = RegNet_x_16gfWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = RegNet_x_16gfWeights.verify(weights)
    params = BlockParams.from_init_params(depth=22, w_0=216, w_a=55.59, w_m=2.1, group_width=128, **kwargs)

    return _regnet(params, weights, progress, **kwargs)


def regnet_x_32gf(weights: Optional[RegNet_x_32gfWeights] = None, progress: bool = True, **kwargs: Any) -> RegNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = RegNet_x_32gfWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = RegNet_x_32gfWeights.verify(weights)
    params = BlockParams.from_init_params(depth=23, w_0=320, w_a=69.86, w_m=2.0, group_width=168, **kwargs)

    return _regnet(params, weights, progress, **kwargs)
