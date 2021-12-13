from functools import partial
from typing import Any, List, Optional, Type, Union

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.resnet import BasicBlock, Bottleneck, ResNet
from ._api import WeightsEnum, Weights
from ._meta import _IMAGENET_CATEGORIES
from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = [
    "ResNet",
    "ResNet18_Weights",
    "ResNet34_Weights",
    "ResNet50_Weights",
    "ResNet101_Weights",
    "ResNet152_Weights",
    "ResNeXt50_32X4D_Weights",
    "ResNeXt101_32X8D_Weights",
    "Wide_ResNet50_2_Weights",
    "Wide_ResNet101_2_Weights",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {"size": (224, 224), "categories": _IMAGENET_CATEGORIES, "interpolation": InterpolationMode.BILINEAR}


class ResNet18_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "acc@1": 69.758,
            "acc@5": 89.078,
        },
    )
    default = ImageNet1K_V1


class ResNet34_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet34-b627a593.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "acc@1": 73.314,
            "acc@5": 91.420,
        },
    )
    default = ImageNet1K_V1


class ResNet50_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "acc@1": 76.130,
            "acc@5": 92.862,
        },
    )
    ImageNet1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet50-f46c3f97.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "acc@1": 80.674,
            "acc@5": 95.166,
        },
    )
    default = ImageNet1K_V2


class ResNet101_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet101-63fe2227.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "acc@1": 77.374,
            "acc@5": 93.546,
        },
    )
    ImageNet1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet101-cd907fc2.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "acc@1": 81.886,
            "acc@5": 95.780,
        },
    )
    default = ImageNet1K_V2


class ResNet152_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet152-394f9c45.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "acc@1": 78.312,
            "acc@5": 94.046,
        },
    )
    ImageNet1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet152-f82ba261.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "acc@1": 82.284,
            "acc@5": 96.002,
        },
    )
    default = ImageNet1K_V2


class ResNeXt50_32X4D_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
            "acc@1": 77.618,
            "acc@5": 93.698,
        },
    )
    ImageNet1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "acc@1": 81.198,
            "acc@5": 95.340,
        },
    )
    default = ImageNet1K_V2


class ResNeXt101_32X8D_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
            "acc@1": 79.312,
            "acc@5": 94.526,
        },
    )
    ImageNet1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "acc@1": 82.834,
            "acc@5": 96.228,
        },
    )
    default = ImageNet1K_V2


class Wide_ResNet50_2_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
            "acc@1": 78.468,
            "acc@5": 94.086,
        },
    )
    ImageNet1K_V2 = Weights(
        url="https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "acc@1": 81.602,
            "acc@5": 95.758,
        },
    )
    default = ImageNet1K_V2


class Wide_ResNet101_2_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
            "acc@1": 78.848,
            "acc@5": 94.284,
        },
    )
    ImageNet1K_V2 = Weights(
        url="https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "acc@1": 82.510,
            "acc@5": 96.020,
        },
    )
    default = ImageNet1K_V2


@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.ImageNet1K_V1))
def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNet34_Weights.ImageNet1K_V1))
def resnet34(*, weights: Optional[ResNet34_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet34_Weights.verify(weights)

    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNet50_Weights.ImageNet1K_V1))
def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet50_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNet101_Weights.ImageNet1K_V1))
def resnet101(*, weights: Optional[ResNet101_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet101_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNet152_Weights.ImageNet1K_V1))
def resnet152(*, weights: Optional[ResNet152_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet152_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 8, 36, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNeXt50_32X4D_Weights.ImageNet1K_V1))
def resnext50_32x4d(
    *, weights: Optional[ResNeXt50_32X4D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    weights = ResNeXt50_32X4D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNeXt101_32X8D_Weights.ImageNet1K_V1))
def resnext101_32x8d(
    *, weights: Optional[ResNeXt101_32X8D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    weights = ResNeXt101_32X8D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 8)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", Wide_ResNet50_2_Weights.ImageNet1K_V1))
def wide_resnet50_2(
    *, weights: Optional[Wide_ResNet50_2_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    weights = Wide_ResNet50_2_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", Wide_ResNet101_2_Weights.ImageNet1K_V1))
def wide_resnet101_2(
    *, weights: Optional[Wide_ResNet101_2_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    weights = Wide_ResNet101_2_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)
