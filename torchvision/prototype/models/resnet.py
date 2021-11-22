import warnings
from functools import partial
from typing import Any, List, Optional, Type, Union

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.resnet import BasicBlock, Bottleneck, ResNet
from ._api import Weights, WeightEntry
from ._meta import _IMAGENET_CATEGORIES


__all__ = [
    "ResNet",
    "ResNet18Weights",
    "ResNet34Weights",
    "ResNet50Weights",
    "ResNet101Weights",
    "ResNet152Weights",
    "ResNeXt50_32x4dWeights",
    "ResNeXt101_32x8dWeights",
    "WideResNet50_2Weights",
    "WideResNet101_2Weights",
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
    weights: Optional[Weights],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        kwargs["num_classes"] = len(weights.meta["categories"])

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {"size": (224, 224), "categories": _IMAGENET_CATEGORIES, "interpolation": InterpolationMode.BILINEAR}


class ResNet18Weights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "acc@1": 69.758,
            "acc@5": 89.078,
        },
    )


class ResNet34Weights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/resnet34-b627a593.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "acc@1": 73.314,
            "acc@5": 91.420,
        },
    )


class ResNet50Weights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "acc@1": 76.130,
            "acc@5": 92.862,
        },
    )
    ImageNet1K_RefV2 = WeightEntry(
        url="https://download.pytorch.org/models/resnet50-f46c3f97.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "acc@1": 80.674,
            "acc@5": 95.166,
        },
    )


class ResNet101Weights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/resnet101-63fe2227.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "acc@1": 77.374,
            "acc@5": 93.546,
        },
    )
    ImageNet1K_RefV2 = WeightEntry(
        url="https://download.pytorch.org/models/resnet101-cd907fc2.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "acc@1": 81.886,
            "acc@5": 95.780,
        },
    )


class ResNet152Weights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/resnet152-394f9c45.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "acc@1": 78.312,
            "acc@5": 94.046,
        },
    )
    ImageNet1K_RefV2 = WeightEntry(
        url="https://download.pytorch.org/models/resnet152-f82ba261.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "acc@1": 82.284,
            "acc@5": 96.002,
        },
    )


class ResNeXt50_32x4dWeights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
            "acc@1": 77.618,
            "acc@5": 93.698,
        },
    )
    ImageNet1K_RefV2 = WeightEntry(
        url="https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "acc@1": 81.198,
            "acc@5": 95.340,
        },
    )


class ResNeXt101_32x8dWeights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
            "acc@1": 79.312,
            "acc@5": 94.526,
        },
    )
    ImageNet1K_RefV2 = WeightEntry(
        url="https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "acc@1": 82.834,
            "acc@5": 96.228,
        },
    )


class WideResNet50_2Weights(Weights):
    ImageNet1K_Community = WeightEntry(
        url="https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
            "acc@1": 78.468,
            "acc@5": 94.086,
        },
    )
    ImageNet1K_RefV2 = WeightEntry(
        url="https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "acc@1": 81.602,
            "acc@5": 95.758,
        },
    )


class WideResNet101_2Weights(Weights):
    ImageNet1K_Community = WeightEntry(
        url="https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
            "acc@1": 78.848,
            "acc@5": 94.284,
        },
    )
    ImageNet1K_RefV2 = WeightEntry(
        url="https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "acc@1": 82.510,
            "acc@5": 96.020,
        },
    )


def resnet18(weights: Optional[ResNet18Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = ResNet18Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None

    weights = ResNet18Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


def resnet34(weights: Optional[ResNet34Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = ResNet34Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None

    weights = ResNet34Weights.verify(weights)

    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)


def resnet50(weights: Optional[ResNet50Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = ResNet50Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = ResNet50Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def resnet101(weights: Optional[ResNet101Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = ResNet101Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None

    weights = ResNet101Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def resnet152(weights: Optional[ResNet152Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = ResNet152Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None

    weights = ResNet152Weights.verify(weights)

    return _resnet(Bottleneck, [3, 8, 36, 3], weights, progress, **kwargs)


def resnext50_32x4d(weights: Optional[ResNeXt50_32x4dWeights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = ResNeXt50_32x4dWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None

    weights = ResNeXt50_32x4dWeights.verify(weights)
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def resnext101_32x8d(weights: Optional[ResNeXt101_32x8dWeights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = ResNeXt101_32x8dWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None

    weights = ResNeXt101_32x8dWeights.verify(weights)
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def wide_resnet50_2(weights: Optional[WideResNet50_2Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = WideResNet50_2Weights.ImageNet1K_Community if kwargs.pop("pretrained") else None

    weights = WideResNet50_2Weights.verify(weights)
    kwargs["width_per_group"] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def wide_resnet101_2(weights: Optional[WideResNet101_2Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = WideResNet101_2Weights.ImageNet1K_Community if kwargs.pop("pretrained") else None

    weights = WideResNet101_2Weights.verify(weights)
    kwargs["width_per_group"] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)
