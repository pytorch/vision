import warnings
from functools import partial
from typing import Any, List, Optional, Type, Union

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ....models.quantization.resnet import (
    QuantizableBasicBlock,
    QuantizableBottleneck,
    QuantizableResNet,
    _replace_relu,
    quantize_model,
)
from .._api import Weights, WeightEntry
from .._meta import _IMAGENET_CATEGORIES
from ..resnet import ResNet18Weights, ResNet50Weights, ResNeXt101_32x8dWeights


__all__ = [
    "QuantizableResNet",
    "QuantizedResNet18Weights",
    "QuantizedResNet50Weights",
    "QuantizedResNeXt101_32x8dWeights",
    "resnet18",
    "resnet50",
    "resnext101_32x8d",
]


def _resnet(
    block: Type[Union[QuantizableBasicBlock, QuantizableBottleneck]],
    layers: List[int],
    weights: Optional[Weights],
    progress: bool,
    quantize: bool,
    **kwargs: Any,
) -> QuantizableResNet:
    if weights is not None:
        kwargs["num_classes"] = len(weights.meta["categories"])
        if "backend" in weights.meta:
            kwargs["backend"] = weights.meta["backend"]
    backend = kwargs.pop("backend", "fbgemm")

    model = QuantizableResNet(block, layers, **kwargs)
    _replace_relu(model)
    if quantize:
        quantize_model(model, backend)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "size": (224, 224),
    "categories": _IMAGENET_CATEGORIES,
    "interpolation": InterpolationMode.BILINEAR,
    "backend": "fbgemm",
    "quantization": "ptq",
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#post-training-quantized-models",
}


class QuantizedResNet18Weights(Weights):
    ImageNet1K_FBGEMM_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/quantized/resnet18_fbgemm_16fa66dd.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "unquantized": ResNet18Weights.ImageNet1K_RefV1,
            "acc@1": 69.494,
            "acc@5": 88.882,
        },
    )


class QuantizedResNet50Weights(Weights):
    ImageNet1K_FBGEMM_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/quantized/resnet50_fbgemm_bf931d71.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "unquantized": ResNet50Weights.ImageNet1K_RefV1,
            "acc@1": 75.920,
            "acc@5": 92.814,
        },
    )
    ImageNet1K_FBGEMM_RefV2 = WeightEntry(
        url="https://download.pytorch.org/models/quantized/resnet50_fbgemm-23753f79.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "unquantized": ResNet50Weights.ImageNet1K_RefV2,
            "acc@1": 80.282,
            "acc@5": 94.976,
        },
    )


class QuantizedResNeXt101_32x8dWeights(Weights):
    ImageNet1K_FBGEMM_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/quantized/resnext101_32x8_fbgemm_09835ccf.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "unquantized": ResNeXt101_32x8dWeights.ImageNet1K_RefV1,
            "acc@1": 78.986,
            "acc@5": 94.480,
        },
    )
    ImageNet1K_FBGEMM_RefV2 = WeightEntry(
        url="https://download.pytorch.org/models/quantized/resnext101_32x8_fbgemm-ee16d00c.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "unquantized": ResNeXt101_32x8dWeights.ImageNet1K_RefV2,
            "acc@1": 82.574,
            "acc@5": 96.132,
        },
    )


def resnet18(
    weights: Optional[Union[QuantizedResNet18Weights, ResNet18Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableResNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        if kwargs.pop("pretrained"):
            weights = QuantizedResNet18Weights.ImageNet1K_FBGEMM_RefV1 if quantize else ResNet18Weights.ImageNet1K_RefV1
        else:
            weights = None

    if quantize:
        weights = QuantizedResNet18Weights.verify(weights)
    else:
        weights = ResNet18Weights.verify(weights)

    return _resnet(QuantizableBasicBlock, [2, 2, 2, 2], weights, progress, quantize, **kwargs)


def resnet50(
    weights: Optional[Union[QuantizedResNet50Weights, ResNet50Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableResNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        if kwargs.pop("pretrained"):
            weights = QuantizedResNet50Weights.ImageNet1K_FBGEMM_RefV1 if quantize else ResNet50Weights.ImageNet1K_RefV1
        else:
            weights = None

    if quantize:
        weights = QuantizedResNet50Weights.verify(weights)
    else:
        weights = ResNet50Weights.verify(weights)

    return _resnet(QuantizableBottleneck, [3, 4, 6, 3], weights, progress, quantize, **kwargs)


def resnext101_32x8d(
    weights: Optional[Union[QuantizedResNeXt101_32x8dWeights, ResNeXt101_32x8dWeights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableResNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        if kwargs.pop("pretrained"):
            weights = (
                QuantizedResNeXt101_32x8dWeights.ImageNet1K_FBGEMM_RefV1
                if quantize
                else ResNeXt101_32x8dWeights.ImageNet1K_RefV1
            )
        else:
            weights = None

    if quantize:
        weights = QuantizedResNeXt101_32x8dWeights.verify(weights)
    else:
        weights = ResNeXt101_32x8dWeights.verify(weights)

    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(QuantizableBottleneck, [3, 4, 23, 3], weights, progress, quantize, **kwargs)
