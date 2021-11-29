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
from .._api import WeightsEnum, Weights
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _deprecated_param, _deprecated_positional, _ovewrite_named_param
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
    weights: Optional[WeightsEnum],
    progress: bool,
    quantize: bool,
    **kwargs: Any,
) -> QuantizableResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        if "backend" in weights.meta:
            _ovewrite_named_param(kwargs, "backend", weights.meta["backend"])
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


class QuantizedResNet18Weights(WeightsEnum):
    ImageNet1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/resnet18_fbgemm_16fa66dd.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "unquantized": ResNet18Weights.ImageNet1K_V1,
            "acc@1": 69.494,
            "acc@5": 88.882,
        },
        default=True,
    )


class QuantizedResNet50Weights(WeightsEnum):
    ImageNet1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/resnet50_fbgemm_bf931d71.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "unquantized": ResNet50Weights.ImageNet1K_V1,
            "acc@1": 75.920,
            "acc@5": 92.814,
        },
        default=False,
    )
    ImageNet1K_FBGEMM_V2 = Weights(
        url="https://download.pytorch.org/models/quantized/resnet50_fbgemm-23753f79.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "unquantized": ResNet50Weights.ImageNet1K_V2,
            "acc@1": 80.282,
            "acc@5": 94.976,
        },
        default=True,
    )


class QuantizedResNeXt101_32x8dWeights(WeightsEnum):
    ImageNet1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/resnext101_32x8_fbgemm_09835ccf.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "unquantized": ResNeXt101_32x8dWeights.ImageNet1K_V1,
            "acc@1": 78.986,
            "acc@5": 94.480,
        },
        default=False,
    )
    ImageNet1K_FBGEMM_V2 = Weights(
        url="https://download.pytorch.org/models/quantized/resnext101_32x8_fbgemm-ee16d00c.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "unquantized": ResNeXt101_32x8dWeights.ImageNet1K_V2,
            "acc@1": 82.574,
            "acc@5": 96.132,
        },
        default=True,
    )


def resnet18(
    weights: Optional[Union[QuantizedResNet18Weights, ResNet18Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableResNet:
    if type(weights) == bool and weights:
        _deprecated_positional(kwargs, "pretrained", "weights", True)
    if "pretrained" in kwargs:
        default_value = QuantizedResNet18Weights.ImageNet1K_FBGEMM_V1 if quantize else ResNet18Weights.ImageNet1K_V1
        weights = _deprecated_param(kwargs, "pretrained", "weights", default_value)  # type: ignore[assignment]
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
    if type(weights) == bool and weights:
        _deprecated_positional(kwargs, "pretrained", "weights", True)
    if "pretrained" in kwargs:
        default_value = QuantizedResNet50Weights.ImageNet1K_FBGEMM_V1 if quantize else ResNet50Weights.ImageNet1K_V1
        weights = _deprecated_param(kwargs, "pretrained", "weights", default_value)  # type: ignore[assignment]
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
    if type(weights) == bool and weights:
        _deprecated_positional(kwargs, "pretrained", "weights", True)
    if "pretrained" in kwargs:
        default_value = (
            QuantizedResNeXt101_32x8dWeights.ImageNet1K_FBGEMM_V1 if quantize else ResNeXt101_32x8dWeights.ImageNet1K_V1
        )
        weights = _deprecated_param(kwargs, "pretrained", "weights", default_value)  # type: ignore[assignment]
    if quantize:
        weights = QuantizedResNeXt101_32x8dWeights.verify(weights)
    else:
        weights = ResNeXt101_32x8dWeights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 8)
    return _resnet(QuantizableBottleneck, [3, 4, 23, 3], weights, progress, quantize, **kwargs)
