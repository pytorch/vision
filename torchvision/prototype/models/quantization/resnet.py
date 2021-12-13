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
from .._utils import handle_legacy_interface, _ovewrite_named_param
from ..resnet import ResNet18_Weights, ResNet50_Weights, ResNeXt101_32X8D_Weights


__all__ = [
    "QuantizableResNet",
    "ResNet18_QuantizedWeights",
    "ResNet50_QuantizedWeights",
    "ResNeXt101_32X8D_QuantizedWeights",
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


class ResNet18_QuantizedWeights(WeightsEnum):
    ImageNet1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/resnet18_fbgemm_16fa66dd.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "unquantized": ResNet18_Weights.ImageNet1K_V1,
            "acc@1": 69.494,
            "acc@5": 88.882,
        },
    )
    default = ImageNet1K_FBGEMM_V1


class ResNet50_QuantizedWeights(WeightsEnum):
    ImageNet1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/resnet50_fbgemm_bf931d71.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "unquantized": ResNet50_Weights.ImageNet1K_V1,
            "acc@1": 75.920,
            "acc@5": 92.814,
        },
    )
    ImageNet1K_FBGEMM_V2 = Weights(
        url="https://download.pytorch.org/models/quantized/resnet50_fbgemm-23753f79.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "unquantized": ResNet50_Weights.ImageNet1K_V2,
            "acc@1": 80.282,
            "acc@5": 94.976,
        },
    )
    default = ImageNet1K_FBGEMM_V2


class ResNeXt101_32X8D_QuantizedWeights(WeightsEnum):
    ImageNet1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/resnext101_32x8_fbgemm_09835ccf.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "unquantized": ResNeXt101_32X8D_Weights.ImageNet1K_V1,
            "acc@1": 78.986,
            "acc@5": 94.480,
        },
    )
    ImageNet1K_FBGEMM_V2 = Weights(
        url="https://download.pytorch.org/models/quantized/resnext101_32x8_fbgemm-ee16d00c.pth",
        transforms=partial(ImageNetEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "unquantized": ResNeXt101_32X8D_Weights.ImageNet1K_V2,
            "acc@1": 82.574,
            "acc@5": 96.132,
        },
    )
    default = ImageNet1K_FBGEMM_V2


@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: ResNet18_QuantizedWeights.ImageNet1K_FBGEMM_V1
        if kwargs.get("quantize", False)
        else ResNet18_Weights.ImageNet1K_V1,
    )
)
def resnet18(
    *,
    weights: Optional[Union[ResNet18_QuantizedWeights, ResNet18_Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableResNet:
    weights = (ResNet18_QuantizedWeights if quantize else ResNet18_Weights).verify(weights)

    return _resnet(QuantizableBasicBlock, [2, 2, 2, 2], weights, progress, quantize, **kwargs)


@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: ResNet50_QuantizedWeights.ImageNet1K_FBGEMM_V1
        if kwargs.get("quantize", False)
        else ResNet50_Weights.ImageNet1K_V1,
    )
)
def resnet50(
    *,
    weights: Optional[Union[ResNet50_QuantizedWeights, ResNet50_Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableResNet:
    weights = (ResNet50_QuantizedWeights if quantize else ResNet50_Weights).verify(weights)

    return _resnet(QuantizableBottleneck, [3, 4, 6, 3], weights, progress, quantize, **kwargs)


@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: ResNeXt101_32X8D_QuantizedWeights.ImageNet1K_FBGEMM_V1
        if kwargs.get("quantize", False)
        else ResNeXt101_32X8D_Weights.ImageNet1K_V1,
    )
)
def resnext101_32x8d(
    *,
    weights: Optional[Union[ResNeXt101_32X8D_QuantizedWeights, ResNeXt101_32X8D_Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableResNet:
    weights = (ResNeXt101_32X8D_QuantizedWeights if quantize else ResNeXt101_32X8D_Weights).verify(weights)

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 8)
    return _resnet(QuantizableBottleneck, [3, 4, 23, 3], weights, progress, quantize, **kwargs)
