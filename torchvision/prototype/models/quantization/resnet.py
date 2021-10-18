import warnings
from functools import partial
from typing import Any, List, Optional, Type, Union

from ....models.quantization.resnet import (
    QuantizableBasicBlock,
    QuantizableBottleneck,
    QuantizableResNet,
    _replace_relu,
    quantize_model,
)
from ...transforms.presets import ImageNetEval
from .._api import Weights, WeightEntry
from .._meta import _IMAGENET_CATEGORIES
from ..resnet import ResNet50Weights


__all__ = ["QuantizableResNet", "QuantizedResNet50Weights", "resnet50"]


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
        model.load_state_dict(weights.state_dict(progress=progress))

    return model


_common_meta = {
    "size": (224, 224),
    "categories": _IMAGENET_CATEGORIES,
    "backend": "fbgemm",
}


class QuantizedResNet50Weights(Weights):
    ImageNet1K_FBGEMM_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/quantized/resnet50_fbgemm_bf931d71.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_common_meta,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#quantized",
            "acc@1": 75.920,
            "acc@5": 92.814,
        },
    )


def resnet50(
    weights: Optional[Union[QuantizedResNet50Weights, ResNet50Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableResNet:
    if "pretrained" in kwargs:
        warnings.warn("The argument pretrained is deprecated, please use weights instead.")
        if kwargs.pop("pretrained"):
            weights = QuantizedResNet50Weights.ImageNet1K_FBGEMM_RefV1 if quantize else ResNet50Weights.ImageNet1K_RefV1
        else:
            weights = None

    if quantize:
        weights = QuantizedResNet50Weights.verify(weights)
    else:
        weights = ResNet50Weights.verify(weights)

    return _resnet(QuantizableBottleneck, [3, 4, 6, 3], weights, progress, quantize, **kwargs)
