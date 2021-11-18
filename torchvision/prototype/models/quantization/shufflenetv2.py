import warnings
from functools import partial
from typing import Any, List, Optional, Union

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ....models.quantization.shufflenetv2 import (
    QuantizableShuffleNetV2,
    _replace_relu,
    quantize_model,
)
from .._api import Weights, WeightEntry
from .._meta import _IMAGENET_CATEGORIES
from ..shufflenetv2 import ShuffleNetV2_x0_5Weights, ShuffleNetV2_x1_0Weights


__all__ = [
    "QuantizableShuffleNetV2",
    "QuantizedShuffleNetV2_x0_5Weights",
    "QuantizedShuffleNetV2_x1_0Weights",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
]


def _shufflenetv2(
    stages_repeats: List[int],
    stages_out_channels: List[int],
    weights: Optional[Weights],
    progress: bool,
    quantize: bool,
    **kwargs: Any,
) -> QuantizableShuffleNetV2:
    if weights is not None:
        kwargs["num_classes"] = len(weights.meta["categories"])
        if "backend" in weights.meta:
            kwargs["backend"] = weights.meta["backend"]
    backend = kwargs.pop("backend", "fbgemm")

    model = QuantizableShuffleNetV2(stages_repeats, stages_out_channels, **kwargs)
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


class QuantizedShuffleNetV2_x0_5Weights(Weights):
    ImageNet1K_FBGEMM_Community = WeightEntry(
        url="https://download.pytorch.org/models/quantized/shufflenetv2_x0.5_fbgemm-00845098.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "unquantized": ShuffleNetV2_x0_5Weights.ImageNet1K_Community,
            "acc@1": 57.972,
            "acc@5": 79.780,
        },
    )


class QuantizedShuffleNetV2_x1_0Weights(Weights):
    ImageNet1K_FBGEMM_Community = WeightEntry(
        url="https://download.pytorch.org/models/quantized/shufflenetv2_x1_fbgemm-db332c57.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "unquantized": ShuffleNetV2_x1_0Weights.ImageNet1K_Community,
            "acc@1": 68.360,
            "acc@5": 87.582,
        },
    )


def shufflenet_v2_x0_5(
    weights: Optional[Union[QuantizedShuffleNetV2_x0_5Weights, ShuffleNetV2_x0_5Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableShuffleNetV2:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        if kwargs.pop("pretrained"):
            weights = (
                QuantizedShuffleNetV2_x0_5Weights.ImageNet1K_FBGEMM_Community
                if quantize
                else ShuffleNetV2_x0_5Weights.ImageNet1K_Community
            )
        else:
            weights = None

    if quantize:
        weights = QuantizedShuffleNetV2_x0_5Weights.verify(weights)
    else:
        weights = ShuffleNetV2_x0_5Weights.verify(weights)

    return _shufflenetv2([4, 8, 4], [24, 48, 96, 192, 1024], weights, progress, quantize, **kwargs)


def shufflenet_v2_x1_0(
    weights: Optional[Union[QuantizedShuffleNetV2_x1_0Weights, ShuffleNetV2_x1_0Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableShuffleNetV2:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        if kwargs.pop("pretrained"):
            weights = (
                QuantizedShuffleNetV2_x1_0Weights.ImageNet1K_FBGEMM_Community
                if quantize
                else ShuffleNetV2_x1_0Weights.ImageNet1K_Community
            )
        else:
            weights = None

    if quantize:
        weights = QuantizedShuffleNetV2_x1_0Weights.verify(weights)
    else:
        weights = ShuffleNetV2_x1_0Weights.verify(weights)

    return _shufflenetv2([4, 8, 4], [24, 116, 232, 464, 1024], weights, progress, quantize, **kwargs)
