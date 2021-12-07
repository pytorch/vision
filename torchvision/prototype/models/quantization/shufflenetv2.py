from functools import partial
from typing import Any, List, Optional, Union

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ....models.quantization.shufflenetv2 import (
    QuantizableShuffleNetV2,
    _replace_relu,
    quantize_model,
)
from .._api import WeightsEnum, Weights
from .._meta import _IMAGENET_CATEGORIES
from .._utils import handle_legacy_interface, _ovewrite_named_param
from ..shufflenetv2 import ShuffleNet_V2_X0_5_Weights, ShuffleNet_V2_X1_0_Weights


__all__ = [
    "QuantizableShuffleNetV2",
    "ShuffleNet_V2_X0_5_QuantizedWeights",
    "ShuffleNet_V2_X1_0_QuantizedWeights",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
]


def _shufflenetv2(
    stages_repeats: List[int],
    stages_out_channels: List[int],
    *,
    weights: Optional[WeightsEnum],
    progress: bool,
    quantize: bool,
    **kwargs: Any,
) -> QuantizableShuffleNetV2:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        if "backend" in weights.meta:
            _ovewrite_named_param(kwargs, "backend", weights.meta["backend"])
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


class ShuffleNet_V2_X0_5_QuantizedWeights(WeightsEnum):
    ImageNet1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/shufflenetv2_x0.5_fbgemm-00845098.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "unquantized": ShuffleNet_V2_X0_5_Weights.ImageNet1K_V1,
            "acc@1": 57.972,
            "acc@5": 79.780,
        },
    )
    default = ImageNet1K_FBGEMM_V1


class ShuffleNet_V2_X1_0_QuantizedWeights(WeightsEnum):
    ImageNet1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/shufflenetv2_x1_fbgemm-db332c57.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_COMMON_META,
            "unquantized": ShuffleNet_V2_X1_0_Weights.ImageNet1K_V1,
            "acc@1": 68.360,
            "acc@5": 87.582,
        },
    )
    default = ImageNet1K_FBGEMM_V1


@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: ShuffleNet_V2_X0_5_QuantizedWeights.ImageNet1K_FBGEMM_V1
        if kwargs.get("quantize", False)
        else ShuffleNet_V2_X0_5_Weights.ImageNet1K_V1,
    )
)
def shufflenet_v2_x0_5(
    *,
    weights: Optional[Union[ShuffleNet_V2_X0_5_QuantizedWeights, ShuffleNet_V2_X0_5_Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableShuffleNetV2:
    weights = (ShuffleNet_V2_X0_5_QuantizedWeights if quantize else ShuffleNet_V2_X0_5_Weights).verify(weights)
    return _shufflenetv2(
        [4, 8, 4], [24, 48, 96, 192, 1024], weights=weights, progress=progress, quantize=quantize, **kwargs
    )


@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: ShuffleNet_V2_X1_0_QuantizedWeights.ImageNet1K_FBGEMM_V1
        if kwargs.get("quantize", False)
        else ShuffleNet_V2_X1_0_Weights.ImageNet1K_V1,
    )
)
def shufflenet_v2_x1_0(
    *,
    weights: Optional[Union[ShuffleNet_V2_X1_0_QuantizedWeights, ShuffleNet_V2_X1_0_Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableShuffleNetV2:
    weights = (ShuffleNet_V2_X1_0_QuantizedWeights if quantize else ShuffleNet_V2_X1_0_Weights).verify(weights)
    return _shufflenetv2(
        [4, 8, 4], [24, 116, 232, 464, 1024], weights=weights, progress=progress, quantize=quantize, **kwargs
    )
