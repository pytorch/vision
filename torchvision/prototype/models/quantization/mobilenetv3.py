from functools import partial
from typing import Any, List, Optional, Union

import torch
from torchvision.prototype.transforms import ImageClassificationEval
from torchvision.transforms.functional import InterpolationMode

from ....models.quantization.mobilenetv3 import (
    InvertedResidualConfig,
    QuantizableInvertedResidual,
    QuantizableMobileNetV3,
    _replace_relu,
)
from .._api import WeightsEnum, Weights
from .._meta import _IMAGENET_CATEGORIES
from .._utils import handle_legacy_interface, _ovewrite_named_param
from ..mobilenetv3 import MobileNet_V3_Large_Weights, _mobilenet_v3_conf


__all__ = [
    "QuantizableMobileNetV3",
    "MobileNet_V3_Large_QuantizedWeights",
    "mobilenet_v3_large",
]


def _mobilenet_v3_model(
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    quantize: bool,
    **kwargs: Any,
) -> QuantizableMobileNetV3:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        if "backend" in weights.meta:
            _ovewrite_named_param(kwargs, "backend", weights.meta["backend"])
    backend = kwargs.pop("backend", "qnnpack")

    model = QuantizableMobileNetV3(inverted_residual_setting, last_channel, block=QuantizableInvertedResidual, **kwargs)
    _replace_relu(model)

    if quantize:
        model.fuse_model(is_qat=True)
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
        torch.ao.quantization.prepare_qat(model, inplace=True)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    if quantize:
        torch.ao.quantization.convert(model, inplace=True)
        model.eval()

    return model


class MobileNet_V3_Large_QuantizedWeights(WeightsEnum):
    IMAGENET1K_QNNPACK_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/mobilenet_v3_large_qnnpack-5bcacf28.pth",
        transforms=partial(ImageClassificationEval, crop_size=224),
        meta={
            "task": "image_classification",
            "architecture": "MobileNetV3",
            "publication_year": 2019,
            "num_params": 5483032,
            "size": (224, 224),
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "backend": "qnnpack",
            "quantization": "qat",
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#qat-mobilenetv3",
            "unquantized": MobileNet_V3_Large_Weights.IMAGENET1K_V1,
            "acc@1": 73.004,
            "acc@5": 90.858,
        },
    )
    DEFAULT = IMAGENET1K_QNNPACK_V1


@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: MobileNet_V3_Large_QuantizedWeights.IMAGENET1K_QNNPACK_V1
        if kwargs.get("quantize", False)
        else MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    )
)
def mobilenet_v3_large(
    *,
    weights: Optional[Union[MobileNet_V3_Large_QuantizedWeights, MobileNet_V3_Large_Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableMobileNetV3:
    weights = (MobileNet_V3_Large_QuantizedWeights if quantize else MobileNet_V3_Large_Weights).verify(weights)

    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large", **kwargs)
    return _mobilenet_v3_model(inverted_residual_setting, last_channel, weights, progress, quantize, **kwargs)
