import warnings
from functools import partial
from typing import Any, Optional, Union

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ....models.quantization.mobilenetv2 import (
    QuantizableInvertedResidual,
    QuantizableMobileNetV2,
    _replace_relu,
    quantize_model,
)
from .._api import Weights, WeightEntry
from .._meta import _IMAGENET_CATEGORIES
from ..mobilenetv2 import MobileNetV2Weights


__all__ = [
    "QuantizableMobileNetV2",
    "QuantizedMobileNetV2Weights",
    "mobilenet_v2",
]


class QuantizedMobileNetV2Weights(Weights):
    ImageNet1K_QNNPACK_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/quantized/mobilenet_v2_qnnpack_37f702c5.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            "size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "backend": "qnnpack",
            "quantization": "qat",
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#qat-mobilenetv2",
            "unquantized": MobileNetV2Weights.ImageNet1K_RefV1,
            "acc@1": 71.658,
            "acc@5": 90.150,
        },
    )


def mobilenet_v2(
    weights: Optional[Union[QuantizedMobileNetV2Weights, MobileNetV2Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableMobileNetV2:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        if kwargs.pop("pretrained"):
            weights = (
                QuantizedMobileNetV2Weights.ImageNet1K_QNNPACK_RefV1
                if quantize
                else MobileNetV2Weights.ImageNet1K_RefV1
            )
        else:
            weights = None

    if quantize:
        weights = QuantizedMobileNetV2Weights.verify(weights)
    else:
        weights = MobileNetV2Weights.verify(weights)

    if weights is not None:
        kwargs["num_classes"] = len(weights.meta["categories"])
        if "backend" in weights.meta:
            kwargs["backend"] = weights.meta["backend"]
    backend = kwargs.pop("backend", "qnnpack")

    model = QuantizableMobileNetV2(block=QuantizableInvertedResidual, **kwargs)
    _replace_relu(model)
    if quantize:
        quantize_model(model, backend)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
