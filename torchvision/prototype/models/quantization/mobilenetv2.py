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
from .._api import WeightsEnum, Weights
from .._meta import _IMAGENET_CATEGORIES
from .._utils import handle_legacy_interface, _ovewrite_named_param
from ..mobilenetv2 import MobileNet_V2_Weights


__all__ = [
    "QuantizableMobileNetV2",
    "MobileNet_V2_QuantizedWeights",
    "mobilenet_v2",
]


class MobileNet_V2_QuantizedWeights(WeightsEnum):
    ImageNet1K_QNNPACK_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/mobilenet_v2_qnnpack_37f702c5.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            "size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "backend": "qnnpack",
            "quantization": "qat",
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#qat-mobilenetv2",
            "unquantized": MobileNet_V2_Weights.ImageNet1K_V1,
            "acc@1": 71.658,
            "acc@5": 90.150,
        },
    )
    default = ImageNet1K_QNNPACK_V1


@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: MobileNet_V2_QuantizedWeights.ImageNet1K_QNNPACK_V1
        if kwargs.get("quantize", False)
        else MobileNet_V2_Weights.ImageNet1K_V1,
    )
)
def mobilenet_v2(
    *,
    weights: Optional[Union[MobileNet_V2_QuantizedWeights, MobileNet_V2_Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableMobileNetV2:
    weights = (MobileNet_V2_QuantizedWeights if quantize else MobileNet_V2_Weights).verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        if "backend" in weights.meta:
            _ovewrite_named_param(kwargs, "backend", weights.meta["backend"])
    backend = kwargs.pop("backend", "qnnpack")

    model = QuantizableMobileNetV2(block=QuantizableInvertedResidual, **kwargs)
    _replace_relu(model)
    if quantize:
        quantize_model(model, backend)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
