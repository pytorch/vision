from functools import partial
from typing import Any, Optional, Union

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ....models.quantization.inception import (
    QuantizableInception3,
    _replace_relu,
    quantize_model,
)
from .._api import WeightsEnum, Weights
from .._meta import _IMAGENET_CATEGORIES
from .._utils import handle_legacy_interface, _ovewrite_named_param
from ..inception import Inception_V3_Weights


__all__ = [
    "QuantizableInception3",
    "Inception_V3_QuantizedWeights",
    "inception_v3",
]


class Inception_V3_QuantizedWeights(WeightsEnum):
    ImageNet1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/inception_v3_google_fbgemm-71447a44.pth",
        transforms=partial(ImageNetEval, crop_size=299, resize_size=342),
        meta={
            "size": (299, 299),
            "categories": _IMAGENET_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "backend": "fbgemm",
            "quantization": "ptq",
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#post-training-quantized-models",
            "unquantized": Inception_V3_Weights.ImageNet1K_V1,
            "acc@1": 77.176,
            "acc@5": 93.354,
        },
    )
    default = ImageNet1K_FBGEMM_V1


@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: Inception_V3_QuantizedWeights.ImageNet1K_FBGEMM_V1
        if kwargs.get("quantize", False)
        else Inception_V3_Weights.ImageNet1K_V1,
    )
)
def inception_v3(
    *,
    weights: Optional[Union[Inception_V3_QuantizedWeights, Inception_V3_Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableInception3:
    weights = (Inception_V3_QuantizedWeights if quantize else Inception_V3_Weights).verify(weights)

    original_aux_logits = kwargs.get("aux_logits", False)
    if weights is not None:
        if "transform_input" not in kwargs:
            _ovewrite_named_param(kwargs, "transform_input", True)
        _ovewrite_named_param(kwargs, "aux_logits", True)
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        if "backend" in weights.meta:
            _ovewrite_named_param(kwargs, "backend", weights.meta["backend"])
    backend = kwargs.pop("backend", "fbgemm")

    model = QuantizableInception3(**kwargs)
    _replace_relu(model)
    if quantize:
        quantize_model(model, backend)

    if weights is not None:
        if quantize and not original_aux_logits:
            model.aux_logits = False
            model.AuxLogits = None
        model.load_state_dict(weights.get_state_dict(progress=progress))
        if not quantize and not original_aux_logits:
            model.aux_logits = False
            model.AuxLogits = None

    return model
