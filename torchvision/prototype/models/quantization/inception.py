import warnings
from functools import partial
from typing import Any, Optional, Union

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ....models.quantization.inception import (
    QuantizableInception3,
    _replace_relu,
    quantize_model,
)
from .._api import Weights, WeightEntry
from .._meta import _IMAGENET_CATEGORIES
from ..inception import InceptionV3Weights


__all__ = [
    "QuantizableInception3",
    "QuantizedInceptionV3Weights",
    "inception_v3",
]


class QuantizedInceptionV3Weights(Weights):
    ImageNet1K_FBGEMM_TFV1 = WeightEntry(
        url="https://download.pytorch.org/models/quantized/inception_v3_google_fbgemm-71447a44.pth",
        transforms=partial(ImageNetEval, crop_size=299, resize_size=342),
        meta={
            "size": (299, 299),
            "categories": _IMAGENET_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "backend": "fbgemm",
            "quantization": "ptq",
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#post-training-quantized-models",
            "unquantized": InceptionV3Weights.ImageNet1K_TFV1,
            "acc@1": 77.176,
            "acc@5": 93.354,
        },
    )


def inception_v3(
    weights: Optional[Union[QuantizedInceptionV3Weights, InceptionV3Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableInception3:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        if kwargs.pop("pretrained"):
            weights = (
                QuantizedInceptionV3Weights.ImageNet1K_FBGEMM_TFV1 if quantize else InceptionV3Weights.ImageNet1K_TFV1
            )
        else:
            weights = None

    if quantize:
        weights = QuantizedInceptionV3Weights.verify(weights)
    else:
        weights = InceptionV3Weights.verify(weights)

    original_aux_logits = kwargs.get("aux_logits", False)
    if weights is not None:
        if "transform_input" not in kwargs:
            kwargs["transform_input"] = True
        kwargs["aux_logits"] = True
        kwargs["num_classes"] = len(weights.meta["categories"])
        if "backend" in weights.meta:
            kwargs["backend"] = weights.meta["backend"]
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
