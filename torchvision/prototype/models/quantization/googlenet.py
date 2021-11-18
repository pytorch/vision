import warnings
from functools import partial
from typing import Any, Optional, Union

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ....models.quantization.googlenet import (
    QuantizableGoogLeNet,
    _replace_relu,
    quantize_model,
)
from .._api import Weights, WeightEntry
from .._meta import _IMAGENET_CATEGORIES
from ..googlenet import GoogLeNetWeights


__all__ = [
    "QuantizableGoogLeNet",
    "QuantizedGoogLeNetWeights",
    "googlenet",
]


class QuantizedGoogLeNetWeights(Weights):
    ImageNet1K_FBGEMM_TFV1 = WeightEntry(
        url="https://download.pytorch.org/models/quantized/googlenet_fbgemm-c00238cf.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            "size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "backend": "fbgemm",
            "quantization": "ptq",
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#post-training-quantized-models",
            "unquantized": GoogLeNetWeights.ImageNet1K_TFV1,
            "acc@1": 69.826,
            "acc@5": 89.404,
        },
    )


def googlenet(
    weights: Optional[Union[QuantizedGoogLeNetWeights, GoogLeNetWeights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableGoogLeNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        if kwargs.pop("pretrained"):
            weights = QuantizedGoogLeNetWeights.ImageNet1K_FBGEMM_TFV1 if quantize else GoogLeNetWeights.ImageNet1K_TFV1
        else:
            weights = None

    if quantize:
        weights = QuantizedGoogLeNetWeights.verify(weights)
    else:
        weights = GoogLeNetWeights.verify(weights)

    original_aux_logits = kwargs.get("aux_logits", False)
    if weights is not None:
        if "transform_input" not in kwargs:
            kwargs["transform_input"] = True
        kwargs["aux_logits"] = True
        kwargs["init_weights"] = False
        kwargs["num_classes"] = len(weights.meta["categories"])
        if "backend" in weights.meta:
            kwargs["backend"] = weights.meta["backend"]
    backend = kwargs.pop("backend", "fbgemm")

    model = QuantizableGoogLeNet(**kwargs)
    _replace_relu(model)
    if quantize:
        quantize_model(model, backend)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
        if not original_aux_logits:
            model.aux_logits = False
            model.aux1 = None  # type: ignore[assignment]
            model.aux2 = None  # type: ignore[assignment]
        else:
            warnings.warn(
                "auxiliary heads in the pretrained googlenet model are NOT pretrained, so make sure to train them"
            )

    return model
