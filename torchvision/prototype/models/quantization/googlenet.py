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
from .._api import WeightsEnum, Weights
from .._meta import _IMAGENET_CATEGORIES
from .._utils import handle_legacy_interface, _ovewrite_named_param
from ..googlenet import GoogLeNet_Weights


__all__ = [
    "QuantizableGoogLeNet",
    "GoogLeNet_QuantizedWeights",
    "googlenet",
]


class GoogLeNet_QuantizedWeights(WeightsEnum):
    ImageNet1K_FBGEMM_V1 = Weights(
        url="https://download.pytorch.org/models/quantized/googlenet_fbgemm-c00238cf.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            "size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "backend": "fbgemm",
            "quantization": "ptq",
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#post-training-quantized-models",
            "unquantized": GoogLeNet_Weights.ImageNet1K_V1,
            "acc@1": 69.826,
            "acc@5": 89.404,
        },
    )
    default = ImageNet1K_FBGEMM_V1


@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: GoogLeNet_QuantizedWeights.ImageNet1K_FBGEMM_V1
        if kwargs.get("quantize", False)
        else GoogLeNet_Weights.ImageNet1K_V1,
    )
)
def googlenet(
    *,
    weights: Optional[Union[GoogLeNet_QuantizedWeights, GoogLeNet_Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableGoogLeNet:
    weights = (GoogLeNet_QuantizedWeights if quantize else GoogLeNet_Weights).verify(weights)

    original_aux_logits = kwargs.get("aux_logits", False)
    if weights is not None:
        if "transform_input" not in kwargs:
            _ovewrite_named_param(kwargs, "transform_input", True)
        _ovewrite_named_param(kwargs, "aux_logits", True)
        _ovewrite_named_param(kwargs, "init_weights", False)
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        if "backend" in weights.meta:
            _ovewrite_named_param(kwargs, "backend", weights.meta["backend"])
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
