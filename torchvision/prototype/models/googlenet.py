import warnings
from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.googlenet import GoogLeNet, GoogLeNetOutputs, _GoogLeNetOutputs
from ._api import WeightsEnum, Weights
from ._meta import _IMAGENET_CATEGORIES
from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = ["GoogLeNet", "GoogLeNetOutputs", "_GoogLeNetOutputs", "GoogLeNet_Weights", "googlenet"]


class GoogLeNet_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/googlenet-1378be20.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            "size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#googlenet",
            "acc@1": 69.778,
            "acc@5": 89.530,
        },
    )
    default = ImageNet1K_V1


@handle_legacy_interface(weights=("pretrained", GoogLeNet_Weights.ImageNet1K_V1))
def googlenet(*, weights: Optional[GoogLeNet_Weights] = None, progress: bool = True, **kwargs: Any) -> GoogLeNet:
    weights = GoogLeNet_Weights.verify(weights)

    original_aux_logits = kwargs.get("aux_logits", False)
    if weights is not None:
        if "transform_input" not in kwargs:
            _ovewrite_named_param(kwargs, "transform_input", True)
        _ovewrite_named_param(kwargs, "aux_logits", True)
        _ovewrite_named_param(kwargs, "init_weights", False)
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = GoogLeNet(**kwargs)

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
