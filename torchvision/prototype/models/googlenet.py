import warnings
from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.googlenet import GoogLeNet, GoogLeNetOutputs, _GoogLeNetOutputs
from ._api import Weights, WeightEntry
from ._meta import _IMAGENET_CATEGORIES


__all__ = ["GoogLeNet", "GoogLeNetOutputs", "_GoogLeNetOutputs", "GoogLeNetWeights", "googlenet"]


class GoogLeNetWeights(Weights):
    ImageNet1K_TFV1 = WeightEntry(
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


def googlenet(weights: Optional[GoogLeNetWeights] = None, progress: bool = True, **kwargs: Any) -> GoogLeNet:
    if "pretrained" in kwargs:
        warnings.warn("The argument pretrained is deprecated, please use weights instead.")
        weights = GoogLeNetWeights.ImageNet1K_TFV1 if kwargs.pop("pretrained") else None
    weights = GoogLeNetWeights.verify(weights)

    original_aux_logits = kwargs.get("aux_logits", False)
    if weights is not None:
        if "transform_input" not in kwargs:
            # Similar to the comment about num_classes, I think we should raise and error if the user manually specified transform_input=False
            kwargs["transform_input"] = True
        if original_aux_logits:
            # Nit: maybe move this as an `else` clause below
            warnings.warn(
                "auxiliary heads in the pretrained googlenet model are NOT pretrained, so make sure to train them"
            )
        kwargs["aux_logits"] = True
        kwargs["init_weights"] = False
        kwargs["num_classes"] = len(weights.meta["categories"])

    model = GoogLeNet(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.state_dict(progress=progress))
        # I understand this is present in the current code, just curious why this is needed?
        if not original_aux_logits:
            model.aux_logits = False
            model.aux1 = None  # type: ignore[assignment]
            model.aux2 = None  # type: ignore[assignment]

    return model
