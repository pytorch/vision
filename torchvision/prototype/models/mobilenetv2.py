import warnings
from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.mobilenetv2 import MobileNetV2
from ._api import Weights, WeightEntry
from ._meta import _IMAGENET_CATEGORIES


__all__ = ["MobileNetV2", "MobileNetV2Weights", "mobilenet_v2"]


class MobileNetV2Weights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            "size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv2",
            "acc@1": 71.878,
            "acc@5": 90.286,
        },
    )


def mobilenet_v2(weights: Optional[MobileNetV2Weights] = None, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = MobileNetV2Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = MobileNetV2Weights.verify(weights)

    if weights is not None:
        kwargs["num_classes"] = len(weights.meta["categories"])

    model = MobileNetV2(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
