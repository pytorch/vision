import warnings
from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.alexnet import AlexNet
from ._api import Weights, WeightEntry
from ._meta import _IMAGENET_CATEGORIES


__all__ = ["AlexNet", "AlexNetWeights", "alexnet"]


class AlexNetWeights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            "size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "acc@1": 56.522,
            "acc@5": 79.066,
        },
    )


def alexnet(weights: Optional[AlexNetWeights] = None, progress: bool = True, **kwargs: Any) -> AlexNet:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = AlexNetWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = AlexNetWeights.verify(weights)
    if weights is not None:
        kwargs["num_classes"] = len(weights.meta["categories"])

    model = AlexNet(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
