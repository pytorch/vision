import warnings
from functools import partial
from typing import Any, List, Optional, Type, Union

from ...models.vgg import VGG, make_layers, cfgs
from ..transforms.presets import ImageNetEval
from ._api import Weights, WeightEntry
from ._meta import _IMAGENET_CATEGORIES


__all__ = ["VGG", "VGG11Weights", "vgg11"]

def _vgg(arch: str, cfg: str, batch_norm: bool, weights: Optional[Weights], progress: bool, **kwargs: Any) -> VGG:
    if weights is not None:
        kwargs["num_classes"] = len(weights.meta["categories"])
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.state_dict(progress=progress))
    return model


_common_meta = {
    "size": (224, 224),
    "categories": _IMAGENET_CATEGORIES,
}


class VGG11Weights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/vgg11-8a719046.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_common_meta,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "acc@1": 69.020,
            "acc@5": 88.628
        },
    )



def vgg11(weights: Optional[VGG11Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    if "pretrained" in kwargs:
        warnings.warn("The argument pretrained is deprecated, please use weights instead.")
        weights = VGG11Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = VGG11Weights.verify(weights)

    return _vgg("vgg11", "A", False, weights, progress, **kwargs)
