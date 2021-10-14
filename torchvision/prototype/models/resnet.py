import warnings
from functools import partial
from typing import Any, List, Optional, Type, Union

from ...models.resnet import BasicBlock, Bottleneck, ResNet
from ..transforms.presets import ImageNetEval
from ._api import Weights, WeightEntry
from ._meta import _IMAGENET_CATEGORIES


__all__ = ["ResNet", "ResNet50Weights", "resnet50"]


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[Weights],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        kwargs["num_classes"] = len(weights.meta["categories"])

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.state_dict(progress=progress))

    return model


_common_meta = {
    "size": (224, 224),
    "categories": _IMAGENET_CATEGORIES,
}


class ResNet50Weights(Weights):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_common_meta,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification",
            "acc@1": 76.130,
            "acc@5": 92.862,
        },
    )
    ImageNet1K_RefV2 = WeightEntry(
        url="https://download.pytorch.org/models/resnet50-tmp.pth",
        transforms=partial(ImageNetEval, crop_size=224),
        meta={
            **_common_meta,
            "recipe": "https://github.com/pytorch/vision/issues/3995",
            "acc@1": 80.352,
            "acc@5": 95.148,
        },
    )


def resnet50(weights: Optional[ResNet50Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    if "pretrained" in kwargs:
        warnings.warn("The argument pretrained is deprecated, please use weights instead.")
        weights = ResNet50Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None
    weights = ResNet50Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)
