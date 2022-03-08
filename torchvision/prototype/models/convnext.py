from functools import partial
from typing import Any, List, Optional

from torchvision.prototype.transforms import ImageClassificationEval
from torchvision.transforms.functional import InterpolationMode

from ...models.convnext import ConvNeXt, CNBlockConfig
from ._api import WeightsEnum, Weights
from ._meta import _IMAGENET_CATEGORIES
from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = [
    "ConvNeXt",
    "ConvNeXt_Tiny_Weights",
    "ConvNeXt_Small_Weights",
    "ConvNeXt_Base_Weights",
    "ConvNeXt_Large_Weights",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
]


def _convnext(
    block_setting: List[CNBlockConfig],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ConvNeXt:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "task": "image_classification",
    "architecture": "ConvNeXt",
    "publication_year": 2022,
    "size": (224, 224),
    "min_size": (32, 32),
    "categories": _IMAGENET_CATEGORIES,
    "interpolation": InterpolationMode.BILINEAR,
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#convnext",
}


class ConvNeXt_Tiny_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
        transforms=partial(ImageClassificationEval, crop_size=224, resize_size=236),
        meta={
            **_COMMON_META,
            "num_params": 28589128,
            "acc@1": 82.520,
            "acc@5": 96.146,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ConvNeXt_Small_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_small-0c510722.pth",
        transforms=partial(ImageClassificationEval, crop_size=224, resize_size=230),
        meta={
            **_COMMON_META,
            "num_params": 50223688,
            "acc@1": 83.616,
            "acc@5": 96.650,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ConvNeXt_Base_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_base-6075fbad.pth",
        transforms=partial(ImageClassificationEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 88591464,
            "acc@1": 84.062,
            "acc@5": 96.870,
        },
    )
    DEFAULT = IMAGENET1K_V1


class ConvNeXt_Large_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_large-ea097f82.pth",
        transforms=partial(ImageClassificationEval, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 197767336,
            "acc@1": 84.414,
            "acc@5": 96.976,
        },
    )
    DEFAULT = IMAGENET1K_V1


@handle_legacy_interface(weights=("pretrained", ConvNeXt_Tiny_Weights.IMAGENET1K_V1))
def convnext_tiny(*, weights: Optional[ConvNeXt_Tiny_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    weights = ConvNeXt_Tiny_Weights.verify(weights)

    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ConvNeXt_Small_Weights.IMAGENET1K_V1))
def convnext_small(
    *, weights: Optional[ConvNeXt_Small_Weights] = None, progress: bool = True, **kwargs: Any
) -> ConvNeXt:
    weights = ConvNeXt_Small_Weights.verify(weights)

    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ConvNeXt_Base_Weights.IMAGENET1K_V1))
def convnext_base(*, weights: Optional[ConvNeXt_Base_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    weights = ConvNeXt_Base_Weights.verify(weights)

    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ConvNeXt_Large_Weights.IMAGENET1K_V1))
def convnext_large(
    *, weights: Optional[ConvNeXt_Large_Weights] = None, progress: bool = True, **kwargs: Any
) -> ConvNeXt:
    weights = ConvNeXt_Large_Weights.verify(weights)

    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)
