from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.convnext import ConvNeXt, CNBlockConfig
from ._api import WeightsEnum, Weights
from ._meta import _IMAGENET_CATEGORIES
from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = ["ConvNeXt", "ConvNeXt_Tiny_Weights", "convnext_tiny"]


class ConvNeXt_Tiny_Weights(WeightsEnum):
    ImageNet1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_tiny-47b116bd.pth",
        transforms=partial(ImageNetEval, crop_size=236),
        meta={
            "task": "image_classification",
            "architecture": "ConvNeXt",
            "publication_year": 2022,
            "num_params": 28589128,
            "size": (224, 224),
            "min_size": (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#convnext",
            "acc@1": 82.520,
            "acc@5": 96.146,
        },
    )
    default = ImageNet1K_V1


@handle_legacy_interface(weights=("pretrained", None))
def convnext_tiny(*, weights: Optional[ConvNeXt_Tiny_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    weights = ConvNeXt_Tiny_Weights.verify(weights)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    model = ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
