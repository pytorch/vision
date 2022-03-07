from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import ImageClassificationEval
from torchvision.transforms.functional import InterpolationMode

from ...models.inception import Inception3, InceptionOutputs, _InceptionOutputs
from ._api import WeightsEnum, Weights
from ._meta import _IMAGENET_CATEGORIES
from ._utils import handle_legacy_interface, _ovewrite_named_param


__all__ = ["Inception3", "InceptionOutputs", "_InceptionOutputs", "Inception_V3_Weights", "inception_v3"]


class Inception_V3_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth",
        transforms=partial(ImageClassificationEval, crop_size=299, resize_size=342),
        meta={
            "task": "image_classification",
            "architecture": "InceptionV3",
            "publication_year": 2015,
            "num_params": 27161264,
            "size": (299, 299),
            "min_size": (75, 75),
            "categories": _IMAGENET_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#inception-v3",
            "acc@1": 77.294,
            "acc@5": 93.450,
        },
    )
    DEFAULT = IMAGENET1K_V1


@handle_legacy_interface(weights=("pretrained", Inception_V3_Weights.IMAGENET1K_V1))
def inception_v3(*, weights: Optional[Inception_V3_Weights] = None, progress: bool = True, **kwargs: Any) -> Inception3:
    weights = Inception_V3_Weights.verify(weights)

    original_aux_logits = kwargs.get("aux_logits", True)
    if weights is not None:
        if "transform_input" not in kwargs:
            _ovewrite_named_param(kwargs, "transform_input", True)
        _ovewrite_named_param(kwargs, "aux_logits", True)
        _ovewrite_named_param(kwargs, "init_weights", False)
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = Inception3(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
        if not original_aux_logits:
            model.aux_logits = False
            model.AuxLogits = None

    return model
