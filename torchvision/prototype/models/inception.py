import warnings
from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import ImageNetEval
from torchvision.transforms.functional import InterpolationMode

from ...models.inception import Inception3, InceptionOutputs, _InceptionOutputs
from ._api import Weights, WeightEntry
from ._meta import _IMAGENET_CATEGORIES


__all__ = ["Inception3", "InceptionOutputs", "_InceptionOutputs", "InceptionV3Weights", "inception_v3"]


class InceptionV3Weights(Weights):
    ImageNet1K_TFV1 = WeightEntry(
        url="https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth",
        transforms=partial(ImageNetEval, crop_size=299, resize_size=342),
        meta={
            "size": (299, 299),
            "categories": _IMAGENET_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#inception-v3",
            "acc@1": 77.294,
            "acc@5": 93.450,
        },
    )


def inception_v3(weights: Optional[InceptionV3Weights] = None, progress: bool = True, **kwargs: Any) -> Inception3:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = InceptionV3Weights.ImageNet1K_TFV1 if kwargs.pop("pretrained") else None
    weights = InceptionV3Weights.verify(weights)

    original_aux_logits = kwargs.get("aux_logits", True)
    if weights is not None:
        if "transform_input" not in kwargs:
            kwargs["transform_input"] = True
        kwargs["aux_logits"] = True
        kwargs["init_weights"] = False
        kwargs["num_classes"] = len(weights.meta["categories"])

    model = Inception3(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
        if not original_aux_logits:
            model.aux_logits = False
            model.AuxLogits = None

    return model
