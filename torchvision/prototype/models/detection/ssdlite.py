import warnings
from functools import partial
from typing import Any, Callable, Optional

from torch import nn
from torchvision.prototype.transforms import CocoEval
from torchvision.transforms.functional import InterpolationMode

from ....models.detection.ssdlite import (
    _mobilenet_extractor,
    _normal_init,
    _validate_trainable_layers,
    DefaultBoxGenerator,
    det_utils,
    SSD,
    SSDLiteHead,
)
from .._api import Weights, WeightEntry
from .._meta import _COCO_CATEGORIES
from ..mobilenetv3 import MobileNetV3LargeWeights, mobilenet_v3_large


__all__ = [
    "SSDlite320MobileNetV3LargeFPNWeights",
    "ssdlite320_mobilenet_v3_large",
]


class SSDlite320MobileNetV3LargeFPNWeights(Weights):
    Coco_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth",
        transforms=CocoEval,
        meta={
            "size": (320, 320),
            "categories": _COCO_CATEGORIES,
            "interpolation": InterpolationMode.BILINEAR,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/detection#ssdlite320-mobilenetv3-large",
            "map": 21.3,
        },
    )


def ssdlite320_mobilenet_v3_large(
    weights: Optional[SSDlite320MobileNetV3LargeFPNWeights] = None,
    weights_backbone: Optional[MobileNetV3LargeWeights] = None,
    progress: bool = True,
    num_classes: int = 91,
    trainable_backbone_layers: Optional[int] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    **kwargs: Any,
) -> SSD:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = SSDlite320MobileNetV3LargeFPNWeights.Coco_RefV1 if kwargs.pop("pretrained") else None
    weights = SSDlite320MobileNetV3LargeFPNWeights.verify(weights)
    if "pretrained_backbone" in kwargs:
        warnings.warn("The parameter pretrained_backbone is deprecated, please use weights_backbone instead.")
        weights_backbone = MobileNetV3LargeWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained_backbone") else None
    weights_backbone = MobileNetV3LargeWeights.verify(weights_backbone)

    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the parameter.")

    if weights is not None:
        weights_backbone = None
        num_classes = len(weights.meta["categories"])

    trainable_backbone_layers = _validate_trainable_layers(
        weights is not None or weights_backbone is not None, trainable_backbone_layers, 6, 6
    )

    # Enable reduced tail if no pretrained backbone is selected. See Table 6 of MobileNetV3 paper.
    reduce_tail = weights_backbone is None

    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    backbone = mobilenet_v3_large(
        weights=weights_backbone, progress=progress, norm_layer=norm_layer, reduced_tail=reduce_tail, **kwargs
    )
    if weights_backbone is None:
        # Change the default initialization scheme if not pretrained
        _normal_init(backbone)
    backbone = _mobilenet_extractor(
        backbone,
        trainable_backbone_layers,
        norm_layer,
    )

    size = (320, 320)
    anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()
    assert len(out_channels) == len(anchor_generator.aspect_ratios)

    defaults = {
        "score_thresh": 0.001,
        "nms_thresh": 0.55,
        "detections_per_img": 300,
        "topk_candidates": 300,
        # Rescale the input in a way compatible to the backbone:
        # The following mean/std rescale the data from [0, 1] to [-1, -1]
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    kwargs: Any = {**defaults, **kwargs}
    model = SSD(
        backbone,
        anchor_generator,
        size,
        num_classes,
        head=SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer),
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
