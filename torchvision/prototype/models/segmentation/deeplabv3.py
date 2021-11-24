import warnings
from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import VocEval
from torchvision.transforms.functional import InterpolationMode

from ....models.segmentation.deeplabv3 import DeepLabV3, _deeplabv3_mobilenetv3, _deeplabv3_resnet
from .._api import Weights, WeightEntry
from .._meta import _VOC_CATEGORIES
from ..mobilenetv3 import MobileNetV3LargeWeights, mobilenet_v3_large
from ..resnet import resnet50, resnet101
from ..resnet import ResNet50Weights, ResNet101Weights


__all__ = [
    "DeepLabV3",
    "DeepLabV3ResNet50Weights",
    "DeepLabV3ResNet101Weights",
    "DeepLabV3MobileNetV3LargeWeights",
    "deeplabv3_mobilenet_v3_large",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
]


_COMMON_META = {
    "categories": _VOC_CATEGORIES,
    "interpolation": InterpolationMode.BILINEAR,
}


class DeepLabV3ResNet50Weights(Weights):
    CocoWithVocLabels_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",
        transforms=partial(VocEval, resize_size=520),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_resnet50",
            "mIoU": 66.4,
            "acc": 92.4,
        },
    )


class DeepLabV3ResNet101Weights(Weights):
    CocoWithVocLabels_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth",
        transforms=partial(VocEval, resize_size=520),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#fcn_resnet101",
            "mIoU": 67.4,
            "acc": 92.4,
        },
    )


class DeepLabV3MobileNetV3LargeWeights(Weights):
    CocoWithVocLabels_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
        transforms=partial(VocEval, resize_size=520),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_mobilenet_v3_large",
            "mIoU": 60.3,
            "acc": 91.2,
        },
    )


def deeplabv3_resnet50(
    weights: Optional[DeepLabV3ResNet50Weights] = None,
    weights_backbone: Optional[ResNet50Weights] = None,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    **kwargs: Any,
) -> DeepLabV3:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = DeepLabV3ResNet50Weights.CocoWithVocLabels_RefV1 if kwargs.pop("pretrained") else None

    weights = DeepLabV3ResNet50Weights.verify(weights)
    if "pretrained_backbone" in kwargs:
        warnings.warn("The parameter pretrained_backbone is deprecated, please use weights_backbone instead.")
        weights_backbone = ResNet50Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained_backbone") else None
    weights_backbone = ResNet50Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        aux_loss = True
        num_classes = len(weights.meta["categories"])

    backbone = resnet50(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def deeplabv3_resnet101(
    weights: Optional[DeepLabV3ResNet101Weights] = None,
    weights_backbone: Optional[ResNet101Weights] = None,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    **kwargs: Any,
) -> DeepLabV3:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = DeepLabV3ResNet101Weights.CocoWithVocLabels_RefV1 if kwargs.pop("pretrained") else None

    weights = DeepLabV3ResNet101Weights.verify(weights)
    if "pretrained_backbone" in kwargs:
        warnings.warn("The parameter pretrained_backbone is deprecated, please use weights_backbone instead.")
        weights_backbone = ResNet101Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained_backbone") else None
    weights_backbone = ResNet101Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        aux_loss = True
        num_classes = len(weights.meta["categories"])

    backbone = resnet101(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def deeplabv3_mobilenet_v3_large(
    weights: Optional[DeepLabV3MobileNetV3LargeWeights] = None,
    weights_backbone: Optional[MobileNetV3LargeWeights] = None,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    **kwargs: Any,
) -> DeepLabV3:
    if "pretrained" in kwargs:
        warnings.warn("The parameter pretrained is deprecated, please use weights instead.")
        weights = DeepLabV3MobileNetV3LargeWeights.CocoWithVocLabels_RefV1 if kwargs.pop("pretrained") else None

    weights = DeepLabV3MobileNetV3LargeWeights.verify(weights)
    if "pretrained_backbone" in kwargs:
        warnings.warn("The parameter pretrained_backbone is deprecated, please use weights_backbone instead.")
        weights_backbone = MobileNetV3LargeWeights.ImageNet1K_RefV1 if kwargs.pop("pretrained_backbone") else None
    weights_backbone = MobileNetV3LargeWeights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        aux_loss = True
        num_classes = len(weights.meta["categories"])

    backbone = mobilenet_v3_large(weights=weights_backbone, dilated=True)
    model = _deeplabv3_mobilenetv3(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
