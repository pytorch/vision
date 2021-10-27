import warnings
from functools import partial
from typing import Any, Callable, Optional, Type, Dict

from torchvision.prototype.models.resnet import resnet50, resnet101

from ....models.segmentation.deeplabv3 import DeepLabV3, _deeplabv3_mobilenetv3, _deeplabv3_resnet
from ...transforms.presets import VocEval
from .._api import Weights, WeightEntry
from .._meta import _VOC_CATEGORIES
from ..mobilenetv3 import MobileNetV3LargeWeights, mobilenet_v3_large
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


class DeepLabV3ResNet50Weights(Weights):
    CocoWithVocLabels_RefV1 = WeightEntry(
        url="https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",
        transforms=partial(VocEval, resize_size=520),
        meta={
            "categories": _VOC_CATEGORIES,
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
            "categories": _VOC_CATEGORIES,
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
            "categories": _VOC_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_mobilenet_v3_large",
            "mIoU": 60.3,
            "acc": 91.2,
        },
    )


def _deeplabv3(
    weights_class: Type[Weights],
    model_builder: Callable,
    weights_backbone_class: Type[Weights],
    backbone_model_builder: Callable,
    progress: bool,
    num_classes: int,
    backbone_args: Dict[str, Any],
    weights: Optional[Weights] = None,
    weights_backbone: Optional[Weights] = None,
    aux_loss: Optional[bool] = None,
    **kwargs: Any,
) -> DeepLabV3:
    if "pretrained" in kwargs:
        warnings.warn("The argument pretrained is deprecated, please use weights instead.")
        weights = weights_class.CocoWithVocLabels_RefV1 if kwargs.pop("pretrained") else None

    weights = weights_class.verify(weights)
    if "pretrained_backbone" in kwargs:
        warnings.warn("The argument pretrained_backbone is deprecated, please use weights_backbone instead.")
        weights_backbone = weights_backbone_class.ImageNet1K_RefV1 if kwargs.pop("pretrained_backbone") else None
    weights_backbone = weights_backbone_class.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        aux_loss = True
        num_classes = len(weights.meta["categories"])

    backbone = backbone_model_builder(weights=weights_backbone, **backbone_args)
    model = model_builder(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.state_dict(progress=progress))

    return model


def deeplabv3_resnet50(
    weights: Optional[DeepLabV3ResNet50Weights] = None,
    weights_backbone: Optional[ResNet50Weights] = None,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    **kwargs: Any,
) -> DeepLabV3:
    return _deeplabv3(
        DeepLabV3ResNet50Weights,
        _deeplabv3_resnet,
        ResNet50Weights,
        resnet50,
        progress,
        num_classes,
        {"replace_stride_with_dilation": [False, True, True]},
        weights,
        weights_backbone,
        aux_loss,
        kwargs=kwargs,
    )


def deeplabv3_resnet101(
    weights: Optional[DeepLabV3ResNet50Weights] = None,
    weights_backbone: Optional[ResNet50Weights] = None,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    **kwargs: Any,
) -> DeepLabV3:
    return _deeplabv3(
        DeepLabV3ResNet101Weights,
        _deeplabv3_resnet,
        ResNet101Weights,
        resnet101,
        progress,
        num_classes,
        {"replace_stride_with_dilation": [False, True, True]},
        weights,
        weights_backbone,
        aux_loss,
        kwargs=kwargs,
    )


def deeplabv3_mobilenet_v3_large(
    weights: Optional[DeepLabV3MobileNetV3LargeWeights] = None,
    weights_backbone: Optional[MobileNetV3LargeWeights] = None,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    **kwargs: Any,
) -> DeepLabV3:
    return _deeplabv3(
        DeepLabV3MobileNetV3LargeWeights,
        _deeplabv3_mobilenetv3,
        MobileNetV3LargeWeights,
        mobilenet_v3_large,
        progress,
        num_classes,
        {"dilated": True},
        weights,
        aux_loss,
        weights_backbone,
        kwargs=kwargs,
    )
