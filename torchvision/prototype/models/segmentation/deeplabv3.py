from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import VocEval
from torchvision.transforms.functional import InterpolationMode

from ....models.segmentation.deeplabv3 import DeepLabV3, _deeplabv3_mobilenetv3, _deeplabv3_resnet
from .._api import WeightsEnum, Weights
from .._meta import _VOC_CATEGORIES
from .._utils import handle_legacy_interface, _ovewrite_value_param
from ..mobilenetv3 import MobileNet_V3_Large_Weights, mobilenet_v3_large
from ..resnet import resnet50, resnet101
from ..resnet import ResNet50_Weights, ResNet101_Weights


__all__ = [
    "DeepLabV3",
    "DeepLabV3_ResNet50_Weights",
    "DeepLabV3_ResNet101_Weights",
    "DeepLabV3_MobileNet_V3_Large_Weights",
    "deeplabv3_mobilenet_v3_large",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
]


_COMMON_META = {
    "categories": _VOC_CATEGORIES,
    "interpolation": InterpolationMode.BILINEAR,
}


class DeepLabV3_ResNet50_Weights(WeightsEnum):
    CocoWithVocLabels_V1 = Weights(
        url="https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",
        transforms=partial(VocEval, resize_size=520),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_resnet50",
            "mIoU": 66.4,
            "acc": 92.4,
        },
    )
    default = CocoWithVocLabels_V1


class DeepLabV3_ResNet101_Weights(WeightsEnum):
    CocoWithVocLabels_V1 = Weights(
        url="https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth",
        transforms=partial(VocEval, resize_size=520),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#fcn_resnet101",
            "mIoU": 67.4,
            "acc": 92.4,
        },
    )
    default = CocoWithVocLabels_V1


class DeepLabV3_MobileNet_V3_Large_Weights(WeightsEnum):
    CocoWithVocLabels_V1 = Weights(
        url="https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
        transforms=partial(VocEval, resize_size=520),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_mobilenet_v3_large",
            "mIoU": 60.3,
            "acc": 91.2,
        },
    )
    default = CocoWithVocLabels_V1


@handle_legacy_interface(
    weights=("pretrained", DeepLabV3_ResNet50_Weights.CocoWithVocLabels_V1),
    weights_backbone=("pretrained_backbone", ResNet50_Weights.ImageNet1K_V1),
)
def deeplabv3_resnet50(
    *,
    weights: Optional[DeepLabV3_ResNet50_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet50_Weights] = None,
    **kwargs: Any,
) -> DeepLabV3:
    weights = DeepLabV3_ResNet50_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param(aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = resnet50(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


@handle_legacy_interface(
    weights=("pretrained", DeepLabV3_ResNet101_Weights.CocoWithVocLabels_V1),
    weights_backbone=("pretrained_backbone", ResNet101_Weights.ImageNet1K_V1),
)
def deeplabv3_resnet101(
    *,
    weights: Optional[DeepLabV3_ResNet101_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet101_Weights] = None,
    **kwargs: Any,
) -> DeepLabV3:
    weights = DeepLabV3_ResNet101_Weights.verify(weights)
    weights_backbone = ResNet101_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param(aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = resnet101(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


@handle_legacy_interface(
    weights=("pretrained", DeepLabV3_MobileNet_V3_Large_Weights.CocoWithVocLabels_V1),
    weights_backbone=("pretrained_backbone", MobileNet_V3_Large_Weights.ImageNet1K_V1),
)
def deeplabv3_mobilenet_v3_large(
    *,
    weights: Optional[DeepLabV3_MobileNet_V3_Large_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[MobileNet_V3_Large_Weights] = None,
    **kwargs: Any,
) -> DeepLabV3:
    weights = DeepLabV3_MobileNet_V3_Large_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param(aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = mobilenet_v3_large(weights=weights_backbone, dilated=True)
    model = _deeplabv3_mobilenetv3(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
