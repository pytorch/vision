from functools import partial
from typing import Any, Optional

from torchvision.prototype.transforms import VocEval
from torchvision.transforms.functional import InterpolationMode

from ....models.segmentation.fcn import FCN, _fcn_resnet
from .._api import WeightsEnum, Weights
from .._meta import _VOC_CATEGORIES
from .._utils import handle_legacy_interface, _ovewrite_value_param
from ..resnet import ResNet50_Weights, ResNet101_Weights, resnet50, resnet101


__all__ = ["FCN", "FCN_ResNet50_Weights", "FCN_ResNet101_Weights", "fcn_resnet50", "fcn_resnet101"]


_COMMON_META = {
    "categories": _VOC_CATEGORIES,
    "interpolation": InterpolationMode.BILINEAR,
}


class FCN_ResNet50_Weights(WeightsEnum):
    CocoWithVocLabels_V1 = Weights(
        url="https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth",
        transforms=partial(VocEval, resize_size=520),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#fcn_resnet50",
            "mIoU": 60.5,
            "acc": 91.4,
        },
    )
    default = CocoWithVocLabels_V1


class FCN_ResNet101_Weights(WeightsEnum):
    CocoWithVocLabels_V1 = Weights(
        url="https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth",
        transforms=partial(VocEval, resize_size=520),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_resnet101",
            "mIoU": 63.7,
            "acc": 91.9,
        },
    )
    default = CocoWithVocLabels_V1


@handle_legacy_interface(
    weights=("pretrained", FCN_ResNet50_Weights.CocoWithVocLabels_V1),
    weights_backbone=("pretrained_backbone", ResNet50_Weights.ImageNet1K_V1),
)
def fcn_resnet50(
    *,
    weights: Optional[FCN_ResNet50_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet50_Weights] = None,
    **kwargs: Any,
) -> FCN:
    weights = FCN_ResNet50_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param(aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = resnet50(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _fcn_resnet(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


@handle_legacy_interface(
    weights=("pretrained", FCN_ResNet101_Weights.CocoWithVocLabels_V1),
    weights_backbone=("pretrained_backbone", ResNet101_Weights.ImageNet1K_V1),
)
def fcn_resnet101(
    *,
    weights: Optional[FCN_ResNet101_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet101_Weights] = None,
    **kwargs: Any,
) -> FCN:
    weights = FCN_ResNet101_Weights.verify(weights)
    weights_backbone = ResNet101_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param(aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = resnet101(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _fcn_resnet(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
