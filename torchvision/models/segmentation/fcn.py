from functools import partial
from typing import Any, Optional

from torch import nn

from ...transforms._presets import SemanticSegmentation
from .._api import WeightsEnum, Weights
from .._meta import _VOC_CATEGORIES
from .._utils import IntermediateLayerGetter, handle_legacy_interface, _ovewrite_value_param
from ..resnet import ResNet, ResNet50_Weights, ResNet101_Weights, resnet50, resnet101
from ._utils import _SimpleSegmentationModel


__all__ = ["FCN", "FCN_ResNet50_Weights", "FCN_ResNet101_Weights", "fcn_resnet50", "fcn_resnet101"]


class FCN(_SimpleSegmentationModel):
    """
    Implements FCN model from
    `"Fully Convolutional Networks for Semantic Segmentation"
    <https://arxiv.org/abs/1411.4038>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    pass


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)


_COMMON_META = {
    "categories": _VOC_CATEGORIES,
    "min_size": (1, 1),
    "_docs": """
        These weights were trained on a subset of COCO, using only the 20 categories that are present in the Pascal VOC
        dataset.
    """,
}


class FCN_ResNet50_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 35322218,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#fcn_resnet50",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 60.5,
                    "pixel_acc": 91.4,
                }
            },
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


class FCN_ResNet101_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 54314346,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_resnet101",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 63.7,
                    "pixel_acc": 91.9,
                }
            },
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


def _fcn_resnet(
    backbone: ResNet,
    num_classes: int,
    aux: Optional[bool],
) -> FCN:
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux else None
    classifier = FCNHead(2048, num_classes)
    return FCN(backbone, classifier, aux_classifier)


@handle_legacy_interface(
    weights=("pretrained", FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1),
    weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1),
)
def fcn_resnet50(
    *,
    weights: Optional[FCN_ResNet50_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    **kwargs: Any,
) -> FCN:
    """Fully-Convolutional Network model with a ResNet-50 backbone from the `Fully Convolutional
    Networks for Semantic Segmentation <https://arxiv.org/abs/1411.4038>`_ paper.

    .. betastatus:: segmentation module

    Args:
        weights (:class:`~torchvision.models.segmentation.FCN_ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.FCN_ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background).
        aux_loss (bool, optional): If True, it uses an auxiliary loss.
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The pretrained
            weights for the backbone.
        **kwargs: parameters passed to the ``torchvision.models.segmentation.fcn.FCN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.segmentation.FCN_ResNet50_Weights
        :members:
    """

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
    weights=("pretrained", FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1),
    weights_backbone=("pretrained_backbone", ResNet101_Weights.IMAGENET1K_V1),
)
def fcn_resnet101(
    *,
    weights: Optional[FCN_ResNet101_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet101_Weights] = ResNet101_Weights.IMAGENET1K_V1,
    **kwargs: Any,
) -> FCN:
    """Fully-Convolutional Network model with a ResNet-101 backbone from the `Fully Convolutional
    Networks for Semantic Segmentation <https://arxiv.org/abs/1411.4038>`_ paper.

    .. betastatus:: segmentation module

    Args:
        weights (:class:`~torchvision.models.segmentation.FCN_ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.FCN_ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background).
        aux_loss (bool, optional): If True, it uses an auxiliary loss.
        weights_backbone (:class:`~torchvision.models.ResNet101_Weights`, optional): The pretrained
            weights for the backbone.
        **kwargs: parameters passed to the ``torchvision.models.segmentation.fcn.FCN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.segmentation.FCN_ResNet101_Weights
        :members:
    """

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


# The dictionary below is internal implementation detail and will be removed in v0.15
from .._utils import _ModelURLs


model_urls = _ModelURLs(
    {
        "fcn_resnet50_coco": FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.url,
        "fcn_resnet101_coco": FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1.url,
    }
)
