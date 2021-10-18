from typing import Any

from torch import nn

from .. import resnet
from ..feature_extraction import create_feature_extractor
from ._utils import _SimpleSegmentationModel, _load_weights


__all__ = ["FCN", "fcn_resnet50", "fcn_resnet101"]


model_urls = {
    "fcn_resnet50_coco": "https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth",
    "fcn_resnet101_coco": "https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth",
}


class FCN(_SimpleSegmentationModel):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.

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

        super(FCNHead, self).__init__(*layers)


def _fcn_resnet(
    backbone_name: str,
    pretrained: bool,
    progress: bool,
    num_classes: int,
    aux: bool,
    pretrained_backbone: bool = True,
) -> FCN:
    if pretrained:
        aux = True
        pretrained_backbone = False

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True]
    )
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = create_feature_extractor(backbone, return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux else None
    classifier = FCNHead(2048, num_classes)
    model = FCN(backbone, classifier, aux_classifier)

    if pretrained:
        arch = "fcn_" + backbone_name + "_coco"
        _load_weights(arch, model, model_urls.get(arch, None), progress)
    return model


def fcn_resnet50(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: bool = False,
    **kwargs: Any,
) -> FCN:
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return _fcn_resnet("resnet50", pretrained, progress, num_classes, aux_loss, **kwargs)


def fcn_resnet101(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: bool = False,
    **kwargs: Any,
) -> FCN:
    """Constructs a Fully-Convolutional Network model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return _fcn_resnet("resnet101", pretrained, progress, num_classes, aux_loss, **kwargs)
