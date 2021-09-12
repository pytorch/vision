from torch import nn
from typing import Any, Optional
from .._utils import IntermediateLayerGetter
from ..._internally_replaced_utils import load_state_dict_from_url
from .. import mobilenetv3
from .. import resnet
from .deeplabv3 import DeepLabHead, DeepLabV3
from .fcn import FCN, FCNHead
from .lraspp import LRASPP


__all__ = ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101',
           'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large']


model_urls = {
    'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
    'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
    'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
    'deeplabv3_mobilenet_v3_large_coco':
        'https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth',
    'lraspp_mobilenet_v3_large_coco': 'https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth',
}


def _segm_model(
    name: str,
    backbone_name: str,
    num_classes: int,
    aux: Optional[bool],
    pretrained_backbone: bool = True
) -> nn.Module:
    if 'resnet' in backbone_name:
        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=[False, True, True])
        out_layer = 'layer4'
        out_inplanes = 2048
        aux_layer = 'layer3'
        aux_inplanes = 1024
    elif 'mobilenet_v3' in backbone_name:
        backbone = mobilenetv3.__dict__[backbone_name](pretrained=pretrained_backbone, dilated=True).features

        # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
        # The first and last blocks are always included because they are the C0 (conv1) and Cn.
        stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
        out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
        out_layer = str(out_pos)
        out_inplanes = backbone[out_pos].out_channels
        aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
        aux_layer = str(aux_pos)
        aux_inplanes = backbone[aux_pos].out_channels
    else:
        raise NotImplementedError('backbone {} is not supported as of now'.format(backbone_name))

    return_layers = {out_layer: 'out'}
    if aux:
        return_layers[aux_layer] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    classifier = model_map[name][0](out_inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model


def _load_model(
    arch_type: str,
    backbone: str,
    pretrained: bool,
    progress: bool,
    num_classes: int,
    aux_loss: Optional[bool],
    **kwargs: Any
) -> nn.Module:
    if pretrained:
        aux_loss = True
        kwargs["pretrained_backbone"] = False
    model = _segm_model(arch_type, backbone, num_classes, aux_loss, **kwargs)
    if pretrained:
        _load_weights(model, arch_type, backbone, progress)
    return model


def _load_weights(model: nn.Module, arch_type: str, backbone: str, progress: bool) -> None:
    arch = arch_type + '_' + backbone + '_coco'
    model_url = model_urls.get(arch, None)
    if model_url is None:
        raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
    else:
        state_dict = load_state_dict_from_url(model_url, progress=progress)
        model.load_state_dict(state_dict)


def _segm_lraspp_mobilenetv3(backbone_name: str, num_classes: int, pretrained_backbone: bool = True) -> LRASPP:
    backbone = mobilenetv3.__dict__[backbone_name](pretrained=pretrained_backbone, dilated=True).features

    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    low_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    high_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    low_channels = backbone[low_pos].out_channels
    high_channels = backbone[high_pos].out_channels

    backbone = IntermediateLayerGetter(backbone, return_layers={str(low_pos): 'low', str(high_pos): 'high'})

    model = LRASPP(backbone, low_channels, high_channels, num_classes)
    return model


def fcn_resnet50(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    **kwargs: Any
) -> nn.Module:
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return _load_model('fcn', 'resnet50', pretrained, progress, num_classes, aux_loss, **kwargs)


def fcn_resnet101(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    **kwargs: Any
) -> nn.Module:
    """Constructs a Fully-Convolutional Network model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return _load_model('fcn', 'resnet101', pretrained, progress, num_classes, aux_loss, **kwargs)


def deeplabv3_resnet50(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    **kwargs: Any
) -> nn.Module:
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return _load_model('deeplabv3', 'resnet50', pretrained, progress, num_classes, aux_loss, **kwargs)


def deeplabv3_resnet101(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    **kwargs: Any
) -> nn.Module:
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): The number of classes
        aux_loss (bool): If True, include an auxiliary classifier
    """
    return _load_model('deeplabv3', 'resnet101', pretrained, progress, num_classes, aux_loss, **kwargs)


def deeplabv3_mobilenet_v3_large(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    **kwargs: Any
) -> nn.Module:
    """Constructs a DeepLabV3 model with a MobileNetV3-Large backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool): If True, it uses an auxiliary loss
    """
    return _load_model('deeplabv3', 'mobilenet_v3_large', pretrained, progress, num_classes, aux_loss, **kwargs)


def lraspp_mobilenet_v3_large(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 21,
    **kwargs: Any
) -> nn.Module:
    """Constructs a Lite R-ASPP Network model with a MobileNetV3-Large backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
    """
    if kwargs.pop("aux_loss", False):
        raise NotImplementedError('This model does not use auxiliary loss')

    backbone_name = 'mobilenet_v3_large'
    if pretrained:
        kwargs["pretrained_backbone"] = False
    model = _segm_lraspp_mobilenetv3(backbone_name, num_classes, **kwargs)

    if pretrained:
        _load_weights(model, 'lraspp', backbone_name, progress)

    return model
