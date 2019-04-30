from ._utils import IntermediateLayerGetter
from . import resnet
from .deeplabv3 import FCN, FCNHead, DeepLabHead, DeepLabV3


def _segm_resnet(name, backbone_name, num_classes, aux):
    backbone = resnet.__dict__[backbone_name](
        pretrained=True,
        replace_stride_with_dilation=[False, True, True])

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)

    model_map = {
        'deeplab': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model


def fcn_resnet50(pretrained=False, num_classes=21, aux_loss=None):
    model = _segm_resnet("fcn", "resnet50", num_classes, aux_loss)
    if pretrained:
        pass
    return model


def fcn_resnet101(pretrained=False, num_classes=21, aux_loss=None):
    model = _segm_resnet("fcn", "resnet101", num_classes, aux_loss)
    if pretrained:
        pass
    return model


def deeplabv3_resnet50(pretrained=False, num_classes=21, aux_loss=None):
    model = _segm_resnet("deeplab", "resnet50", num_classes, aux_loss)
    if pretrained:
        pass
    return model


def deeplabv3_resnet101(pretrained=False, num_classes=21, aux_loss=None):
    model = _segm_resnet("deeplab", "resnet101", num_classes, aux_loss)
    if pretrained:
        pass
    return model
