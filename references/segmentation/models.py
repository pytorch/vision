from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
import torchvision


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k:v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class SegmentationModel(nn.Module):
    def __init__(self, backbone, head):
        super(SegmentationModel, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        # contract: features is a dict of tensors
        # the names from the classifiers in self will be matched to those
        # in the features dict
        assert len(features) == len(self.head)
        assert len(set(features).difference(set(self.head))) == 0

        result = OrderedDict()
        for key, module in self.head.items():
            x = features[key]
            x = module(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result[key] = x
        return result


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, atrous_rate):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def get_resnet(name, aux=False):
    import re
    matcher = re.compile('resnet(\d+)')
    match = matcher.match(name)
    if not match:
        raise ValueError("Invalid ResNet type {}".format(name))

    backbone_name = match.group(0)

    d = None
    if 'dilated' in name:
        d = [False, True, True]

    model = getattr(torchvision.models, backbone_name)(pretrained=True,
                                                       replace_stride_with_dilation=d)

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    model = IntermediateLayerGetter(model, return_layers=return_layers)
    return model


def get_model(name, backbone, num_classes, aux=False):
    if 'resnet' in backbone:
        backbone = get_resnet(backbone, aux)

    inv_return_layers = {v: k for k, v in backbone.return_layers.items()}
    classifiers = nn.ModuleDict()
    if aux:
        layer_name = inv_return_layers['aux']
        inplanes = getattr(backbone, layer_name).outplanes
        classifiers['aux'] = FCNHead(inplanes, num_classes)

    model_map = {
        'deeplab': DeepLabHead,
        'fcn': FCNHead,
    }
    layer_name = inv_return_layers['out']
    inplanes = getattr(backbone, layer_name).outplanes
    classifiers['out'] = model_map[name](inplanes, num_classes)

    model = SegmentationModel(backbone, classifiers)
    return model

