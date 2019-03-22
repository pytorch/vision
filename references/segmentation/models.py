from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
import torchvision


class ResNet(torchvision.models.resnet.ResNet):
    def __init__(self, block, layers, dilated=False, return_layers=None):
        assert len(layers) == 4
        super(ResNet, self).__init__(
                block,
                layers,
        )
        del self.avgpool
        del self.fc

        if not return_layers:
            return_layers = "layer4"
        if isinstance(return_layers, str):
            return_layers = {return_layers: "out"}
        assert set(return_layers).issubset([name for name, _ in self.named_children()])
        self.return_layers = return_layers

        if dilated:
            assert block is torchvision.models.resnet.Bottleneck
            self._add_dilation()

    def _add_dilation(self):
        d = (2, 2)
        for b in self.layer3[1:]:
            b.conv2.padding = d
            b.conv2.dilation = d
        self.layer4[0].conv2.padding = d
        self.layer4[0].conv2.dilation = d
        d = (4, 4)
        for b in self.layer4[1:]:
            b.conv2.padding = d
            b.conv2.dilation = d

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


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__()
        self.aspp = ASPP(in_channels, [12, 24, 36])
        self.block = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        x = self.aspp(x)
        return self.block(x)


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
    model_list = {
        '18': (torchvision.models.resnet.BasicBlock, [2, 2, 2, 2]),
        '50': (torchvision.models.resnet.Bottleneck, [3, 4, 6, 3]),
        '101': (torchvision.models.resnet.Bottleneck, [3, 4, 23, 3]),
    }
    matcher = re.compile('resnet(\d+)')
    match = matcher.match(name)
    if not match:
        raise ValueError("Invalid ResNet type {}".format(name))

    backbone_name = match.group(0)
    num_layers = match.group(1)

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    dilated = True if 'dilated' in name else False
    block, layers = model_list[num_layers]

    model = ResNet(block, layers, dilated=dilated, return_layers=return_layers)

    state_dict = model_zoo.load_url(torchvision.models.resnet.model_urls[backbone_name])
    model.load_state_dict(state_dict, strict=False)

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

