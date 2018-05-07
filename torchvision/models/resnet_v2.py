from torch import nn
from torchvision.models.resnet import conv3x3
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNetV2', 'resnet18_v2', 'resnet34_v2', 'resnet50_v2', 'resnet101_v2',
           'resnet152_v2']


model_urls = {
    'resnet18_v2': '',
    'resnet34_v2': '',
    'resnet50_v2': '',
    'resnet101_v2': '',
    'resnet152_v2': '',
}


class BasicBlockV2(nn.Module):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"<https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18, 34 layers.

    Args:
        inplanes (int): number of input channels.
        planes (int): number of output channels.
        stride (int): stride size.
        downsample (Module) optional downsample module to downsample the input.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV2, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu2 = nn.ReLU(inplace=True)  # just to make better graph
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.relu1(out)
        residual = self.downsample(out) if self.downsample is not None else x
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        return out + residual


class BottleneckV2(nn.Module):
    r"""Bottleneck V2 from
    `"Identity Mappings in Deep Residual Networks"<https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 50, 101, 152 layers.

    Args:
        inplanes (int): number of input channels.
        planes (int): number of output channels.
        stride (int): stride size.
        downsample (Module) optional downsample module to downsample the input.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckV2, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):

        out = self.bn1(x)
        out = self.relu1(out)

        residual = self.downsample(out) if self.downsample is not None else x
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        return out + residual


class ResNetV2(nn.Module):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"<https://arxiv.org/abs/1603.05027>`_ paper.
    Args:
        block (Module) : class for the residual block. Options are BasicBlockV1, BottleneckV1.
        layers (list of int) : numbers of layers in each block
        num_classes (int) :, default 1000, number of classification classes.
    """
    def __init__(self, block, layers, num_classes=1000):
        super(ResNetV2, self).__init__()
        assert block in (BottleneckV2, BasicBlockV2), "Argument block should be BottleneckV2 or BasicBlockV2"
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn5 = nn.BatchNorm2d(self.inplanes)
        self.relu5 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                                   kernel_size=1, stride=stride, bias=False)

        layers = [block(self.inplanes, planes, stride, downsample), ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn5(x)
        x = self.relu5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18_v2(pretrained=False, **kwargs):
    """Constructs a ResNet-18 V2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetV2(BasicBlockV2, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34_v2(pretrained=False, **kwargs):
    """Constructs a ResNet-34 V2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetV2(BasicBlockV2, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50_v2(pretrained=False, **kwargs):
    """Constructs a ResNet-50 V2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetV2(BottleneckV2, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101_v2(pretrained=False, **kwargs):
    """Constructs a ResNet-101 V2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetV2(BottleneckV2, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152_v2(pretrained=False, **kwargs):
    """Constructs a ResNet-152 V2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetV2(BottleneckV2, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
