import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

__all__ = ['GoogLeNet', 'googlenet', 'googlenet_bn']

model_urls = {
    'googlenet': '',
    'googlenet_bn': ''
}


def googlenet(pretrained=False, **kwargs):
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
        model = GoogLeNet(**kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['googlenet']))
        return model

    return GoogLeNet(**kwargs)


def googlenet_bn(pretrained=False, **kwargs):
    r"""GoogLeNet (Inception v1) model architecture with batch normalization from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
        model = GoogLeNet(batch_norm=True, **kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['googlenet_bn']))
        return model

    return GoogLeNet(batch_norm=True, **kwargs)


class GoogLeNet(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, batch_norm=False, init_weights=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, batch_norm, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.lrn1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)
        self.conv2 = BasicConv2d(64, 64, batch_norm, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, batch_norm, kernel_size=3, padding=1)
        self.lrn2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32, batch_norm)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64, batch_norm)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64, batch_norm)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64, batch_norm)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64, batch_norm)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64, batch_norm)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128, batch_norm)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128, batch_norm)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128, batch_norm)
        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes, batch_norm)
            self.aux2 = InceptionAux(528, num_classes, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.2)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.lrn1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.lrn2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.training and self.aux_logits:
            return aux1, aux2, x
        return x


class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, batch_norm=False):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, batch_norm, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, batch_norm, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, batch_norm, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, batch_norm, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, batch_norm, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, batch_norm, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, batch_norm=False):
        super(InceptionAux, self).__init__()
        self.conv = BasicConv2d(in_channels, 128, batch_norm, kernel_size=1)

        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (4, 4))

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.7, training=self.training)
        x = self.fc2(x)

        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=False, **kwargs):
        super(BasicConv2d, self).__init__()
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        return F.relu(x, inplace=True)
