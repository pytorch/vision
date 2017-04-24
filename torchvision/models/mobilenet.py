import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['MobileNet', 'mobilenet']


model_urls = {
    'mobilenet': None,
}


def nearby_int(n):
    return int(round(n))


class DepthwiseSeparableFusedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0):
        super(DepthwiseSeparableFusedConv2d, self).__init__()
        self.components = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      groups=in_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.components(x)


class MobileNet(nn.Module):
    def __init__(self, alpha=1.0, shallow=False, num_classes=1000):
        super(MobileNet, self).__init__()
        layers = [
            nn.Conv2d(3, nearby_int(alpha * 32),
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nearby_int(alpha * 32)),
            nn.ReLU(inplace=True),

            DepthwiseSeparableFusedConv2d(
                nearby_int(alpha * 32), nearby_int(alpha * 64),
                kernel_size=3, padding=1),
            DepthwiseSeparableFusedConv2d(
                nearby_int(alpha * 64), nearby_int(alpha * 128),
                kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableFusedConv2d(
                nearby_int(alpha * 128), nearby_int(alpha * 128),
                kernel_size=3, padding=1),
            DepthwiseSeparableFusedConv2d(
                nearby_int(alpha * 128), nearby_int(alpha * 256),
                kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableFusedConv2d(
                nearby_int(alpha * 256), nearby_int(alpha * 256),
                kernel_size=3, padding=1),
            DepthwiseSeparableFusedConv2d(
                nearby_int(alpha * 256), nearby_int(alpha * 512),
                kernel_size=3, stride=2, padding=1)
        ]
        if not shallow:
            # 5x 512->512 DW-separable convolutions
            layers += [
                DepthwiseSeparableFusedConv2d(
                    nearby_int(alpha * 512), nearby_int(alpha * 512),
                    kernel_size=3, padding=1),
                DepthwiseSeparableFusedConv2d(
                    nearby_int(alpha * 512), nearby_int(alpha * 512),
                    kernel_size=3, padding=1),
                DepthwiseSeparableFusedConv2d(
                    nearby_int(alpha * 512), nearby_int(alpha * 512),
                    kernel_size=3, padding=1),
                DepthwiseSeparableFusedConv2d(
                    nearby_int(alpha * 512), nearby_int(alpha * 512),
                    kernel_size=3, padding=1),
                DepthwiseSeparableFusedConv2d(
                    nearby_int(alpha * 512), nearby_int(alpha * 512),
                    kernel_size=3, padding=1),
            ]
        layers += [
            DepthwiseSeparableFusedConv2d(
                nearby_int(alpha * 512), nearby_int(alpha * 1024),
                kernel_size=3, stride=2, padding=1),
            # Paper specifies stride-2, but unchanged size.
            # Assume its a typo and use stride-1 convolution
            DepthwiseSeparableFusedConv2d(
                nearby_int(alpha * 1024), nearby_int(alpha * 1024),
                kernel_size=3, stride=1, padding=1)
        ]
        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(nearby_int(alpha * 1024), num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def mobilenet(pretrained=False):
    r"""MobileNet model architecture from the `"MobileNets:
    Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNet()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['mobilenet']))
    return model
