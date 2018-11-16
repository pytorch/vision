import torch.nn.functional as F
from torch import nn
from torch.utils import model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv_block(in_chs, out_chs, ks=3, stride=1, preact=False, activation=None):
    """Standard convolution block involving batchnormalization and nonlinear activation.
    Supports preactivation as described in: https://arxiv.org/pdf/1603.05027.pdf

    Args:
        in_chs (int): input channels
        out_chs (int): output channels
        ks (int): conv2d kernel size
        stride (int): conv2d stride
        preact (bool): Whether or not to use preactivation
        activation (nn.Module): Optional nonlinearity replacement. (Default: ReLu(inplace=True) )
    """
    bn_dim = in_chs if preact else out_chs
    activation = nn.ReLU(inplace=True) if activation is None else activation

    bn_act = [nn.BatchNorm2d(bn_dim), activation]

    if preact:
        layers = bn_act + [nn.Conv2d(in_chs, out_chs, ks, padding=ks // 2, stride=stride, bias=False)]
    else:
        layers = [nn.Conv2d(in_chs, out_chs, ks, padding=ks // 2, stride=stride, bias=False)] + bn_act

    return nn.Sequential(*layers)


def standard_initial_block(in_chs, out_chs):
    """Standard initial block in a convolutional neural network.
    7x7 convolution followed by a max pooling layer.

    Args:
        in_chs (int): input channels
        out_chs (int): output channels"""
    layers = [conv_block(in_chs, out_chs, 7), nn.MaxPool2d(3, 2, padding=1)]
    return nn.Sequential(*layers)


class Identity(nn.Module):
    """Convenient block to keep around for NoOp layers"""
    def forward(self, x):
        return x


class ResidualBlock(nn.Module):
    """Abstract notion of a residual block.

    Should not be used without specifying behavior for layers."""

    def __init__(self, in_chs, out_chs, stride=1, preact=False):
        """
        Args:
            in_chs (int): input channels
            out_chs (int): output channels
            stride (int): stride to use within the conv block and adapter
            preact (bool): Whether or not to use preactivation
        """
        super(ResidualBlock, self).__init__()
        self.preact = preact

        self.layers = nn.Sequential(*[Identity()])

        if in_chs != out_chs or stride != 1:
            self.adapter = conv_block(in_chs, out_chs, ks=1, stride=stride, activation=Identity())
        else:
            self.adapter = Identity()

    def forward(self, x):
        if self.preact:
            return self.layers(x) + self.adapter(x)
        else:
            return F.relu(self.layers(x) + self.adapter(x), inplace=True)


class BasicBlock(ResidualBlock):
    """A residual block using standard convolutions as its intermediate processing layers."""

    def __init__(self, in_chs, out_chs, stride=1, preact=False):
        """
        Args:
            in_chs (int): input channels
            out_chs (int): output channels
            stride (int): stride dictating output dim for first conv_block
            preact (bool): whether or not to use preactivation
        """
        super(BasicBlock, self).__init__(in_chs, out_chs, stride=stride, preact=False)

        activation = nn.ReLU(inplace=True) if preact else Identity()
        self.layers = nn.Sequential(*[conv_block(in_chs, out_chs, stride=stride, preact=preact),
                                      conv_block(out_chs, out_chs, activation=activation, preact=preact)])


class Bottleneck(ResidualBlock):
    """A residual block where the intermediate layers compress the features
    by a factor of 4 before projecting into the right channels sizes"""

    def __init__(self, in_chs, out_chs, stride=1, preact=False):
        """
        Args:
            in_chs (int): input channels
            out_chs (int): output channels
            stride (int): stride dictating output dim for block (Applied to middle con_block
            preact (bool): whether or not to use preactivation
        """
        super(Bottleneck, self).__init__(in_chs, out_chs, stride=stride, preact=False)
        cmp_chs = out_chs // 4

        activation = nn.ReLU(inplace=True) if preact else Identity()

        self.layers = nn.Sequential(*[conv_block(in_chs, cmp_chs, ks=1, preact=preact),
                                      conv_block(cmp_chs, cmp_chs, ks=3, stride=stride, preact=preact),
                                      conv_block(cmp_chs, out_chs, ks=1, activation=activation, preact=preact)])


class ResNet(nn.Module):
    """A residual neural network whose behavior is specified by:
        -the block type,
        -the number of groups and blocks per group to use, the output classes,
        -and whether or not to use preactivation within the blocks."""

    def __init__(self, block, group_block_nums, num_classes=1000, preact=False):
        """
        Args:
            block (nn.Module): Type of block to use within netowork. (BasicBlock or Bottleneck)
            group_block_nums (list[int]): Num blocks to use for each group within body of resnet
            num_classes (int): Number of output classes for classification problem
            preact (bool): whether or not to use preactivation within blocks.
        """
        super(ResNet, self).__init__()
        filters = [64 * 2**i for i in range(len(group_block_nums))]

        self.std_init_block = standard_initial_block(3, 64)

        self.groups = [self._make_group(block, 64, filters[0], group_block_nums[0], preact=preact)]
        self.groups += [self._make_group(block,
                                         filters[group_ind],
                                         filters[group_ind + 1],
                                         group_block_num,
                                         stride=2,
                                         preact=preact)
                        for group_ind, group_block_num in enumerate(group_block_nums[1:])]

        self.groups = nn.Sequential(*self.groups)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(filters[-1], num_classes)

    def _make_group(self, block, in_chs, out_chs, num_blocks, stride=1, preact=False):
        blocks = [block(in_chs, out_chs, stride=stride, preact=preact)]
        blocks += [block(out_chs, out_chs, preact=preact) for i in range(1, num_blocks)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.std_init_block(x)
        x = self.groups(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def load_pretrained(model, zoo_key, strict=True):
    """Load model zoo weights into a model.

    Args:
        model (nn.Module): ResNet model
        zoo_key (str): Key for model zoo weights"""
    return model.load_state_dict(model_zoo.load_url(model_urls[zoo_key]), strict=strict)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model if pretrained is False else load_pretrained(model, 'resnet18')


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model if pretrained is False else load_pretrained(model, 'resnet34')


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model if pretrained is False else load_pretrained(model, 'resnet50')


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model if pretrained is False else load_pretrained(model, 'resnet18')


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model if pretrained is False else load_pretrained(model, 'resnet18')
