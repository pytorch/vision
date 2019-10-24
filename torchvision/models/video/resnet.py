import warnings

import torch
import torch.nn as nn

from ..utils import load_state_dict_from_url


__all__ = ['r3d_18', 'mc3_18', 'r2plus1d_18']

model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}


class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 stride=1,
                 padding=1):

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 stride=1,
                 padding=1):

        self.midplanes = (in_planes * out_planes * 3 * 3 * 3) // (
                in_planes * 3 * 3 + 3 * out_planes)
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, self.midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv3DNoTemporal(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 stride=1,
                 padding=1):

        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):

        super(Bottleneck, self).__init__()

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """
    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class VideoResNet(nn.Module):

    # Version 2 adds updated BN params, and
    # solves midplane computation
    _version = 2

    def __init__(self, block, conv_makers, layers,
                 stem, num_classes=400,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # following are v2 updates for maximum reproducibility
                m.eps = 1e-3
                m.momentum = 0.9
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get("version", None)
        assert version in [1, 2]

        # the new changes only apply to the R2+1D models
        if version == 1 and isinstance(self.layer2[0].conv2[0], Conv2Plus1D):
            # V1 of the models had midplanes hard coded into the blocks 
            # and default BN parameters as in Pytorch.
            # All other layer configurations were the same.
            self.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 230)
            self.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 460)
            self.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 921)
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eps = 1e-5
                    m.momentum = 0.1
    
            # The model is now identical to v1, and must be saved as such.
            self._version = 1
            warnings.warn(
                "This is an updated vesrion of the R(2+1D) model that was "
                "updated following discussion in #1265. The performance "
                "deviations are minimal, but this might cause some BW compatibility "
                "issues, depending on the models.",
                UserWarning)

        super(VideoResNet, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)


def _video_resnet(arch, pretrained=False, progress=True, **kwargs):
    model = VideoResNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def r3d_18(pretrained=False, progress=True, **kwargs):
    """Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    return _video_resnet('r3d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] * 4,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)


def mc3_18(pretrained=False, progress=True, **kwargs):
    """Constructor for 18 layer Mixed Convolution network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: MC3 Network definition
    """
    return _video_resnet('mc3_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)


def r2plus1d_18(pretrained=False, progress=True, **kwargs):
    """Constructor for the 18 layer deep R(2+1)D network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R(2+1)D-18 network
    """
    return _video_resnet('r2plus1d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv2Plus1D] * 4,
                         layers=[2, 2, 2, 2],
                         stem=R2Plus1dStem, **kwargs)
