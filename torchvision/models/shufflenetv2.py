import functools

import torch
import torch.nn as nn
from .utils import load_state_dict_from_url

__all__ = ['ShuffleNetV2', 'shufflenetv2_x0_5', 'shufflenetv2_x1_0', 'shufflenetv2_x1_5', 'shufflenetv2_x2_0']

model_urls = {
    'shufflenetv2_x0.5':
        'https://github.com/barrh/Shufflenet-v2-Pytorch/releases/download/v0.1.0/shufflenetv2_x0.5-f707e7126e.pt',
    'shufflenetv2_x1.0':
        'https://github.com/barrh/Shufflenet-v2-Pytorch/releases/download/v0.1.0/shufflenetv2_x1-5666bf0f80.pt',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        pw_conv11 = functools.partial(nn.Conv2d, kernel_size=1, stride=1, padding=0, bias=False)
        dw_conv33 = functools.partial(self.depthwise_conv,
                                      kernel_size=3, stride=self.stride, padding=1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                dw_conv33(inp, inp),
                nn.BatchNorm2d(inp),
                pw_conv11(inp, branch_features),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            pw_conv11(inp if (self.stride > 1) else branch_features, branch_features),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            dw_conv33(branch_features, branch_features),
            nn.BatchNorm2d(branch_features),
            pw_conv11(branch_features, branch_features),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult='1.0'):
        super(ShuffleNetV2, self).__init__()

        self.stage_out_channels = self._getStages(width_mult)
        input_channels = 3
        output_channels = self.stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        stage_repeats = [4, 8, 4]
        for name, repeats, output_channels in zip(
                stage_names, stage_repeats, self.stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self.stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x

    @staticmethod
    def _getStages(mult):
        stages = {
            '0.5': [24, 48, 96, 192, 1024],
            '1.0': [24, 116, 232, 464, 1024],
            '1.5': [24, 176, 352, 704, 1024],
            '2.0': [24, 244, 488, 976, 2048],
        }
        return stages[mult]


def _shufflenetv2(pretrained, progress, width_mult, **kwargs):
    model = ShuffleNetV2(width_mult=width_mult, **kwargs)

    if pretrained:
        arch = 'shufflenetv2_x' + width_mult
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_urls, progress=progress)
            model.load_state_dict(state_dict)

    return model


def shufflenetv2_x0_5(pretrained=False, progress=True, **kwargs):
    return _shufflenetv2(pretrained, progress, '0.5', **kwargs)


def shufflenetv2_x1_0(pretrained=False, progress=True, **kwargs):
    return _shufflenetv2(pretrained, progress, '1.0', **kwargs)


def shufflenetv2_x1_5(pretrained=False, progress=True, **kwargs):
    return _shufflenetv2(pretrained, progress, '1.5', **kwargs)


def shufflenetv2_x2_0(pretrained=False, progress=True, **kwargs):
    return _shufflenetv2(pretrained, progress, '2.0', **kwargs)
