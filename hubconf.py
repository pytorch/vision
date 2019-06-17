# Optional list of dependencies required by the package
dependencies = ['torch']

from torch.hub import load_state_dict_from_url
from torchvision.models.alexnet import alexnet
from torchvision.models.densenet import densenet121, densenet169, densenet201, densenet161
from torchvision.models.inception import inception_v3
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152,\
    resnext50_32x4d, resnext101_32x8d as resnext101_32x8d_base, ResNet
from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1
from torchvision.models.vgg import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from torchvision.models.segmentation import fcn_resnet101, deeplabv3_resnet101
from torchvision.models.googlenet import googlenet
from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0
from torchvision.models.mobilenet import mobilenet_v2


model_types = ['wsl-ig']


model_urls = {
    'resnext101_32x8d-wsl-ig': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
    'resnext101_32x16d-wsl-ig': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d-wsl-ig': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x48d-wsl-ig': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
}


def _resnext(arch, block, layers, pretrained, pretrained_model_type,
            progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        if pretrained_model_type not in model_types:
            raise ValueError("pretrained_model_type should be in "
                                    + str(model_types))
        pretrained_model_name = model_urls[arch] + '-' + pretrained_model_type
        state_dict = load_state_dict_from_url(model_urls[pretrained_model_name],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnext101_32x8d(pretrained=False, pretrained_model_type=None,
                    progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8 model.

    Args:
        pretrained (bool): If True, returns a pre-trained model based on
        pretrained_model_type.
        pretrained_model_type (str): If 'imagenet', returns a model pre-trained on
        ImageNet. If set to 'wsl-ig', returns a model pre-trained on
        weakly-supervised data and finetuned on ImageNet from Figure 5 in
        `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
        progress (bool): If True, displays a progress bar of the download to stderr.
        Defaults to 'imagenet' if no option is provided.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8

    # For backwards compatibility
    if not pretrained or pretrained_model_type == 'imagenet' \
            or pretrained_model_type == None:
        return resnext101_32x8d_base(pretrained, progress, **kwargs)

    return _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, pretrained_model_type, progress, **kwargs)


def resnext101_32x16d(pretrained=False, pretrained_model_type=None,
                    progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x16 model.

    Args:
        pretrained (bool): If True, returns a pre-trained model based on
        pretrained_model_type.
        pretrained_model_type (str): If set to 'wsl-ig' or 'None', returns a
        model pre-trained on weakly-supervised data and finetuned on ImageNet
        from Figure 5 in
        `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_ .
        Currently no other pre-trained model is supported.
        progress (bool): If True, displays a progress bar of the download to stderr.
        Defaults to 'imagenet' if no option is provided.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 16

    if pretrained and pretrained_model_type is None:
        pretrained_model_type = 'wsl-ig'
    if pretrained_model_type != 'wsl-ig':
        raise ValueError("Currently only supported pretrained_model_type is "
                                + "wsl-ig")

    return _resnext('resnext101_32x16d', Bottleneck, [3, 4, 23, 3],
                   pretrained, pretrained_model_type, progress, **kwargs)


def resnext101_32x32d(pretrained=False, pretrained_model_type=None,
                    progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x32 model.

    Args:
        pretrained (bool): If True, returns a pre-trained model based on
        pretrained_model_type.
        pretrained_model_type (str): If set to 'wsl-ig' or 'None', returns a
        model pre-trained on weakly-supervised data and finetuned on ImageNet
        from Figure 5 in
        `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_ .
        Currently no other pre-trained model is supported.
        progress (bool): If True, displays a progress bar of the download to stderr.
        Defaults to 'imagenet' if no option is provided.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 32

    if pretrained and pretrained_model_type is None:
        pretrained_model_type = 'wsl-ig'
    if pretrained_model_type != 'wsl-ig':
        raise ValueError("Currently only supported pretrained_model_type is "
                                + "wsl-ig")

    return _resnext('resnext101_32x32d', Bottleneck, [3, 4, 23, 3],
                   pretrained, pretrained_model_type, progress, **kwargs)


def resnext101_32x48d(pretrained=False, pretrained_model_type=None,
                    progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x48 model.

    Args:
        pretrained (bool): If True, returns a pre-trained model based on
        pretrained_model_type.
        pretrained_model_type (str): If set to 'wsl-ig' or 'None', returns a
        model pre-trained on weakly-supervised data and finetuned on ImageNet
        from Figure 5 in
        `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_ .
        Currently no other pre-trained model is supported.
        progress (bool): If True, displays a progress bar of the download to stderr.
        Defaults to 'imagenet' if no option is provided.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 48

    if pretrained and pretrained_model_type is None:
        pretrained_model_type = 'wsl-ig'
    if pretrained_model_type != 'wsl-ig':
        raise ValueError("Currently only supported pretrained_model_type is "
                                + "wsl-ig")

    return _resnext('resnext101_32x48d', Bottleneck, [3, 4, 23, 3],
                   pretrained, pretrained_model_type, progress, **kwargs)
