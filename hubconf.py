
# Optional list of dependencies required by the package
dependencies = ['torch']


def resnet18(pretrained=False, **kwargs):
    """
    Resnet18 model
    pretrained (bool): a recommended kwargs for all entrypoints
    kwargs are arguments for the function
    """
    from torchvision.models.resnet import resnet18 as _resnet18
    model = _resnet18(pretrained, **kwargs)
    return model


def resnet34(pretrained=False, **kwargs):
    """
    Resnet34 model
    pretrained (bool): a recommended kwargs for all entrypoints
    kwargs are arguments for the function
    """
    from torchvision.models.resnet import resnet34 as _resnet34
    model = _resnet34(pretrained, **kwargs)
    return model


def resnet50(pretrained=False, **kwargs):
    """
    Resnet50 model
    pretrained (bool): a recommended kwargs for all entrypoints
    kwargs are arguments for the function
    """
    from torchvision.models.resnet import resnet50 as _resnet50
    model = _resnet50(pretrained, **kwargs)
    return model


def resnet101(pretrained=False, **kwargs):
    """
    Resnet101 model
    pretrained (bool): a recommended kwargs for all entrypoints
    kwargs are arguments for the function
    """
    from torchvision.models.resnet import resnet101 as _resnet101
    model = _resnet101(pretrained, **kwargs)
    return model


def resnet152(pretrained=False, **kwargs):
    """
    Resnet152 model
    pretrained (bool): a recommended kwargs for all entrypoints
    kwargs are arguments for the function
    """
    from torchvision.models.resnet import resnet152 as _resnet152
    model = _resnet152(pretrained, **kwargs)
    return model


def alexnet(pretrained=False, **kwargs):
    from torchvision.models.alexnet import alexnet as _alexnet
    model = _alexnet(pretrained, **kwargs)
    return model


def densenet121(pretrained=False, **kwargs):
    from torchvision.models.alexnet import densenet121 as _densenet121
    model = _densenet121(pretrained, **kwargs)
    return model


def densenet161(pretrained=False, **kwargs):
    from torchvision.models.alexnet import densenet161 as _densenet161
    model = _densenet161(pretrained, **kwargs)
    return model


def densenet169(pretrained=False, **kwargs):
    from torchvision.models.alexnet import densenet169 as _densenet169
    model = _densenet169(pretrained, **kwargs)
    return model


def densenet201(pretrained=False, **kwargs):
    from torchvision.models.alexnet import densenet201 as _densenet201
    model = _densenet201(pretrained, **kwargs)
    return model


def googlenet(pretrained=False, **kwargs):
    # doc string
    from torchvision.models.googlenet import googlenet as _googlenet
    model = _googlenet(pretrained, **kwargs)
    return model


def inception_v3(pretrained=False, **kwargs):
    # doc string
    from torchvision.models.inception import inception_v3 as _inception_v3
    model = _inception_v3(pretrained, **kwargs)
    return model


def mobilenet_v2(pretrained=False, **kwargs):
    from torchvision.models.mobilenet import mobilenet_v2 as _mobilenet_v2
    model = _mobilenet_v2(pretrained, **kwargs)
    return model


def squeezenet1_0(pretrained=False, **kwargs):
    from torchvision.models.squeezenet import squeezenet1_0 as _squeezenet1_0
    model = _squeezenet1_0(pretrained, **kwargs)
    return model


def squeezenet1_1(pretrained=False, **kwargs):
    from torchvision.models.squeezenet import squeezenet1_1 as _squeezenet1_1
    model = _squeezenet1_1(pretrained, **kwargs)
    return model


def vgg11(pretrained=False, **kwargs):
    from torchvision.models.vgg import vgg11 as _vgg11
    model = _vgg11(pretrained, **kwargs)
    return model


def vgg11_bn(pretrained=False, **kwargs):
    from torchvision.models.vgg import vgg11_bn as _vgg11_bn
    model = _vgg11_bn(pretrained, **kwargs)
    return model


def vgg13(pretrained=False, **kwargs):
    from torchvision.models.vgg import vgg13 as _vgg13
    model = _vgg13(pretrained, **kwargs)
    return model


def vgg13_bn(pretrained=False, **kwargs):
    from torchvision.models.vgg import vgg13_bn as _vgg13_bn
    model = _vgg13_bn(pretrained, **kwargs)
    return model


def vgg16(pretrained=False, **kwargs):
    from torchvision.models.vgg import vgg16 as _vgg16
    model = _vgg16(pretrained, **kwargs)
    return model


def vgg16_bn(pretrained=False, **kwargs):
    from torchvision.models.vgg import vgg16_bn as _vgg16_bn
    model = _vgg16_bn(pretrained, **kwargs)
    return model


def vgg19(pretrained=False, **kwargs):
    from torchvision.models.vgg import vgg19 as _vgg19
    model = _vgg19(pretrained, **kwargs)
    return model


def vgg19_bn(pretrained=False, **kwargs):
    from torchvision.models.vgg import vgg19_bn as _vgg19_bn
    model = _vgg19_bn(pretrained, **kwargs)
    return model
