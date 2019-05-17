import torch
from torchvision import models, transforms, _C_tests

from PIL import Image
import torchvision.transforms.functional as F


def test_model(model, tensor, func, name):
    traced_script_module = torch.jit.trace(model, tensor)
    traced_script_module.save("model.pt")

    py_output = model.forward(tensor)
    cpp_output = func("model.pt", tensor)

    assert torch.allclose(py_output, cpp_output), 'Output mismatch of ' + name + ' models'


image = Image.open('assets/grace_hopper_517x606.jpg')
image = image.resize((224, 224))
x = F.to_tensor(image)
x = x.view(1, 3, 224, 224)

pretrained = False

test_model(models.alexnet(pretrained), x, _C_tests.forward_alexnet, 'Alexnet')

test_model(models.vgg11(pretrained), x, _C_tests.forward_vgg11, 'VGG11')
test_model(models.vgg13(pretrained), x, _C_tests.forward_vgg13, 'VGG13')
test_model(models.vgg16(pretrained), x, _C_tests.forward_vgg16, 'VGG16')
test_model(models.vgg19(pretrained), x, _C_tests.forward_vgg19, 'VGG19')

test_model(models.vgg11_bn(pretrained), x, _C_tests.forward_vgg11bn, 'VGG11BN')
test_model(models.vgg13_bn(pretrained), x, _C_tests.forward_vgg13bn, 'VGG13BN')
test_model(models.vgg16_bn(pretrained), x, _C_tests.forward_vgg16bn, 'VGG16BN')
test_model(models.vgg19_bn(pretrained), x, _C_tests.forward_vgg19bn, 'VGG19BN')

test_model(models.resnet18(pretrained), x, _C_tests.forward_resnet18, 'Resnet18')
test_model(models.resnet34(pretrained), x, _C_tests.forward_resnet34, 'Resnet34')
test_model(models.resnet50(pretrained), x, _C_tests.forward_resnet50, 'Resnet50')
test_model(models.resnet101(pretrained), x, _C_tests.forward_resnet101, 'Resnet101')
test_model(models.resnet152(pretrained), x, _C_tests.forward_resnet152, 'Resnet152')
test_model(models.resnext50_32x4d(pretrained), x, _C_tests.forward_resnext50_32x4d, 'ResNext50_32x4d')
test_model(models.resnext101_32x8d(pretrained), x, _C_tests.forward_resnext101_32x8d, 'ResNext101_32x8d')

test_model(models.squeezenet1_0(pretrained), x, _C_tests.forward_squeezenet1_0, 'Squeezenet1.0')
test_model(models.squeezenet1_1(pretrained), x, _C_tests.forward_squeezenet1_1, 'Squeezenet1.1')

test_model(models.densenet121(pretrained), x, _C_tests.forward_densenet121, 'Densenet121')
test_model(models.densenet169(pretrained), x, _C_tests.forward_densenet169, 'Densenet169')
test_model(models.densenet201(pretrained), x, _C_tests.forward_densenet201, 'Densenet201')
test_model(models.densenet161(pretrained), x, _C_tests.forward_densenet161, 'Densenet161')

test_model(models.mobilenet_v2(pretrained), x, _C_tests.forward_mobilenetv2, 'MobileNet')

test_model(models.googlenet(pretrained), x, _C_tests.forward_googlenet, 'GoogLeNet')
test_model(models.inception_v3(pretrained), x, _C_tests.forward_inceptionv3, 'Inceptionv3')
