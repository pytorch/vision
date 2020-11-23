import torch
import os
import unittest
from torchvision import models, transforms
import sys

from PIL import Image
import torchvision.transforms.functional as F

try:
    from torchvision import _C_tests
except ImportError:
    _C_tests = None


def process_model(model, tensor, func, name):
    model.eval()
    traced_script_module = torch.jit.trace(model, tensor)
    traced_script_module.save("model.pt")

    py_output = model.forward(tensor)
    cpp_output = func("model.pt", tensor)

    assert torch.allclose(py_output, cpp_output), 'Output mismatch of ' + name + ' models'


def read_image1():
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'encode_jpeg',
                              'grace_hopper_517x606.jpg')
    image = Image.open(image_path)
    image = image.resize((224, 224))
    x = F.to_tensor(image)
    return x.view(1, 3, 224, 224)


def read_image2():
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'encode_jpeg',
                              'grace_hopper_517x606.jpg')
    image = Image.open(image_path)
    image = image.resize((299, 299))
    x = F.to_tensor(image)
    x = x.view(1, 3, 299, 299)
    return torch.cat([x, x], 0)


@unittest.skipIf(
    sys.platform == "darwin" or True,
    "C++ models are broken on OS X at the moment, and there's a BC breakage on master; "
    "see https://github.com/pytorch/vision/issues/1191")
class Tester(unittest.TestCase):
    pretrained = False
    image = read_image1()

    def test_alexnet(self):
        process_model(models.alexnet(self.pretrained), self.image, _C_tests.forward_alexnet, 'Alexnet')

    def test_vgg11(self):
        process_model(models.vgg11(self.pretrained), self.image, _C_tests.forward_vgg11, 'VGG11')

    def test_vgg13(self):
        process_model(models.vgg13(self.pretrained), self.image, _C_tests.forward_vgg13, 'VGG13')

    def test_vgg16(self):
        process_model(models.vgg16(self.pretrained), self.image, _C_tests.forward_vgg16, 'VGG16')

    def test_vgg19(self):
        process_model(models.vgg19(self.pretrained), self.image, _C_tests.forward_vgg19, 'VGG19')

    def test_vgg11_bn(self):
        process_model(models.vgg11_bn(self.pretrained), self.image, _C_tests.forward_vgg11bn, 'VGG11BN')

    def test_vgg13_bn(self):
        process_model(models.vgg13_bn(self.pretrained), self.image, _C_tests.forward_vgg13bn, 'VGG13BN')

    def test_vgg16_bn(self):
        process_model(models.vgg16_bn(self.pretrained), self.image, _C_tests.forward_vgg16bn, 'VGG16BN')

    def test_vgg19_bn(self):
        process_model(models.vgg19_bn(self.pretrained), self.image, _C_tests.forward_vgg19bn, 'VGG19BN')

    def test_resnet18(self):
        process_model(models.resnet18(self.pretrained), self.image, _C_tests.forward_resnet18, 'Resnet18')

    def test_resnet34(self):
        process_model(models.resnet34(self.pretrained), self.image, _C_tests.forward_resnet34, 'Resnet34')

    def test_resnet50(self):
        process_model(models.resnet50(self.pretrained), self.image, _C_tests.forward_resnet50, 'Resnet50')

    def test_resnet101(self):
        process_model(models.resnet101(self.pretrained), self.image, _C_tests.forward_resnet101, 'Resnet101')

    def test_resnet152(self):
        process_model(models.resnet152(self.pretrained), self.image, _C_tests.forward_resnet152, 'Resnet152')

    def test_resnext50_32x4d(self):
        process_model(models.resnext50_32x4d(), self.image, _C_tests.forward_resnext50_32x4d, 'ResNext50_32x4d')

    def test_resnext101_32x8d(self):
        process_model(models.resnext101_32x8d(), self.image, _C_tests.forward_resnext101_32x8d, 'ResNext101_32x8d')

    def test_wide_resnet50_2(self):
        process_model(models.wide_resnet50_2(), self.image, _C_tests.forward_wide_resnet50_2, 'WideResNet50_2')

    def test_wide_resnet101_2(self):
        process_model(models.wide_resnet101_2(), self.image, _C_tests.forward_wide_resnet101_2, 'WideResNet101_2')

    def test_squeezenet1_0(self):
        process_model(models.squeezenet1_0(self.pretrained), self.image,
                      _C_tests.forward_squeezenet1_0, 'Squeezenet1.0')

    def test_squeezenet1_1(self):
        process_model(models.squeezenet1_1(self.pretrained), self.image,
                      _C_tests.forward_squeezenet1_1, 'Squeezenet1.1')

    def test_densenet121(self):
        process_model(models.densenet121(self.pretrained), self.image, _C_tests.forward_densenet121, 'Densenet121')

    def test_densenet169(self):
        process_model(models.densenet169(self.pretrained), self.image, _C_tests.forward_densenet169, 'Densenet169')

    def test_densenet201(self):
        process_model(models.densenet201(self.pretrained), self.image, _C_tests.forward_densenet201, 'Densenet201')

    def test_densenet161(self):
        process_model(models.densenet161(self.pretrained), self.image, _C_tests.forward_densenet161, 'Densenet161')

    def test_mobilenet_v2(self):
        process_model(models.mobilenet_v2(self.pretrained), self.image, _C_tests.forward_mobilenetv2, 'MobileNet')

    def test_googlenet(self):
        process_model(models.googlenet(self.pretrained), self.image, _C_tests.forward_googlenet, 'GoogLeNet')

    def test_mnasnet0_5(self):
        process_model(models.mnasnet0_5(self.pretrained), self.image, _C_tests.forward_mnasnet0_5, 'MNASNet0_5')

    def test_mnasnet0_75(self):
        process_model(models.mnasnet0_75(self.pretrained), self.image, _C_tests.forward_mnasnet0_75, 'MNASNet0_75')

    def test_mnasnet1_0(self):
        process_model(models.mnasnet1_0(self.pretrained), self.image, _C_tests.forward_mnasnet1_0, 'MNASNet1_0')

    def test_mnasnet1_3(self):
        process_model(models.mnasnet1_3(self.pretrained), self.image, _C_tests.forward_mnasnet1_3, 'MNASNet1_3')

    def test_inception_v3(self):
        self.image = read_image2()
        process_model(models.inception_v3(self.pretrained), self.image, _C_tests.forward_inceptionv3, 'Inceptionv3')


if __name__ == '__main__':
    unittest.main()
