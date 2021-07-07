# Optional list of dependencies required by the package
dependencies = ['torch']

# classification
from torchvision.models.alexnet import alexnet
from torchvision.models.densenet import densenet121, densenet169, densenet201, densenet161
from torchvision.models.inception import inception_v3
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152,\
    resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1
from torchvision.models.vgg import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from torchvision.models.googlenet import googlenet
from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0
from torchvision.models.mobilenetv2 import mobilenet_v2
from torchvision.models.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small
from torchvision.models.mnasnet import mnasnet0_5, mnasnet0_75, mnasnet1_0, \
    mnasnet1_3

# segmentation
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, \
    deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large
