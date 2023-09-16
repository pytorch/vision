# Optional list of dependencies required by the package
dependencies = ["torch"]

from torchvision.models import get_model_weights, get_weight
from torchvision.models.alexnet import alexnet
from torchvision.models.convnext import convnext_base, convnext_large, convnext_small, convnext_tiny
from torchvision.models.densenet import densenet121, densenet161, densenet169, densenet201
from torchvision.models.efficientnet import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
    efficientnet_v2_l,
    efficientnet_v2_m,
    efficientnet_v2_s,
)
from torchvision.models.googlenet import googlenet
from torchvision.models.inception import inception_v3
from torchvision.models.maxvit import maxvit_t
from torchvision.models.mnasnet import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
from torchvision.models.mobilenetv2 import mobilenet_v2
from torchvision.models.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small
from torchvision.models.optical_flow import raft_large, raft_small
from torchvision.models.regnet import (
    regnet_x_16gf,
    regnet_x_1_6gf,
    regnet_x_32gf,
    regnet_x_3_2gf,
    regnet_x_400mf,
    regnet_x_800mf,
    regnet_x_8gf,
    regnet_y_128gf,
    regnet_y_16gf,
    regnet_y_1_6gf,
    regnet_y_32gf,
    regnet_y_3_2gf,
    regnet_y_400mf,
    regnet_y_800mf,
    regnet_y_8gf,
)
from torchvision.models.resnet import (
    resnet101,
    resnet152,
    resnet18,
    resnet34,
    resnet50,
    resnext101_32x8d,
    resnext101_64x4d,
    resnext50_32x4d,
    wide_resnet101_2,
    wide_resnet50_2,
)
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet101,
    deeplabv3_resnet50,
    fcn_resnet101,
    fcn_resnet50,
    lraspp_mobilenet_v3_large,
)
from torchvision.models.shufflenetv2 import (
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
    shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
)
from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1
from torchvision.models.swin_transformer import swin_b, swin_s, swin_t, swin_v2_b, swin_v2_s, swin_v2_t
from torchvision.models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from torchvision.models.video import (
    mc3_18,
    mvit_v1_b,
    mvit_v2_s,
    r2plus1d_18,
    r3d_18,
    s3d,
    swin3d_b,
    swin3d_s,
    swin3d_t,
)
from torchvision.models.vision_transformer import vit_b_16, vit_b_32, vit_h_14, vit_l_16, vit_l_32
