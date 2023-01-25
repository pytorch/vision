from .alexnet import *
from .convnext import *
from .densenet import *
from .efficientnet import *
from .googlenet import *
from .inception import *
from .mnasnet import *
from .mobilenet import *
from .regnet import *
from .resnet import *
from .shufflenetv2 import *
from .squeezenet import *
from .vgg import *
from .vision_transformer import *
from .swin_transformer import *
from .maxvit import *
from . import detection, optical_flow, quantization, segmentation, video
from ._api import get_model, get_model_builder, get_model_weights, get_weight, list_models
# We're making these public for packages like torchgeo who are interested in
# using them https://github.com/pytorch/vision/issues/7094
# TODO: we could / should document them publicly as well?
from ._api import Weights, WeightsEnum
