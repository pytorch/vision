import warnings

from torchvision.transforms._functional_tensor import *  # noqa

warnings.warn(
    "The torchvision.transforms.functional_tensor module is deprecated "
    "in 0.15 and will be **removed in 0.17**. Please don't rely on it. "
    "You probably just need to use APIs in "
    "torchvision.transforms.functional or in "
    "torchvision.transforms.v2.functional."
)
