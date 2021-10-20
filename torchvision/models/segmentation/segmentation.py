import warnings

# Import all methods/classes for BC:
from . import *  # noqa: F401, F403


warnings.warn(
    "The 'torchvision.models.segmentation.segmentation' module is deprecated. Please use directly the parent module "
    "instead."
)
