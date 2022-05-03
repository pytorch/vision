import warnings

# Import all methods/classes for BC:
from . import *  # noqa: F401, F403


warnings.warn(
    "The 'torchvision.models.segmentation.segmentation' module is deprecated since 0.12 and will be removed in "
    "0.14. Please use the 'torchvision.models.segmentation' directly instead."
)
