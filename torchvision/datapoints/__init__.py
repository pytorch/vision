import torch
from torchvision import _BETA_TRANSFORMS_WARNING, _WARN_ABOUT_BETA_TRANSFORMS

from ._bounding_box import BoundingBoxes, BoundingBoxFormat
from ._datapoint import Datapoint
from ._image import Image
from ._mask import Mask
from ._torch_function_helpers import set_return_type
from ._video import Video

if _WARN_ABOUT_BETA_TRANSFORMS:
    import warnings

    warnings.warn(_BETA_TRANSFORMS_WARNING)
