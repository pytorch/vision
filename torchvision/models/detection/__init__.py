from .faster_rcnn import *
from .fcos import *
from .keypoint_rcnn import *
from .mask_rcnn import *
from .retinanet import *
from .ssd import *
from .ssdlite import *

import warnings

from .. import _BETA_DETECTION_IS_ENABLED

if not _BETA_DETECTION_IS_ENABLED:
    warnings.warn(
        "The torchvision.models.detection module is still in Beta stage, which means "
        "that backward compatibility isn't fully guaranteed. "
        "Please visit <SOME_URL> to learn more about what we are planning to change in future versions. "
        "To silence this warning, please call torchvision.models.enable_beta_detection."
    )
