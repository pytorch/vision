from typing import Any, Dict, Optional, Union

import numpy as np
import PIL.Image
import torch

from torchvision import datapoints
from torchvision.transforms.v2 import functional as F, Transform

from torchvision.transforms.v2.utils import is_simple_tensor


class PILToTensor(Transform):
    """[BETA] Convert a PIL Image to a tensor of the same type - this does not scale values.

    .. v2betastatus:: PILToTensor transform

    This transform does not support torchscript.

    Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
    """

    _transformed_types = (PIL.Image.Image,)

    def _transform(self, inpt: PIL.Image.Image, params: Dict[str, Any]) -> torch.Tensor:
        return F.pil_to_tensor(inpt)


class ToImageTensor(Transform):
    """[BETA] Convert a tensor, ndarray, or PIL Image to :class:`~torchvision.datapoints.Image`
    ; this does not scale values.

    .. v2betastatus:: ToImageTensor transform

    This transform does not support torchscript.
    """

    _transformed_types = (is_simple_tensor, PIL.Image.Image, np.ndarray)

    def _transform(
        self, inpt: Union[torch.Tensor, PIL.Image.Image, np.ndarray], params: Dict[str, Any]
    ) -> datapoints.Image:
        return F.to_image_tensor(inpt)


class ToImagePIL(Transform):
    """[BETA] Convert a tensor or an ndarray to PIL Image - this does not scale values.

    .. v2betastatus:: ToImagePIL transform

    This transform does not support torchscript.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.

    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
            - If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
            - If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
            - If the input has 2 channels, the ``mode`` is assumed to be ``LA``.
            - If the input has 1 channel, the ``mode`` is determined by the data type (i.e ``int``, ``float``,
            ``short``).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    """

    _transformed_types = (is_simple_tensor, datapoints.Image, np.ndarray)

    def __init__(self, mode: Optional[str] = None) -> None:
        super().__init__()
        self.mode = mode

    def _transform(
        self, inpt: Union[torch.Tensor, PIL.Image.Image, np.ndarray], params: Dict[str, Any]
    ) -> PIL.Image.Image:
        return F.to_image_pil(inpt, mode=self.mode)


# We changed the name to align them with the new naming scheme. Still, `ToPILImage` is
# prevalent and well understood. Thus, we just alias it without deprecating the old name.
ToPILImage = ToImagePIL
