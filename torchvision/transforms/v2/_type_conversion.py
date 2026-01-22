from typing import Any, Optional, TYPE_CHECKING, Union

import numpy as np
import PIL.Image
import torch

from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F, Transform
from torchvision.transforms.v2._utils import is_pure_tensor
from torchvision.transforms.v2.functional._utils import _import_cvcuda

if TYPE_CHECKING:
    import cvcuda  # type: ignore[import-not-found]


class PILToTensor(Transform):
    """Convert a PIL Image to a tensor of the same type - this does not scale values.

    This transform does not support torchscript.

    Convert a PIL Image with H height, W width, and C channels to a Tensor of shape (C x H x W).

    Example:
        >>> from PIL import Image
        >>> from torchvision.transforms import v2
        >>> img = Image.new("RGB", (320, 240))  # size (W=320, H=240)
        >>> tensor = v2.PILToTensor()(img)
        >>> print(tensor.shape)
        torch.Size([3, 240, 320])
    """

    _transformed_types = (PIL.Image.Image,)

    def transform(self, inpt: PIL.Image.Image, params: dict[str, Any]) -> torch.Tensor:
        return F.pil_to_tensor(inpt)


class ToImage(Transform):
    """Convert a tensor, ndarray, or PIL Image to :class:`~torchvision.tv_tensors.Image`
    ; this does not scale values.

    This transform does not support torchscript.
    """

    _transformed_types = (is_pure_tensor, PIL.Image.Image, np.ndarray)

    def transform(
        self, inpt: Union[torch.Tensor, PIL.Image.Image, np.ndarray], params: dict[str, Any]
    ) -> tv_tensors.Image:
        return F.to_image(inpt)


class ToPILImage(Transform):
    """Convert a tensor or an ndarray to PIL Image

    This transform does not support torchscript.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while adjusting the value range depending on the ``mode``.

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

    _transformed_types = (is_pure_tensor, tv_tensors.Image, np.ndarray)

    def __init__(self, mode: Optional[str] = None) -> None:
        super().__init__()
        self.mode = mode

    def transform(
        self, inpt: Union[torch.Tensor, PIL.Image.Image, np.ndarray], params: dict[str, Any]
    ) -> PIL.Image.Image:
        return F.to_pil_image(inpt, mode=self.mode)


class ToPureTensor(Transform):
    """Convert all TVTensors to pure tensors, removing associated metadata (if any).

    This doesn't scale or change the values, only the type.
    """

    _transformed_types = (tv_tensors.TVTensor,)

    def transform(self, inpt: Any, params: dict[str, Any]) -> torch.Tensor:
        return inpt.as_subclass(torch.Tensor)


class ToCVCUDATensor(Transform):
    """Convert a ``torch.Tensor`` with NCHW shape to a ``cvcuda.Tensor``.
    If the input tensor is on CPU, it will automatically be transferred to GPU.
    Only 1-channel and 3-channel images are supported.

    This transform does not support torchscript.
    """

    def transform(self, inpt: torch.Tensor, params: dict[str, Any]) -> "cvcuda.Tensor":
        return F.to_cvcuda_tensor(inpt)


class CVCUDAToTensor(Transform):
    """Convert a ``cvcuda.Tensor`` to a ``torch.Tensor`` with NCHW shape.

    This function does not support torchscript.
    """

    try:
        cvcuda = _import_cvcuda()
        _transformed_types = (cvcuda.Tensor,)
    except ImportError:
        pass

    def transform(self, inpt: Any, params: dict[str, Any]) -> torch.Tensor:
        return F.cvcuda_to_tensor(inpt)
