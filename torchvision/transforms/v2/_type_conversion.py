from typing import Any, Optional, Union

import numpy as np
import PIL.Image
import torch

from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F, Transform

from torchvision.transforms.v2._utils import is_pure_tensor
from torchvision.transforms.v2.functional._cvcuda import _is_cvcuda_tensor


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
    """Convert CHW/NCHW Torch images to CV-CUDA, without scaling or permuting values.

    Requires CV-CUDA >= 0.19.0. CPU inputs move to CUDA; CUDA inputs preserve their
    device unless ``device`` is specified. Non-contiguous inputs are copied. This
    transform does not support TorchScript or autograd.

    Args:
        device (torch.device or str, optional): Target CUDA device.
    """

    _transformed_types = (is_pure_tensor, tv_tensors.Image)

    def __init__(self, *, device=None) -> None:
        super().__init__()
        self.device = device

    def transform(self, inpt: torch.Tensor, params: dict[str, Any]):
        return F.to_cvcuda_tensor(inpt, device=self.device)


class CVCUDAToTensor(Transform):
    """Convert CV-CUDA tensors to zero-copy CUDA Torch tensors.

    Preserves shape, axis order, dtype, values, and CUDA device without permuting channels.

    This transform does not support TorchScript.
    """

    _transformed_types = (_is_cvcuda_tensor,)

    def transform(self, inpt: Any, params: dict[str, Any]) -> torch.Tensor:
        return F.cvcuda_to_tensor(inpt)
