from __future__ import annotations

from typing import Any, Optional, Union

import PIL.Image
import torch

from ._datapoint import Datapoint


class Image(Datapoint):
    """[BETA] :class:`torch.Tensor` subclass for images.

    Args:
        data (tensor-like, PIL.Image.Image): Any data that can be turned into a tensor with :func:`torch.as_tensor` as
            well as PIL images.
        dtype (torch.dtype, optional): Desired data type of the bounding box. If omitted, will be inferred from
            ``data``.
        device (torch.device, optional): Desired device of the bounding box. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the bounding box is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations on the bounding box. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    """

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> Image:
        if isinstance(data, PIL.Image.Image):
            from torchvision.transforms.v2 import functional as F

            data = F.pil_to_tensor(data)

        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        if tensor.ndim < 2:
            raise ValueError
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        return cls._wrap(tensor)

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr()


_ImageType = Union[torch.Tensor, PIL.Image.Image, Image]
_ImageTypeJIT = torch.Tensor
_TensorImageType = Union[torch.Tensor, Image]
_TensorImageTypeJIT = torch.Tensor
