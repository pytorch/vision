from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Tuple, Union

import torch

from ._datapoint import Datapoint


class BoundingBoxFormat(Enum):
    """[BETA] Coordinate format of a bounding box.

    Available formats are

    * ``XYXY``
    * ``XYWH``
    * ``CXCYWH``
    """

    XYXY = "XYXY"
    XYWH = "XYWH"
    CXCYWH = "CXCYWH"


class BoundingBoxes(Datapoint):
    """[BETA] :class:`torch.Tensor` subclass for bounding boxes.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        format (BoundingBoxFormat, str): Format of the bounding box.
        canvas_size (two-tuple of ints): Height and width of the corresponding image or video.
        dtype (torch.dtype, optional): Desired data type of the bounding box. If omitted, will be inferred from
            ``data``.
        device (torch.device, optional): Desired device of the bounding box. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the bounding box is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations on the bounding box. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    """

    format: BoundingBoxFormat
    canvas_size: Tuple[int, int]

    @classmethod
    def _wrap(cls, tensor: torch.Tensor, *, format: BoundingBoxFormat, canvas_size: Tuple[int, int]) -> BoundingBoxes:
        bounding_boxes = tensor.as_subclass(cls)
        bounding_boxes.format = format
        bounding_boxes.canvas_size = canvas_size
        return bounding_boxes

    def __new__(
        cls,
        data: Any,
        *,
        format: Union[BoundingBoxFormat, str],
        canvas_size: Tuple[int, int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> BoundingBoxes:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

        if isinstance(format, str):
            format = BoundingBoxFormat[format.upper()]

        return cls._wrap(tensor, format=format, canvas_size=canvas_size)

    @classmethod
    def wrap_like(
        cls,
        other: BoundingBoxes,
        tensor: torch.Tensor,
        *,
        format: Optional[BoundingBoxFormat] = None,
        canvas_size: Optional[Tuple[int, int]] = None,
    ) -> BoundingBoxes:
        """Wrap a :class:`torch.Tensor` as :class:`BoundingBoxes` from a reference.

        Args:
            other (BoundingBoxes): Reference bounding box.
            tensor (Tensor): Tensor to be wrapped as :class:`BoundingBoxes`
            format (BoundingBoxFormat, str, optional): Format of the bounding box.  If omitted, it is taken from the
                reference.
            canvas_size (two-tuple of ints, optional): Height and width of the corresponding image or video. If
                omitted, it is taken from the reference.

        """
        if isinstance(format, str):
            format = BoundingBoxFormat[format.upper()]

        return cls._wrap(
            tensor,
            format=format if format is not None else other.format,
            canvas_size=canvas_size if canvas_size is not None else other.canvas_size,
        )

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr(format=self.format, canvas_size=self.canvas_size)
