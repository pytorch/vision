from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torchvision.transforms import InterpolationMode  # TODO: this needs to be moved out of transforms

from ._datapoint import _FillTypeJIT, Datapoint


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

    def horizontal_flip(self) -> BoundingBoxes:
        output = self._F.horizontal_flip_bounding_boxes(
            self.as_subclass(torch.Tensor), format=self.format, canvas_size=self.canvas_size
        )
        return BoundingBoxes.wrap_like(self, output)

    def vertical_flip(self) -> BoundingBoxes:
        output = self._F.vertical_flip_bounding_boxes(
            self.as_subclass(torch.Tensor), format=self.format, canvas_size=self.canvas_size
        )
        return BoundingBoxes.wrap_like(self, output)

    def resize(  # type: ignore[override]
        self,
        size: List[int],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        max_size: Optional[int] = None,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> BoundingBoxes:
        output, canvas_size = self._F.resize_bounding_boxes(
            self.as_subclass(torch.Tensor),
            canvas_size=self.canvas_size,
            size=size,
            max_size=max_size,
        )
        return BoundingBoxes.wrap_like(self, output, canvas_size=canvas_size)

    def crop(self, top: int, left: int, height: int, width: int) -> BoundingBoxes:
        output, canvas_size = self._F.crop_bounding_boxes(
            self.as_subclass(torch.Tensor), self.format, top=top, left=left, height=height, width=width
        )
        return BoundingBoxes.wrap_like(self, output, canvas_size=canvas_size)

    def center_crop(self, output_size: List[int]) -> BoundingBoxes:
        output, canvas_size = self._F.center_crop_bounding_boxes(
            self.as_subclass(torch.Tensor), format=self.format, canvas_size=self.canvas_size, output_size=output_size
        )
        return BoundingBoxes.wrap_like(self, output, canvas_size=canvas_size)

    def resized_crop(
        self,
        top: int,
        left: int,
        height: int,
        width: int,
        size: List[int],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> BoundingBoxes:
        output, canvas_size = self._F.resized_crop_bounding_boxes(
            self.as_subclass(torch.Tensor), self.format, top, left, height, width, size=size
        )
        return BoundingBoxes.wrap_like(self, output, canvas_size=canvas_size)

    def pad(
        self,
        padding: Union[int, Sequence[int]],
        fill: Optional[Union[int, float, List[float]]] = None,
        padding_mode: str = "constant",
    ) -> BoundingBoxes:
        output, canvas_size = self._F.pad_bounding_boxes(
            self.as_subclass(torch.Tensor),
            format=self.format,
            canvas_size=self.canvas_size,
            padding=padding,
            padding_mode=padding_mode,
        )
        return BoundingBoxes.wrap_like(self, output, canvas_size=canvas_size)

    def rotate(
        self,
        angle: float,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        expand: bool = False,
        center: Optional[List[float]] = None,
        fill: _FillTypeJIT = None,
    ) -> BoundingBoxes:
        output, canvas_size = self._F.rotate_bounding_boxes(
            self.as_subclass(torch.Tensor),
            format=self.format,
            canvas_size=self.canvas_size,
            angle=angle,
            expand=expand,
            center=center,
        )
        return BoundingBoxes.wrap_like(self, output, canvas_size=canvas_size)

    def affine(
        self,
        angle: Union[int, float],
        translate: List[float],
        scale: float,
        shear: List[float],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: _FillTypeJIT = None,
        center: Optional[List[float]] = None,
    ) -> BoundingBoxes:
        output = self._F.affine_bounding_boxes(
            self.as_subclass(torch.Tensor),
            self.format,
            self.canvas_size,
            angle,
            translate=translate,
            scale=scale,
            shear=shear,
            center=center,
        )
        return BoundingBoxes.wrap_like(self, output)

    def perspective(
        self,
        startpoints: Optional[List[List[int]]],
        endpoints: Optional[List[List[int]]],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        fill: _FillTypeJIT = None,
        coefficients: Optional[List[float]] = None,
    ) -> BoundingBoxes:
        output = self._F.perspective_bounding_boxes(
            self.as_subclass(torch.Tensor),
            format=self.format,
            canvas_size=self.canvas_size,
            startpoints=startpoints,
            endpoints=endpoints,
            coefficients=coefficients,
        )
        return BoundingBoxes.wrap_like(self, output)

    def elastic(
        self,
        displacement: torch.Tensor,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        fill: _FillTypeJIT = None,
    ) -> BoundingBoxes:
        output = self._F.elastic_bounding_boxes(
            self.as_subclass(torch.Tensor), self.format, self.canvas_size, displacement=displacement
        )
        return BoundingBoxes.wrap_like(self, output)
