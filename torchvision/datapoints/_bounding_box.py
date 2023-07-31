from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torchvision.transforms import InterpolationMode  # TODO: this needs to be moved out of transforms

from ._datapoint import _FillTypeJIT, Datapoint


class BBoxFormat(Enum):
    """[BETA] Coordinate format of a bounding box.

    Available formats are

    * ``XYXY``
    * ``XYWH``
    * ``CXCYWH``
    """

    XYXY = "XYXY"
    XYWH = "XYWH"
    CXCYWH = "CXCYWH"


class BBoxes(Datapoint):
    """[BETA] :class:`torch.Tensor` subclass for bounding boxes.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        format (BBoxFormat, str): Format of the bounding box.
        spatial_size (two-tuple of ints): Height and width of the corresponding image or video.
        dtype (torch.dtype, optional): Desired data type of the bounding box. If omitted, will be inferred from
            ``data``.
        device (torch.device, optional): Desired device of the bounding box. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the bounding box is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations on the bounding box. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    """

    format: BBoxFormat
    spatial_size: Tuple[int, int]

    @classmethod
    def _wrap(cls, tensor: torch.Tensor, *, format: BBoxFormat, spatial_size: Tuple[int, int]) -> BBoxes:
        bounding_boxes = tensor.as_subclass(cls)
        bounding_boxes.format = format
        bounding_boxes.spatial_size = spatial_size
        return bounding_boxes

    def __new__(
        cls,
        data: Any,
        *,
        format: Union[BBoxFormat, str],
        spatial_size: Tuple[int, int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> BBoxes:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

        if isinstance(format, str):
            format = BBoxFormat[format.upper()]

        return cls._wrap(tensor, format=format, spatial_size=spatial_size)

    @classmethod
    def wrap_like(
        cls,
        other: BBoxes,
        tensor: torch.Tensor,
        *,
        format: Optional[BBoxFormat] = None,
        spatial_size: Optional[Tuple[int, int]] = None,
    ) -> BBoxes:
        """Wrap a :class:`torch.Tensor` as :class:`BBoxes` from a reference.

        Args:
            other (BBoxes): Reference bounding box.
            tensor (Tensor): Tensor to be wrapped as :class:`BBoxes`
            format (BBoxFormat, str, optional): Format of the bounding box.  If omitted, it is taken from the
                reference.
            spatial_size (two-tuple of ints, optional): Height and width of the corresponding image or video. If
                omitted, it is taken from the reference.

        """
        if isinstance(format, str):
            format = BBoxFormat[format.upper()]

        return cls._wrap(
            tensor,
            format=format if format is not None else other.format,
            spatial_size=spatial_size if spatial_size is not None else other.spatial_size,
        )

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr(format=self.format, spatial_size=self.spatial_size)

    def horizontal_flip(self) -> BBoxes:
        output = self._F.horizontal_flip_bounding_boxes(
            self.as_subclass(torch.Tensor), format=self.format, spatial_size=self.spatial_size
        )
        return BBoxes.wrap_like(self, output)

    def vertical_flip(self) -> BBoxes:
        output = self._F.vertical_flip_bounding_boxes(
            self.as_subclass(torch.Tensor), format=self.format, spatial_size=self.spatial_size
        )
        return BBoxes.wrap_like(self, output)

    def resize(  # type: ignore[override]
        self,
        size: List[int],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        max_size: Optional[int] = None,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> BBoxes:
        output, spatial_size = self._F.resize_bounding_boxes(
            self.as_subclass(torch.Tensor),
            spatial_size=self.spatial_size,
            size=size,
            max_size=max_size,
        )
        return BBoxes.wrap_like(self, output, spatial_size=spatial_size)

    def crop(self, top: int, left: int, height: int, width: int) -> BBoxes:
        output, spatial_size = self._F.crop_bounding_boxes(
            self.as_subclass(torch.Tensor), self.format, top=top, left=left, height=height, width=width
        )
        return BBoxes.wrap_like(self, output, spatial_size=spatial_size)

    def center_crop(self, output_size: List[int]) -> BBoxes:
        output, spatial_size = self._F.center_crop_bounding_boxes(
            self.as_subclass(torch.Tensor), format=self.format, spatial_size=self.spatial_size, output_size=output_size
        )
        return BBoxes.wrap_like(self, output, spatial_size=spatial_size)

    def resized_crop(
        self,
        top: int,
        left: int,
        height: int,
        width: int,
        size: List[int],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> BBoxes:
        output, spatial_size = self._F.resized_crop_bounding_boxes(
            self.as_subclass(torch.Tensor), self.format, top, left, height, width, size=size
        )
        return BBoxes.wrap_like(self, output, spatial_size=spatial_size)

    def pad(
        self,
        padding: Union[int, Sequence[int]],
        fill: Optional[Union[int, float, List[float]]] = None,
        padding_mode: str = "constant",
    ) -> BBoxes:
        output, spatial_size = self._F.pad_bounding_boxes(
            self.as_subclass(torch.Tensor),
            format=self.format,
            spatial_size=self.spatial_size,
            padding=padding,
            padding_mode=padding_mode,
        )
        return BBoxes.wrap_like(self, output, spatial_size=spatial_size)

    def rotate(
        self,
        angle: float,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        expand: bool = False,
        center: Optional[List[float]] = None,
        fill: _FillTypeJIT = None,
    ) -> BBoxes:
        output, spatial_size = self._F.rotate_bounding_boxes(
            self.as_subclass(torch.Tensor),
            format=self.format,
            spatial_size=self.spatial_size,
            angle=angle,
            expand=expand,
            center=center,
        )
        return BBoxes.wrap_like(self, output, spatial_size=spatial_size)

    def affine(
        self,
        angle: Union[int, float],
        translate: List[float],
        scale: float,
        shear: List[float],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: _FillTypeJIT = None,
        center: Optional[List[float]] = None,
    ) -> BBoxes:
        output = self._F.affine_bounding_boxes(
            self.as_subclass(torch.Tensor),
            self.format,
            self.spatial_size,
            angle,
            translate=translate,
            scale=scale,
            shear=shear,
            center=center,
        )
        return BBoxes.wrap_like(self, output)

    def perspective(
        self,
        startpoints: Optional[List[List[int]]],
        endpoints: Optional[List[List[int]]],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        fill: _FillTypeJIT = None,
        coefficients: Optional[List[float]] = None,
    ) -> BBoxes:
        output = self._F.perspective_bounding_boxes(
            self.as_subclass(torch.Tensor),
            format=self.format,
            spatial_size=self.spatial_size,
            startpoints=startpoints,
            endpoints=endpoints,
            coefficients=coefficients,
        )
        return BBoxes.wrap_like(self, output)

    def elastic(
        self,
        displacement: torch.Tensor,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        fill: _FillTypeJIT = None,
    ) -> BBoxes:
        output = self._F.elastic_bounding_boxes(
            self.as_subclass(torch.Tensor), self.format, self.spatial_size, displacement=displacement
        )
        return BBoxes.wrap_like(self, output)
