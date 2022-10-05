from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torchvision._utils import StrEnum
from torchvision.transforms import InterpolationMode  # TODO: this needs to be moved out of transforms

from ._feature import _Feature, FillTypeJIT


class BoundingBoxFormat(StrEnum):
    XYXY = StrEnum.auto()
    XYWH = StrEnum.auto()
    CXCYWH = StrEnum.auto()


class BoundingBox(_Feature):
    format: BoundingBoxFormat
    image_size: Tuple[int, int]

    def __new__(
        cls,
        data: Any,
        *,
        format: Union[BoundingBoxFormat, str],
        image_size: Tuple[int, int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: bool = False,
    ) -> BoundingBox:
        bounding_box = super().__new__(cls, data, dtype=dtype, device=device, requires_grad=requires_grad)

        if isinstance(format, str):
            format = BoundingBoxFormat.from_str(format.upper())
        bounding_box.format = format

        bounding_box.image_size = image_size

        return bounding_box

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr(format=self.format, image_size=self.image_size)

    @classmethod
    def new_like(
        cls,
        other: BoundingBox,
        data: Any,
        *,
        format: Optional[Union[BoundingBoxFormat, str]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> BoundingBox:
        return super().new_like(
            other,
            data,
            format=format if format is not None else other.format,
            image_size=image_size if image_size is not None else other.image_size,
            **kwargs,
        )

    def to_format(self, format: Union[str, BoundingBoxFormat]) -> BoundingBox:
        if isinstance(format, str):
            format = BoundingBoxFormat.from_str(format.upper())

        return BoundingBox.new_like(
            self, self._F.convert_format_bounding_box(self, old_format=self.format, new_format=format), format=format
        )

    def horizontal_flip(self) -> BoundingBox:
        output = self._F.horizontal_flip_bounding_box(self, format=self.format, image_size=self.image_size)
        return BoundingBox.new_like(self, output)

    def vertical_flip(self) -> BoundingBox:
        output = self._F.vertical_flip_bounding_box(self, format=self.format, image_size=self.image_size)
        return BoundingBox.new_like(self, output)

    def resize(  # type: ignore[override]
        self,
        size: List[int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        max_size: Optional[int] = None,
        antialias: bool = False,
    ) -> BoundingBox:
        output, image_size = self._F.resize_bounding_box(self, image_size=self.image_size, size=size, max_size=max_size)
        return BoundingBox.new_like(self, output, image_size=image_size)

    def crop(self, top: int, left: int, height: int, width: int) -> BoundingBox:
        output, image_size = self._F.crop_bounding_box(
            self, self.format, top=top, left=left, height=height, width=width
        )
        return BoundingBox.new_like(self, output, image_size=image_size)

    def center_crop(self, output_size: List[int]) -> BoundingBox:
        output, image_size = self._F.center_crop_bounding_box(
            self, format=self.format, image_size=self.image_size, output_size=output_size
        )
        return BoundingBox.new_like(self, output, image_size=image_size)

    def resized_crop(
        self,
        top: int,
        left: int,
        height: int,
        width: int,
        size: List[int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = False,
    ) -> BoundingBox:
        output, image_size = self._F.resized_crop_bounding_box(self, self.format, top, left, height, width, size=size)
        return BoundingBox.new_like(self, output, image_size=image_size)

    def pad(
        self,
        padding: Union[int, Sequence[int]],
        fill: FillTypeJIT = None,
        padding_mode: str = "constant",
    ) -> BoundingBox:
        output, image_size = self._F.pad_bounding_box(
            self, format=self.format, image_size=self.image_size, padding=padding, padding_mode=padding_mode
        )
        return BoundingBox.new_like(self, output, image_size=image_size)

    def rotate(
        self,
        angle: float,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        expand: bool = False,
        fill: FillTypeJIT = None,
        center: Optional[List[float]] = None,
    ) -> BoundingBox:
        output, image_size = self._F.rotate_bounding_box(
            self, format=self.format, image_size=self.image_size, angle=angle, expand=expand, center=center
        )
        return BoundingBox.new_like(self, output, image_size=image_size)

    def affine(
        self,
        angle: Union[int, float],
        translate: List[float],
        scale: float,
        shear: List[float],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: FillTypeJIT = None,
        center: Optional[List[float]] = None,
    ) -> BoundingBox:
        output = self._F.affine_bounding_box(
            self,
            self.format,
            self.image_size,
            angle,
            translate=translate,
            scale=scale,
            shear=shear,
            center=center,
        )
        return BoundingBox.new_like(self, output, dtype=output.dtype)

    def perspective(
        self,
        perspective_coeffs: List[float],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: FillTypeJIT = None,
    ) -> BoundingBox:
        output = self._F.perspective_bounding_box(self, self.format, perspective_coeffs)
        return BoundingBox.new_like(self, output, dtype=output.dtype)

    def elastic(
        self,
        displacement: torch.Tensor,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: FillTypeJIT = None,
    ) -> BoundingBox:
        output = self._F.elastic_bounding_box(self, self.format, displacement)
        return BoundingBox.new_like(self, output, dtype=output.dtype)
