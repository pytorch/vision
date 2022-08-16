from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
from torchvision.transforms import InterpolationMode

from ._feature import _Feature


class SegmentationMask(_Feature):
    def horizontal_flip(self) -> SegmentationMask:
        from torchvision.prototype.transforms import functional as _F

        output = _F.horizontal_flip_segmentation_mask(self)
        return SegmentationMask.new_like(self, output)

    def vertical_flip(self) -> SegmentationMask:
        from torchvision.prototype.transforms import functional as _F

        output = _F.vertical_flip_segmentation_mask(self)
        return SegmentationMask.new_like(self, output)

    def resize(  # type: ignore[override]
        self,
        size: List[int],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        max_size: Optional[int] = None,
        antialias: bool = False,
    ) -> SegmentationMask:
        from torchvision.prototype.transforms import functional as _F

        output = _F.resize_segmentation_mask(self, size, max_size=max_size)
        return SegmentationMask.new_like(self, output)

    def crop(self, top: int, left: int, height: int, width: int) -> SegmentationMask:
        from torchvision.prototype.transforms import functional as _F

        output = _F.crop_segmentation_mask(self, top, left, height, width)
        return SegmentationMask.new_like(self, output)

    def center_crop(self, output_size: List[int]) -> SegmentationMask:
        from torchvision.prototype.transforms import functional as _F

        output = _F.center_crop_segmentation_mask(self, output_size=output_size)
        return SegmentationMask.new_like(self, output)

    def resized_crop(
        self,
        top: int,
        left: int,
        height: int,
        width: int,
        size: List[int],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        antialias: bool = False,
    ) -> SegmentationMask:
        from torchvision.prototype.transforms import functional as _F

        output = _F.resized_crop_segmentation_mask(self, top, left, height, width, size=size)
        return SegmentationMask.new_like(self, output)

    def pad(
        self,
        padding: Union[int, Sequence[int]],
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
        padding_mode: str = "constant",
    ) -> SegmentationMask:
        from torchvision.prototype.transforms import functional as _F

        # This cast does Sequence[int] -> List[int] and is required to make mypy happy
        if not isinstance(padding, int):
            padding = list(padding)

        output = _F.pad_segmentation_mask(self, padding, padding_mode=padding_mode)
        return SegmentationMask.new_like(self, output)

    def rotate(
        self,
        angle: float,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        expand: bool = False,
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
        center: Optional[List[float]] = None,
    ) -> SegmentationMask:
        from torchvision.prototype.transforms import functional as _F

        output = _F.rotate_segmentation_mask(self, angle, expand=expand, center=center)
        return SegmentationMask.new_like(self, output)

    def affine(
        self,
        angle: float,
        translate: List[float],
        scale: float,
        shear: List[float],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
        center: Optional[List[float]] = None,
    ) -> SegmentationMask:
        from torchvision.prototype.transforms import functional as _F

        output = _F.affine_segmentation_mask(
            self,
            angle,
            translate=translate,
            scale=scale,
            shear=shear,
            center=center,
        )
        return SegmentationMask.new_like(self, output)

    def perspective(
        self,
        perspective_coeffs: List[float],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
    ) -> SegmentationMask:
        from torchvision.prototype.transforms import functional as _F

        output = _F.perspective_segmentation_mask(self, perspective_coeffs)
        return SegmentationMask.new_like(self, output)

    def elastic(
        self,
        displacement: torch.Tensor,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
    ) -> SegmentationMask:
        from torchvision.prototype.transforms import functional as _F

        output = _F.elastic_segmentation_mask(self, displacement)
        return SegmentationMask.new_like(self, output, dtype=output.dtype)
