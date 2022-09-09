from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
from torchvision.transforms import InterpolationMode

from ._feature import _Feature


class SegmentationMask(_Feature):
    def horizontal_flip(self) -> SegmentationMask:
        output = self._F.horizontal_flip_segmentation_mask(self)
        return SegmentationMask.new_like(self, output)

    def vertical_flip(self) -> SegmentationMask:
        output = self._F.vertical_flip_segmentation_mask(self)
        return SegmentationMask.new_like(self, output)

    def resize(  # type: ignore[override]
        self,
        size: List[int],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        max_size: Optional[int] = None,
        antialias: bool = False,
    ) -> SegmentationMask:
        output = self._F.resize_segmentation_mask(self, size, max_size=max_size)
        return SegmentationMask.new_like(self, output)

    def crop(self, top: int, left: int, height: int, width: int) -> SegmentationMask:
        output = self._F.crop_segmentation_mask(self, top, left, height, width)
        return SegmentationMask.new_like(self, output)

    def center_crop(self, output_size: List[int]) -> SegmentationMask:
        output = self._F.center_crop_segmentation_mask(self, output_size=output_size)
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
        output = self._F.resized_crop_segmentation_mask(self, top, left, height, width, size=size)
        return SegmentationMask.new_like(self, output)

    def pad(
        self,
        padding: Union[int, Sequence[int]],
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
        padding_mode: str = "constant",
    ) -> SegmentationMask:
        # This cast does Sequence[int] -> List[int] and is required to make mypy happy
        if not isinstance(padding, int):
            padding = list(padding)

        output = self._F.pad_segmentation_mask(self, padding, padding_mode=padding_mode)
        return SegmentationMask.new_like(self, output)

    def rotate(
        self,
        angle: float,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        expand: bool = False,
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
        center: Optional[List[float]] = None,
    ) -> SegmentationMask:
        output = self._F.rotate_segmentation_mask(self, angle, expand=expand, center=center)
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
        output = self._F.affine_segmentation_mask(
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
        output = self._F.perspective_segmentation_mask(self, perspective_coeffs)
        return SegmentationMask.new_like(self, output)

    def elastic(
        self,
        displacement: torch.Tensor,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
    ) -> SegmentationMask:
        output = self._F.elastic_segmentation_mask(self, displacement)
        return SegmentationMask.new_like(self, output, dtype=output.dtype)
