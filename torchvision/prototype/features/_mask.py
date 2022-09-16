from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
from torchvision.transforms import InterpolationMode

from ._feature import _Feature


class Mask(_Feature):
    def horizontal_flip(self) -> Mask:
        output = self._F.horizontal_flip_mask(self)
        return Mask.new_like(self, output)

    def vertical_flip(self) -> Mask:
        output = self._F.vertical_flip_mask(self)
        return Mask.new_like(self, output)

    def resize(  # type: ignore[override]
        self,
        size: List[int],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        max_size: Optional[int] = None,
        antialias: bool = False,
    ) -> Mask:
        output = self._F.resize_mask(self, size, max_size=max_size)
        return Mask.new_like(self, output)

    def crop(self, top: int, left: int, height: int, width: int) -> Mask:
        output = self._F.crop_mask(self, top, left, height, width)
        return Mask.new_like(self, output)

    def center_crop(self, output_size: List[int]) -> Mask:
        output = self._F.center_crop_mask(self, output_size=output_size)
        return Mask.new_like(self, output)

    def resized_crop(
        self,
        top: int,
        left: int,
        height: int,
        width: int,
        size: List[int],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        antialias: bool = False,
    ) -> Mask:
        output = self._F.resized_crop_mask(self, top, left, height, width, size=size)
        return Mask.new_like(self, output)

    def pad(
        self,
        padding: Union[int, Sequence[int]],
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
        padding_mode: str = "constant",
    ) -> Mask:
        # This cast does Sequence[int] -> List[int] and is required to make mypy happy
        if not isinstance(padding, int):
            padding = list(padding)

        if isinstance(fill, (int, float)) or fill is None:
            if fill is None:
                fill = 0
            output = self._F.pad_mask(self, padding, padding_mode=padding_mode, fill=fill)
        else:
            # Let's raise an error for vector fill on masks
            raise ValueError("Non-scalar fill value is not supported")

        return Mask.new_like(self, output)

    def rotate(
        self,
        angle: float,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        expand: bool = False,
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
        center: Optional[List[float]] = None,
    ) -> Mask:
        output = self._F.rotate_mask(self, angle, expand=expand, center=center)
        return Mask.new_like(self, output)

    def affine(
        self,
        angle: float,
        translate: List[float],
        scale: float,
        shear: List[float],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
        center: Optional[List[float]] = None,
    ) -> Mask:
        output = self._F.affine_mask(
            self,
            angle,
            translate=translate,
            scale=scale,
            shear=shear,
            center=center,
        )
        return Mask.new_like(self, output)

    def perspective(
        self,
        perspective_coeffs: List[float],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
    ) -> Mask:
        output = self._F.perspective_mask(self, perspective_coeffs)
        return Mask.new_like(self, output)

    def elastic(
        self,
        displacement: torch.Tensor,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
    ) -> Mask:
        output = self._F.elastic_mask(self, displacement)
        return Mask.new_like(self, output, dtype=output.dtype)
