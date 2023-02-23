from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import PIL.Image
import torch
from torchvision.transforms import InterpolationMode

from ._datapoint import _FillTypeJIT, Datapoint


class Mask(Datapoint):
    """[BETA] Randomly selects a rectangle region in the input image or video and erases its pixels.

    .. betastatus:: RandomErasing transform

    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

    Returns:
        Erased input.

    Example:
        >>> from torchvision.transforms import v2 as transforms
        >>>
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.PILToTensor(),
        >>>   transforms.ConvertImageDtype(torch.float),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasing(),
        >>> ])
    """

    @classmethod
    def _wrap(cls, tensor: torch.Tensor) -> Mask:
        return tensor.as_subclass(cls)

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> Mask:
        if isinstance(data, PIL.Image.Image):
            from torchvision.transforms.v2 import functional as F

            data = F.pil_to_tensor(data)

        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        return cls._wrap(tensor)

    @classmethod
    def wrap_like(
        cls,
        other: Mask,
        tensor: torch.Tensor,
    ) -> Mask:
        return cls._wrap(tensor)

    @property
    def spatial_size(self) -> Tuple[int, int]:
        return tuple(self.shape[-2:])  # type: ignore[return-value]

    def horizontal_flip(self) -> Mask:
        output = self._F.horizontal_flip_mask(self.as_subclass(torch.Tensor))
        return Mask.wrap_like(self, output)

    def vertical_flip(self) -> Mask:
        output = self._F.vertical_flip_mask(self.as_subclass(torch.Tensor))
        return Mask.wrap_like(self, output)

    def resize(  # type: ignore[override]
        self,
        size: List[int],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        max_size: Optional[int] = None,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> Mask:
        output = self._F.resize_mask(self.as_subclass(torch.Tensor), size, max_size=max_size)
        return Mask.wrap_like(self, output)

    def crop(self, top: int, left: int, height: int, width: int) -> Mask:
        output = self._F.crop_mask(self.as_subclass(torch.Tensor), top, left, height, width)
        return Mask.wrap_like(self, output)

    def center_crop(self, output_size: List[int]) -> Mask:
        output = self._F.center_crop_mask(self.as_subclass(torch.Tensor), output_size=output_size)
        return Mask.wrap_like(self, output)

    def resized_crop(
        self,
        top: int,
        left: int,
        height: int,
        width: int,
        size: List[int],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> Mask:
        output = self._F.resized_crop_mask(self.as_subclass(torch.Tensor), top, left, height, width, size=size)
        return Mask.wrap_like(self, output)

    def pad(
        self,
        padding: List[int],
        fill: Optional[Union[int, float, List[float]]] = None,
        padding_mode: str = "constant",
    ) -> Mask:
        output = self._F.pad_mask(self.as_subclass(torch.Tensor), padding, padding_mode=padding_mode, fill=fill)
        return Mask.wrap_like(self, output)

    def rotate(
        self,
        angle: float,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        expand: bool = False,
        center: Optional[List[float]] = None,
        fill: _FillTypeJIT = None,
    ) -> Mask:
        output = self._F.rotate_mask(self.as_subclass(torch.Tensor), angle, expand=expand, center=center, fill=fill)
        return Mask.wrap_like(self, output)

    def affine(
        self,
        angle: Union[int, float],
        translate: List[float],
        scale: float,
        shear: List[float],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: _FillTypeJIT = None,
        center: Optional[List[float]] = None,
    ) -> Mask:
        output = self._F.affine_mask(
            self.as_subclass(torch.Tensor),
            angle,
            translate=translate,
            scale=scale,
            shear=shear,
            fill=fill,
            center=center,
        )
        return Mask.wrap_like(self, output)

    def perspective(
        self,
        startpoints: Optional[List[List[int]]],
        endpoints: Optional[List[List[int]]],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: _FillTypeJIT = None,
        coefficients: Optional[List[float]] = None,
    ) -> Mask:
        output = self._F.perspective_mask(
            self.as_subclass(torch.Tensor), startpoints, endpoints, fill=fill, coefficients=coefficients
        )
        return Mask.wrap_like(self, output)

    def elastic(
        self,
        displacement: torch.Tensor,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: _FillTypeJIT = None,
    ) -> Mask:
        output = self._F.elastic_mask(self.as_subclass(torch.Tensor), displacement, fill=fill)
        return Mask.wrap_like(self, output)
