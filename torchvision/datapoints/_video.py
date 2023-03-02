from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import torch
from torchvision.transforms.functional import InterpolationMode

from ._datapoint import _FillTypeJIT, Datapoint


class Video(Datapoint):
    """[BETA] :class:`torch.Tensor` subclass for videos.

    Args:
        data (tensor-like): Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        dtype (torch.dtype, optional): Desired data type of the bounding box. If omitted, will be inferred from
            ``data``.
        device (torch.device, optional): Desired device of the bounding box. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the bounding box is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations on the bounding box. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    """

    @classmethod
    def _wrap(cls, tensor: torch.Tensor) -> Video:
        video = tensor.as_subclass(cls)
        return video

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> Video:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        if data.ndim < 4:
            raise ValueError
        return cls._wrap(tensor)

    @classmethod
    def wrap_like(cls, other: Video, tensor: torch.Tensor) -> Video:
        return cls._wrap(tensor)

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr()

    @property
    def spatial_size(self) -> Tuple[int, int]:
        return tuple(self.shape[-2:])  # type: ignore[return-value]

    @property
    def num_channels(self) -> int:
        return self.shape[-3]

    @property
    def num_frames(self) -> int:
        return self.shape[-4]

    def horizontal_flip(self) -> Video:
        output = self._F.horizontal_flip_video(self.as_subclass(torch.Tensor))
        return Video.wrap_like(self, output)

    def vertical_flip(self) -> Video:
        output = self._F.vertical_flip_video(self.as_subclass(torch.Tensor))
        return Video.wrap_like(self, output)

    def resize(  # type: ignore[override]
        self,
        size: List[int],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        max_size: Optional[int] = None,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> Video:
        output = self._F.resize_video(
            self.as_subclass(torch.Tensor),
            size,
            interpolation=interpolation,
            max_size=max_size,
            antialias=antialias,
        )
        return Video.wrap_like(self, output)

    def crop(self, top: int, left: int, height: int, width: int) -> Video:
        output = self._F.crop_video(self.as_subclass(torch.Tensor), top, left, height, width)
        return Video.wrap_like(self, output)

    def center_crop(self, output_size: List[int]) -> Video:
        output = self._F.center_crop_video(self.as_subclass(torch.Tensor), output_size=output_size)
        return Video.wrap_like(self, output)

    def resized_crop(
        self,
        top: int,
        left: int,
        height: int,
        width: int,
        size: List[int],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> Video:
        output = self._F.resized_crop_video(
            self.as_subclass(torch.Tensor),
            top,
            left,
            height,
            width,
            size=list(size),
            interpolation=interpolation,
            antialias=antialias,
        )
        return Video.wrap_like(self, output)

    def pad(
        self,
        padding: List[int],
        fill: Optional[Union[int, float, List[float]]] = None,
        padding_mode: str = "constant",
    ) -> Video:
        output = self._F.pad_video(self.as_subclass(torch.Tensor), padding, fill=fill, padding_mode=padding_mode)
        return Video.wrap_like(self, output)

    def rotate(
        self,
        angle: float,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        expand: bool = False,
        center: Optional[List[float]] = None,
        fill: _FillTypeJIT = None,
    ) -> Video:
        output = self._F.rotate_video(
            self.as_subclass(torch.Tensor), angle, interpolation=interpolation, expand=expand, fill=fill, center=center
        )
        return Video.wrap_like(self, output)

    def affine(
        self,
        angle: Union[int, float],
        translate: List[float],
        scale: float,
        shear: List[float],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        fill: _FillTypeJIT = None,
        center: Optional[List[float]] = None,
    ) -> Video:
        output = self._F.affine_video(
            self.as_subclass(torch.Tensor),
            angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=interpolation,
            fill=fill,
            center=center,
        )
        return Video.wrap_like(self, output)

    def perspective(
        self,
        startpoints: Optional[List[List[int]]],
        endpoints: Optional[List[List[int]]],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        fill: _FillTypeJIT = None,
        coefficients: Optional[List[float]] = None,
    ) -> Video:
        output = self._F.perspective_video(
            self.as_subclass(torch.Tensor),
            startpoints,
            endpoints,
            interpolation=interpolation,
            fill=fill,
            coefficients=coefficients,
        )
        return Video.wrap_like(self, output)

    def elastic(
        self,
        displacement: torch.Tensor,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        fill: _FillTypeJIT = None,
    ) -> Video:
        output = self._F.elastic_video(
            self.as_subclass(torch.Tensor), displacement, interpolation=interpolation, fill=fill
        )
        return Video.wrap_like(self, output)

    def rgb_to_grayscale(self, num_output_channels: int = 1) -> Video:
        output = self._F.rgb_to_grayscale_image_tensor(
            self.as_subclass(torch.Tensor), num_output_channels=num_output_channels
        )
        return Video.wrap_like(self, output)

    def adjust_brightness(self, brightness_factor: float) -> Video:
        output = self._F.adjust_brightness_video(self.as_subclass(torch.Tensor), brightness_factor=brightness_factor)
        return Video.wrap_like(self, output)

    def adjust_saturation(self, saturation_factor: float) -> Video:
        output = self._F.adjust_saturation_video(self.as_subclass(torch.Tensor), saturation_factor=saturation_factor)
        return Video.wrap_like(self, output)

    def adjust_contrast(self, contrast_factor: float) -> Video:
        output = self._F.adjust_contrast_video(self.as_subclass(torch.Tensor), contrast_factor=contrast_factor)
        return Video.wrap_like(self, output)

    def adjust_sharpness(self, sharpness_factor: float) -> Video:
        output = self._F.adjust_sharpness_video(self.as_subclass(torch.Tensor), sharpness_factor=sharpness_factor)
        return Video.wrap_like(self, output)

    def adjust_hue(self, hue_factor: float) -> Video:
        output = self._F.adjust_hue_video(self.as_subclass(torch.Tensor), hue_factor=hue_factor)
        return Video.wrap_like(self, output)

    def adjust_gamma(self, gamma: float, gain: float = 1) -> Video:
        output = self._F.adjust_gamma_video(self.as_subclass(torch.Tensor), gamma=gamma, gain=gain)
        return Video.wrap_like(self, output)

    def posterize(self, bits: int) -> Video:
        output = self._F.posterize_video(self.as_subclass(torch.Tensor), bits=bits)
        return Video.wrap_like(self, output)

    def solarize(self, threshold: float) -> Video:
        output = self._F.solarize_video(self.as_subclass(torch.Tensor), threshold=threshold)
        return Video.wrap_like(self, output)

    def autocontrast(self) -> Video:
        output = self._F.autocontrast_video(self.as_subclass(torch.Tensor))
        return Video.wrap_like(self, output)

    def equalize(self) -> Video:
        output = self._F.equalize_video(self.as_subclass(torch.Tensor))
        return Video.wrap_like(self, output)

    def invert(self) -> Video:
        output = self._F.invert_video(self.as_subclass(torch.Tensor))
        return Video.wrap_like(self, output)

    def gaussian_blur(self, kernel_size: List[int], sigma: Optional[List[float]] = None) -> Video:
        output = self._F.gaussian_blur_video(self.as_subclass(torch.Tensor), kernel_size=kernel_size, sigma=sigma)
        return Video.wrap_like(self, output)

    def normalize(self, mean: List[float], std: List[float], inplace: bool = False) -> Video:
        output = self._F.normalize_video(self.as_subclass(torch.Tensor), mean=mean, std=std, inplace=inplace)
        return Video.wrap_like(self, output)


_VideoType = Union[torch.Tensor, Video]
_VideoTypeJIT = torch.Tensor
_TensorVideoType = Union[torch.Tensor, Video]
_TensorVideoTypeJIT = torch.Tensor
