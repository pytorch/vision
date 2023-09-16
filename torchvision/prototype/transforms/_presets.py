"""
This file is part of the private API. Please do not use directly these classes as they will be modified on
future versions without warning. The classes should be accessed only via the transforms argument of Weights.
"""
from typing import List, Optional, Tuple, Union

import PIL.Image

import torch
from torch import Tensor

from torchvision.transforms.v2 import functional as F, InterpolationMode

from torchvision.transforms.v2.functional._geometry import _check_interpolation

__all__ = ["StereoMatching"]


class StereoMatching(torch.nn.Module):
    def __init__(
        self,
        *,
        use_gray_scale: bool = False,
        resize_size: Optional[Tuple[int, ...]],
        mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
        std: Tuple[float, ...] = (0.5, 0.5, 0.5),
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()

        # pacify mypy
        self.resize_size: Union[None, List]

        if resize_size is not None:
            self.resize_size = list(resize_size)
        else:
            self.resize_size = None

        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = _check_interpolation(interpolation)
        self.use_gray_scale = use_gray_scale

    def forward(self, left_image: Tensor, right_image: Tensor) -> Tuple[Tensor, Tensor]:
        def _process_image(img: PIL.Image.Image) -> Tensor:
            if not isinstance(img, Tensor):
                img = F.pil_to_tensor(img)
            if self.resize_size is not None:
                # We hard-code antialias=False to preserve results after we changed
                # its default from None to True (see
                # https://github.com/pytorch/vision/pull/7160)
                # TODO: we could re-train the stereo models with antialias=True?
                img = F.resize(img, self.resize_size, interpolation=self.interpolation, antialias=False)
            if self.use_gray_scale is True:
                img = F.rgb_to_grayscale(img)
            img = F.convert_image_dtype(img, torch.float)
            img = F.normalize(img, mean=self.mean, std=self.std)
            img = img.contiguous()
            return img

        left_image = _process_image(left_image)
        right_image = _process_image(right_image)
        return left_image, right_image

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``. "
            f"Finally the values are first rescaled to ``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and "
            f"``std={self.std}``."
        )
