from typing import Tuple

import torch
from torch import Tensor, nn

from ... import transforms as T
from ...transforms import functional as F


__all__ = ["ConvertImageDtype", "ImageNetEval"]


# Allows handling of both PIL and Tensor images
class ConvertImageDtype(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, img: Tensor) -> Tensor:
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        return F.convert_image_dtype(img, self.dtype)


class ImageNetEval(nn.Module):
    def __init__(
        self,
        crop_size: int,
        resize_size: int = 256,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.transforms = T.Compose(
            [
                T.Resize(resize_size, interpolation=interpolation),
                T.CenterCrop(crop_size),
                ConvertImageDtype(dtype=torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def forward(self, img: Tensor) -> Tensor:
        return self.transforms(img)
