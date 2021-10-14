from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from ... import transforms as T
from ...transforms import functional as F


__all__ = ["CocoEval", "ImageNetEval"]


class CocoEval(nn.Module):
    def forward(
        self, img: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        return F.convert_image_dtype(img, torch.float), target


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
        self._resize = T.Resize(resize_size, interpolation=interpolation)
        self._crop = T.CenterCrop(crop_size)
        self._normalize = T.Normalize(mean=mean, std=std)

    def forward(self, img: Tensor) -> Tensor:
        img = self._crop(self._resize(img))
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        return self._normalize(img)
