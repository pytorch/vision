from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from ... import transforms as T
from ...transforms import functional as F


__all__ = ["CocoEval", "ImageNetEval", "Kinect400Eval", "VocEval"]


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


class Kinect400Eval(nn.Module):
    def __init__(
        self,
        resize_size: Tuple[int, int],
        crop_size: Tuple[int, int],
        mean: Tuple[float, ...] = (0.43216, 0.394666, 0.37645),
        std: Tuple[float, ...] = (0.22803, 0.22145, 0.216989),
        interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self._convert = T.ConvertImageDtype(torch.float)
        self._resize = T.Resize(resize_size, interpolation=interpolation)
        self._normalize = T.Normalize(mean=mean, std=std)
        self._crop = T.CenterCrop(crop_size)

    def forward(self, vid: Tensor) -> Tensor:
        vid = vid.permute(0, 3, 1, 2)  # (T, H, W, C) => (T, C, H, W)
        vid = self._convert(vid)
        vid = self._resize(vid)
        vid = self._normalize(vid)
        vid = self._crop(vid)
        return vid.permute(1, 0, 2, 3)  # (T, C, H, W) => (C, T, H, W)


class VocEval(nn.Module):
    def __init__(
        self,
        resize_size: int,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
        interpolation_target: T.InterpolationMode = T.InterpolationMode.NEAREST,
    ) -> None:
        super().__init__()
        self._size = [resize_size]
        self._mean = list(mean)
        self._std = list(std)
        self._interpolation = interpolation
        self._interpolation_target = interpolation_target

    def forward(self, img: Tensor, target: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        img = F.resize(img, self._size, interpolation=self._interpolation)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self._mean, std=self._std)
        if target:
            target = F.resize(target, self._size, interpolation=self._interpolation_target)
            if not isinstance(target, Tensor):
                target = F.pil_to_tensor(target)
            target = target.squeeze(0).to(torch.int64)
        return img, target
