import random
from typing import Tuple

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class SimpleCopyPaste(torch.nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, batch: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # validate inputs
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}.")
        if target.ndim != 3:
            raise ValueError(f"Target ndim should be 3. Got {target.ndim}.")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        # check inplace
        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # TODO: Apply random scale jittering and random horizontal flipping

        # TODO: Pad images smaller than their original size with gray pixel values

        # TODO: select a random subset of objects from one of the images and paste them onto the other image

        # TODO: Smooth out the edges of the pasted objects using a Gaussian filter on the mask

        # get binary paste mask
        paste_binary_mask = (target_rolled != 0).to(target_rolled.dtype)
        # delete pixels from source mask using paste mask
        target.mul_(1 - paste_binary_mask)
        # Combine paste mask with source mask
        target.add_(target_rolled)

        # get paste image using paste image mask
        paste_image = batch_rolled * torch.unsqueeze(paste_binary_mask, 1)
        # delete pixels from source image using paste binary mask
        batch.mul_(torch.unsqueeze(1 - paste_binary_mask, 1))
        # Combine paste image with source image
        batch.add_(paste_image)

        return batch, target

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(" f", p={self.p}" f", inplace={self.inplace}" f")"
        return s
