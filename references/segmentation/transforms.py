import random
from typing import Tuple

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F, InterpolationMode


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


class ScaleJitter:
    # Referenced from references/detection/transforms.py

    """Randomly resizes the image and its mask within the specified scale range.
    The class implements the Scale Jitter augmentation as described in the paper
    `"Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" <https://arxiv.org/abs/2012.07177>`_.

    Args:
        target_size (tuple of ints): The target size for the transform provided in (height, weight) format.
        scale_range (tuple of ints): scaling factor interval, e.g (a, b), then scale is randomly sampled from the
            range a <= scale <= b.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
    """

    def __init__(
        self,
        target_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (0.1, 2.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.target_size = target_size
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_height, orig_width = F.get_dimensions(image)

        scale = self.scale_range[0] + torch.rand(1) * (self.scale_range[1] - self.scale_range[0])
        r = min(self.target_size[1] / orig_height, self.target_size[0] / orig_width) * scale
        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        image = F.resize(image, [new_height, new_width], interpolation=self.interpolation)
        target = F.resize(torch.unsqueeze(target, 0), [new_height, new_width], interpolation=InterpolationMode.NEAREST)

        return image, target


class FixedSizeCrop:
    # Referenced from references/detection/transforms.py

    def __init__(self, size, fill=0, padding_mode="constant"):
        super().__init__()
        size = tuple(T.transforms._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.crop_height = size[0]
        self.crop_width = size[1]
        self.fill = fill
        self.padding_mode = padding_mode

    def _pad(self, image, target, padding):
        if isinstance(padding, int):
            pad_left = pad_right = pad_top = pad_bottom = padding
        elif len(padding) == 1:
            pad_left = pad_right = pad_top = pad_bottom = padding[0]
        elif len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        elif len(padding) == 4:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]
        else:
            # TODO: fix this error
            raise ValueError("padding ndim should be int, (int, int) or (int, int, int, int)")

        padding = [pad_left, pad_top, pad_right, pad_bottom]
        image = F.pad(image, padding, self.fill, self.padding_mode)
        target = F.pad(target, padding, 0, self.padding_mode)

        return image, target

    def _crop(self, image, target, top, left, height, width):
        image = F.crop(image, top, left, height, width)
        target = F.crop(target, top, left, height, width)
        return image, target

    def __call__(self, img, target=None):
        _, height, width = F.get_dimensions(img)
        new_height = min(height, self.crop_height)
        new_width = min(width, self.crop_width)

        if new_height != height or new_width != width:
            offset_height = max(height - self.crop_height, 0)
            offset_width = max(width - self.crop_width, 0)

            r = torch.rand(1)
            top = int(offset_height * r)
            left = int(offset_width * r)

            img, target = self._crop(img, target, top, left, new_height, new_width)

        pad_bottom = max(self.crop_height - new_height, 0)
        pad_right = max(self.crop_width - new_width, 0)
        if pad_bottom != 0 or pad_right != 0:
            img, target = self._pad(img, target, [0, 0, pad_right, pad_bottom])

        return img, target


class SimpleCopyPaste(torch.nn.Module):
    def __init__(self, p: float = 0.5, jittering_type="LSJ", inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

        # TODO: Apply random scale jittering ( resize and crop )
        if jittering_type == "LSJ":
            scale_range = (0.1, 2.0)
        elif jittering_type == "SSJ":
            scale_range = (0.8, 1.25)
        else:
            # TODO: add invalid option error
            raise ValueError("Invalid jittering type")

        self.transforms = Compose(
            [
                ScaleJitter(target_size=(1024, 1024), scale_range=scale_range),
                FixedSizeCrop(size=(1024, 1024), fill=105),
                RandomHorizontalFlip(0.5),
            ]
        )

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
        for i, (image, mask) in enumerate(zip(batch, target)):
            batch[i], target[i] = self.transforms(image, mask)

        # if not self.inplace:
        #     batch = batch.clone()
        #     target = target.clone()

        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

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
