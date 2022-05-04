from typing import Tuple

import torch
from torchvision.transforms import functional_tensor as _FT


erase_image_tensor = _FT.erase


def _mixup_tensor(input: torch.Tensor, batch_dim: int, lam: float) -> torch.Tensor:
    input = input.clone()
    return input.roll(1, batch_dim).mul_(1 - lam).add_(input.mul_(lam))


def mixup_image_tensor(image_batch: torch.Tensor, *, lam: float) -> torch.Tensor:
    if image_batch.ndim < 4:
        raise ValueError("Need a batch of images")

    return _mixup_tensor(image_batch, -4, lam)


def mixup_one_hot_label(one_hot_label_batch: torch.Tensor, *, lam: float) -> torch.Tensor:
    if one_hot_label_batch.ndim < 2:
        raise ValueError("Need a batch of one hot labels")

    return _mixup_tensor(one_hot_label_batch, -2, lam)


def cutmix_image_tensor(image_batch: torch.Tensor, *, box: Tuple[int, int, int, int]) -> torch.Tensor:
    if image_batch.ndim < 4:
        raise ValueError("Need a batch of images")

    x1, y1, x2, y2 = box
    image_rolled = image_batch.roll(1, -4)

    image_batch = image_batch.clone()
    image_batch[..., y1:y2, x1:x2] = image_rolled[..., y1:y2, x1:x2]
    return image_batch


def cutmix_one_hot_label(one_hot_label_batch: torch.Tensor, *, lam_adjusted: float) -> torch.Tensor:
    if one_hot_label_batch.ndim < 2:
        raise ValueError("Need a batch of one hot labels")

    return _mixup_tensor(one_hot_label_batch, -2, lam_adjusted)
