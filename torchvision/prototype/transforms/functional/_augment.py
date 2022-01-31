from typing import Tuple

import torch
from torchvision.transforms import functional as _F


erase_image = _F.erase


def _mixup(input: torch.Tensor, batch_dim: int, lam: float, inplace: bool) -> torch.Tensor:
    if not inplace:
        input = input.clone()

    input_rolled = input.roll(1, batch_dim)
    return input.mul_(lam).add_(input_rolled.mul_(1 - lam))


def mixup_image(image_batch: torch.Tensor, *, lam: float, inplace: bool = False) -> torch.Tensor:
    return _mixup(image_batch, -4, lam, inplace)


def mixup_one_hot_label(one_hot_label_batch: torch.Tensor, *, lam: float, inplace: bool = False) -> torch.Tensor:
    return _mixup(one_hot_label_batch, -2, lam, inplace)


def cutmix_image(image: torch.Tensor, *, box: Tuple[int, int, int, int], inplace: bool = False) -> torch.Tensor:
    if not inplace:
        image = image.clone()

    x1, y1, x2, y2 = box
    image_rolled = image.roll(1, -4)

    image[..., y1:y2, x1:x2] = image_rolled[..., y1:y2, x1:x2]
    return image


def cutmix_one_hot_label(
    one_hot_label_batch: torch.Tensor, *, lam_adjusted: float, inplace: bool = False
) -> torch.Tensor:
    return mixup_one_hot_label(one_hot_label_batch, lam=lam_adjusted, inplace=inplace)
