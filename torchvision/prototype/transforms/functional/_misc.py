from typing import Optional, List

import PIL.Image
import torch
from torchvision.transforms import functional_tensor as _FT, functional_pil as _FP
from torchvision.transforms.functional import to_tensor, to_pil_image


get_dimensions_image_tensor = _FT.get_dimensions
get_dimensions_image_pil = _FP.get_dimensions

normalize_image_tensor = _FT.normalize


def gaussian_blur_image_tensor(
    img: torch.Tensor, kernel_size: List[int], sigma: Optional[List[float]] = None
) -> torch.Tensor:
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if len(kernel_size) != 2:
        raise ValueError(f"If kernel_size is a sequence its length should be 2. Got {len(kernel_size)}")
    for ksize in kernel_size:
        if ksize % 2 == 0 or ksize < 0:
            raise ValueError(f"kernel_size should have odd and positive integers. Got {kernel_size}")

    if sigma is None:
        sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]

    if sigma is not None and not isinstance(sigma, (int, float, list, tuple)):
        raise TypeError(f"sigma should be either float or sequence of floats. Got {type(sigma)}")
    if isinstance(sigma, (int, float)):
        sigma = [float(sigma), float(sigma)]
    if isinstance(sigma, (list, tuple)) and len(sigma) == 1:
        sigma = [sigma[0], sigma[0]]
    if len(sigma) != 2:
        raise ValueError(f"If sigma is a sequence, its length should be 2. Got {len(sigma)}")
    for s in sigma:
        if s <= 0.0:
            raise ValueError(f"sigma should have positive values. Got {sigma}")

    return _FT.gaussian_blur(img, kernel_size, sigma)


def gaussian_blur_image_pil(img: PIL.Image, kernel_size: List[int], sigma: Optional[List[float]] = None) -> PIL.Image:
    return to_pil_image(gaussian_blur_image_tensor(to_tensor(img), kernel_size=kernel_size, sigma=sigma))
