from typing import List, Optional

import PIL.Image
import torch
from torchvision.prototype import features
from torchvision.transforms import functional_tensor as _FT
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

normalize_image_tensor = _FT.normalize


def normalize(
    inpt: features.TensorImageTypeJIT, mean: List[float], std: List[float], inplace: bool = False
) -> torch.Tensor:
    if torch.jit.is_scripting():
        correct_type = isinstance(inpt, torch.Tensor)
    else:
        correct_type = features.is_simple_tensor(inpt) or isinstance(inpt, features.Image)
        inpt = inpt.as_subclass(torch.Tensor)  # type: ignore[arg-type]
    if not correct_type:
        raise TypeError(f"img should be Tensor Image. Got {type(inpt)}")

    # Image instance after normalization is not Image anymore due to unknown data range
    # Thus we return Tensor for input Image
    return normalize_image_tensor(inpt, mean=mean, std=std, inplace=inplace)


def gaussian_blur_image_tensor(
    image: torch.Tensor, kernel_size: List[int], sigma: Optional[List[float]] = None
) -> torch.Tensor:
    # TODO: consider deprecating integers from sigma on the future
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

    return _FT.gaussian_blur(image, kernel_size, sigma)


@torch.jit.unused
def gaussian_blur_image_pil(
    image: PIL.Image.Image, kernel_size: List[int], sigma: Optional[List[float]] = None
) -> PIL.Image.Image:
    t_img = pil_to_tensor(image)
    output = gaussian_blur_image_tensor(t_img, kernel_size=kernel_size, sigma=sigma)
    return to_pil_image(output, mode=image.mode)


def gaussian_blur(
    inpt: features.InputTypeJIT, kernel_size: List[int], sigma: Optional[List[float]] = None
) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return gaussian_blur_image_tensor(inpt, kernel_size=kernel_size, sigma=sigma)
    elif isinstance(inpt, features._Feature):
        return inpt.gaussian_blur(kernel_size=kernel_size, sigma=sigma)
    else:
        return gaussian_blur_image_pil(inpt, kernel_size=kernel_size, sigma=sigma)
