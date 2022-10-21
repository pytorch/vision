import math
from typing import List, Optional, Union

import PIL.Image
import torch
from torch.nn.functional import conv2d, pad as torch_pad
from torchvision.prototype import features
from torchvision.transforms import functional_tensor as _FT
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

normalize_image_tensor = _FT.normalize


def normalize_video(video: torch.Tensor, mean: List[float], std: List[float], inplace: bool = False) -> torch.Tensor:
    return normalize_image_tensor(video, mean, std, inplace=inplace)


def normalize(
    inpt: Union[features.TensorImageTypeJIT, features.TensorVideoTypeJIT],
    mean: List[float],
    std: List[float],
    inplace: bool = False,
) -> torch.Tensor:
    if torch.jit.is_scripting():
        correct_type = isinstance(inpt, torch.Tensor)
    else:
        correct_type = features.is_simple_tensor(inpt) or isinstance(inpt, (features.Image, features.Video))
        inpt = inpt.as_subclass(torch.Tensor)
    if not correct_type:
        raise TypeError(f"img should be Tensor Image. Got {type(inpt)}")

    # Image or Video type should not be retained after normalization due to unknown data range
    # Thus we return Tensor for input Image
    return normalize_image_tensor(inpt, mean=mean, std=std, inplace=inplace)


def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> torch.Tensor:
    lim = (kernel_size - 1) / (2 * math.sqrt(2) * sigma)
    x = torch.linspace(-lim, lim, steps=kernel_size)
    kernel1d = torch.softmax(-x.pow_(2), dim=0)
    return kernel1d


def _get_gaussian_kernel2d(
    kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = kernel1d_y.unsqueeze(-1) * kernel1d_x
    return kernel2d


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

    if image.numel() == 0:
        return image

    shape = image.shape

    if image.ndim > 4:
        image = image.reshape((-1,) + shape[-3:])
        needs_unsquash = True
    else:
        needs_unsquash = False

    dtype = image.dtype if torch.is_floating_point(image) else torch.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=image.device)
    kernel = kernel.expand(image.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    image, need_cast, need_squeeze, out_dtype = _FT._cast_squeeze_in(image, [kernel.dtype])

    # padding = (left, right, top, bottom)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    output = torch_pad(image, padding, mode="reflect")
    output = conv2d(output, kernel, groups=output.shape[-3])

    output = _FT._cast_squeeze_out(output, need_cast, need_squeeze, out_dtype)

    if needs_unsquash:
        output = output.reshape(shape)

    return output


@torch.jit.unused
def gaussian_blur_image_pil(
    image: PIL.Image.Image, kernel_size: List[int], sigma: Optional[List[float]] = None
) -> PIL.Image.Image:
    t_img = pil_to_tensor(image)
    output = gaussian_blur_image_tensor(t_img, kernel_size=kernel_size, sigma=sigma)
    return to_pil_image(output, mode=image.mode)


def gaussian_blur_video(
    video: torch.Tensor, kernel_size: List[int], sigma: Optional[List[float]] = None
) -> torch.Tensor:
    return gaussian_blur_image_tensor(video, kernel_size, sigma)


def gaussian_blur(
    inpt: features.InputTypeJIT, kernel_size: List[int], sigma: Optional[List[float]] = None
) -> features.InputTypeJIT:
    if isinstance(inpt, torch.Tensor) and (torch.jit.is_scripting() or not isinstance(inpt, features._Feature)):
        return gaussian_blur_image_tensor(inpt, kernel_size=kernel_size, sigma=sigma)
    elif isinstance(inpt, features._Feature):
        return inpt.gaussian_blur(kernel_size=kernel_size, sigma=sigma)
    else:
        return gaussian_blur_image_pil(inpt, kernel_size=kernel_size, sigma=sigma)
