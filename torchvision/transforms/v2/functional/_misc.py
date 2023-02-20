import math
from typing import List, Optional, Union

import PIL.Image
import torch
from torch.nn.functional import conv2d, pad as torch_pad

from torchvision import datapoints
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from torchvision.utils import _log_api_usage_once

from ._utils import is_simple_tensor


def normalize_image_tensor(
    image: torch.Tensor, mean: List[float], std: List[float], inplace: bool = False
) -> torch.Tensor:
    if not image.is_floating_point():
        raise TypeError(f"Input tensor should be a float tensor. Got {image.dtype}.")

    if image.ndim < 3:
        raise ValueError(f"Expected tensor to be a tensor image of size (..., C, H, W). Got {image.shape}.")

    if isinstance(std, (tuple, list)):
        divzero = not all(std)
    elif isinstance(std, (int, float)):
        divzero = std == 0
    else:
        divzero = False
    if divzero:
        raise ValueError("std evaluated to zero, leading to division by zero.")

    dtype = image.dtype
    device = image.device
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    std = torch.as_tensor(std, dtype=dtype, device=device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    if inplace:
        image = image.sub_(mean)
    else:
        image = image.sub(mean)

    return image.div_(std)


def normalize_video(video: torch.Tensor, mean: List[float], std: List[float], inplace: bool = False) -> torch.Tensor:
    return normalize_image_tensor(video, mean, std, inplace=inplace)


def normalize(
    inpt: Union[datapoints._TensorImageTypeJIT, datapoints._TensorVideoTypeJIT],
    mean: List[float],
    std: List[float],
    inplace: bool = False,
) -> torch.Tensor:
    if not torch.jit.is_scripting():
        _log_api_usage_once(normalize)
    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return normalize_image_tensor(inpt, mean=mean, std=std, inplace=inplace)
    elif isinstance(inpt, (datapoints.Image, datapoints.Video)):
        return inpt.normalize(mean=mean, std=std, inplace=inplace)
    else:
        raise TypeError(
            f"Input can either be a plain tensor or an `Image` or `Video` datapoint, " f"but got {type(inpt)} instead."
        )


def _get_gaussian_kernel1d(kernel_size: int, sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    lim = (kernel_size - 1) / (2.0 * math.sqrt(2.0) * sigma)
    x = torch.linspace(-lim, lim, steps=kernel_size, dtype=dtype, device=device)
    kernel1d = torch.softmax(x.pow_(2).neg_(), dim=0)
    return kernel1d


def _get_gaussian_kernel2d(
    kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0], dtype, device)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1], dtype, device)
    kernel2d = kernel1d_y.unsqueeze(-1) * kernel1d_x
    return kernel2d


def gaussian_blur_image_tensor(
    image: torch.Tensor, kernel_size: List[int], sigma: Optional[List[float]] = None
) -> torch.Tensor:
    # TODO: consider deprecating integers from sigma on the future
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    elif len(kernel_size) != 2:
        raise ValueError(f"If kernel_size is a sequence its length should be 2. Got {len(kernel_size)}")
    for ksize in kernel_size:
        if ksize % 2 == 0 or ksize < 0:
            raise ValueError(f"kernel_size should have odd and positive integers. Got {kernel_size}")

    if sigma is None:
        sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]
    else:
        if isinstance(sigma, (list, tuple)):
            length = len(sigma)
            if length == 1:
                s = float(sigma[0])
                sigma = [s, s]
            elif length != 2:
                raise ValueError(f"If sigma is a sequence, its length should be 2. Got {length}")
        elif isinstance(sigma, (int, float)):
            s = float(sigma)
            sigma = [s, s]
        else:
            raise TypeError(f"sigma should be either float or sequence of floats. Got {type(sigma)}")
    for s in sigma:
        if s <= 0.0:
            raise ValueError(f"sigma should have positive values. Got {sigma}")

    if image.numel() == 0:
        return image

    dtype = image.dtype
    shape = image.shape
    ndim = image.ndim
    if ndim == 3:
        image = image.unsqueeze(dim=0)
    elif ndim > 4:
        image = image.reshape((-1,) + shape[-3:])

    fp = torch.is_floating_point(image)
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype if fp else torch.float32, device=image.device)
    kernel = kernel.expand(shape[-3], 1, kernel.shape[0], kernel.shape[1])

    output = image if fp else image.to(dtype=torch.float32)

    # padding = (left, right, top, bottom)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    output = torch_pad(output, padding, mode="reflect")
    output = conv2d(output, kernel, groups=shape[-3])

    if ndim == 3:
        output = output.squeeze(dim=0)
    elif ndim > 4:
        output = output.reshape(shape)

    if not fp:
        output = output.round_().to(dtype=dtype)

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
    inpt: datapoints._InputTypeJIT, kernel_size: List[int], sigma: Optional[List[float]] = None
) -> datapoints._InputTypeJIT:
    if not torch.jit.is_scripting():
        _log_api_usage_once(gaussian_blur)

    if torch.jit.is_scripting() or is_simple_tensor(inpt):
        return gaussian_blur_image_tensor(inpt, kernel_size=kernel_size, sigma=sigma)
    elif isinstance(inpt, datapoints._datapoint.Datapoint):
        return inpt.gaussian_blur(kernel_size=kernel_size, sigma=sigma)
    elif isinstance(inpt, PIL.Image.Image):
        return gaussian_blur_image_pil(inpt, kernel_size=kernel_size, sigma=sigma)
    else:
        raise TypeError(
            f"Input can either be a plain tensor, any TorchVision datapoint, or a PIL image, "
            f"but got {type(inpt)} instead."
        )
