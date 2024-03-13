import io

import PIL.Image

import torch
from torchvision import tv_tensors
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import _log_api_usage_once

from ._utils import _get_kernel, _register_kernel_internal


def erase(
    inpt: torch.Tensor,
    i: int,
    j: int,
    h: int,
    w: int,
    v: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.RandomErase` for details."""
    if torch.jit.is_scripting():
        return erase_image(inpt, i=i, j=j, h=h, w=w, v=v, inplace=inplace)

    _log_api_usage_once(erase)

    kernel = _get_kernel(erase, type(inpt))
    return kernel(inpt, i=i, j=j, h=h, w=w, v=v, inplace=inplace)


@_register_kernel_internal(erase, torch.Tensor)
@_register_kernel_internal(erase, tv_tensors.Image)
def erase_image(
    image: torch.Tensor, i: int, j: int, h: int, w: int, v: torch.Tensor, inplace: bool = False
) -> torch.Tensor:
    if not inplace:
        image = image.clone()

    image[..., i : i + h, j : j + w] = v
    return image


@_register_kernel_internal(erase, PIL.Image.Image)
def _erase_image_pil(
    image: PIL.Image.Image, i: int, j: int, h: int, w: int, v: torch.Tensor, inplace: bool = False
) -> PIL.Image.Image:
    t_img = pil_to_tensor(image)
    output = erase_image(t_img, i=i, j=j, h=h, w=w, v=v, inplace=inplace)
    return to_pil_image(output, mode=image.mode)


@_register_kernel_internal(erase, tv_tensors.Video)
def erase_video(
    video: torch.Tensor, i: int, j: int, h: int, w: int, v: torch.Tensor, inplace: bool = False
) -> torch.Tensor:
    return erase_image(video, i=i, j=j, h=h, w=w, v=v, inplace=inplace)


def jpeg(image: torch.Tensor, quality: int) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.JPEG` for details."""
    if torch.jit.is_scripting():
        return jpeg_image(image, quality=quality)

    _log_api_usage_once(jpeg_image)

    kernel = _get_kernel(jpeg_image, type(image))
    return kernel(image, quality=quality)


@_register_kernel_internal(jpeg, torch.Tensor)
@_register_kernel_internal(jpeg, tv_tensors.Image)
def jpeg_image(image: torch.Tensor, quality: int) -> torch.Tensor:
    leading_dims = image.shape[:-3]
    image = image.reshape(-1, *image.shape[-3:])
    image = [decode_jpeg(encode_jpeg(x, quality=quality)) for x in image]
    image = torch.stack(image, dim=0).unflatten(0, leading_dims)
    return image


@_register_kernel_internal(erase, PIL.Image.Image)
def _jpeg_image_pil(image: PIL.Image.Image, quality: int) -> PIL.Image.Image:
    raw_jpeg = io.BytesIO()
    image.save(raw_jpeg, format="JPEG", quality=quality)
    return PIL.Image.open(raw_jpeg)
