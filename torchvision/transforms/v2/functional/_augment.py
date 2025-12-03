import io
from typing import TYPE_CHECKING

import numpy as np

import PIL.Image

import torch
from torchvision import tv_tensors
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import _log_api_usage_once

from ._utils import _get_kernel, _import_cvcuda, _is_cvcuda_available, _register_kernel_internal


CVCUDA_AVAILABLE = _is_cvcuda_available()

if TYPE_CHECKING:
    import cvcuda  # type: ignore[import-not-found]
if CVCUDA_AVAILABLE:
    cvcuda = _import_cvcuda()  # noqa: F811


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


def _erase_cvcuda(
    image: "cvcuda.Tensor",
    i: int,
    j: int,
    h: int,
    w: int,
    v: torch.Tensor,
    inplace: bool = False,
) -> "cvcuda.Tensor":
    if inplace:
        raise ValueError("inplace is not supported for cvcuda.Tensor")

    anchor = torch.tensor(np.array([j, i]), dtype=torch.int32, device="cuda")
    cv_anchor = cvcuda.as_tensor(anchor, "NC").reshape((2,), "N")
    erasing = torch.tensor(np.array([w, h, 7]), dtype=torch.int32, device="cuda")
    cv_erasing = cvcuda.as_tensor(erasing, "NC").reshape((3,), "N")
    imgIdx = torch.tensor(np.array([0]), dtype=torch.int32, device="cuda")
    cv_imgIdx = cvcuda.as_tensor(imgIdx, "N").reshape((1,), "N")

    num_channels = image.shape[3]
    # Flatten v and expand to match the number of channels if it's a single value
    # CV-CUDA erase expects values as float32
    v_dup = v.clone()
    v_flat = v_dup.flatten().to(dtype=torch.float32, device="cuda")
    if v_flat.numel() == 1:
        v_flat = v_flat.repeat(num_channels)
    cv_values = cvcuda.as_tensor(v_flat, "NC").reshape((num_channels,), "N")

    result = cvcuda.erase(
        src=image,
        anchor=cv_anchor,
        erasing=cv_erasing,
        values=cv_values,
        imgIdx=cv_imgIdx,
        random=False,
        seed=0,
    )

    return result


if CVCUDA_AVAILABLE:
    _register_kernel_internal(erase, _import_cvcuda().Tensor)(_erase_cvcuda)


def jpeg(image: torch.Tensor, quality: int) -> torch.Tensor:
    """See :class:`~torchvision.transforms.v2.JPEG` for details."""
    if torch.jit.is_scripting():
        return jpeg_image(image, quality=quality)

    _log_api_usage_once(jpeg)

    kernel = _get_kernel(jpeg, type(image))
    return kernel(image, quality=quality)


@_register_kernel_internal(jpeg, torch.Tensor)
@_register_kernel_internal(jpeg, tv_tensors.Image)
def jpeg_image(image: torch.Tensor, quality: int) -> torch.Tensor:
    original_shape = image.shape
    image = image.view((-1,) + image.shape[-3:])

    if image.shape[0] == 0:  # degenerate
        return image.reshape(original_shape).clone()

    images = []
    for i in range(image.shape[0]):
        # isinstance checks are needed for torchscript.
        encoded_image = encode_jpeg(image[i], quality=quality)
        assert isinstance(encoded_image, torch.Tensor)
        decoded_image = decode_jpeg(encoded_image)
        assert isinstance(decoded_image, torch.Tensor)
        images.append(decoded_image)

    images = torch.stack(images, dim=0).view(original_shape)
    return images


@_register_kernel_internal(jpeg, tv_tensors.Video)
def jpeg_video(video: torch.Tensor, quality: int) -> torch.Tensor:
    return jpeg_image(video, quality=quality)


@_register_kernel_internal(jpeg, PIL.Image.Image)
def _jpeg_image_pil(image: PIL.Image.Image, quality: int) -> PIL.Image.Image:
    raw_jpeg = io.BytesIO()
    image.save(raw_jpeg, format="JPEG", quality=quality)

    # we need to copy since PIL.Image.open() will return PIL.JpegImagePlugin.JpegImageFile
    # which is a sub-class of PIL.Image.Image. this will fail check_transform() test.
    return PIL.Image.open(raw_jpeg).copy()
