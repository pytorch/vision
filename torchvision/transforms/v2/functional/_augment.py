import ctypes
import io
from typing import TYPE_CHECKING

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


def _erase_image_cvcuda(
    image: "cvcuda.Tensor",
    i: int,
    j: int,
    h: int,
    w: int,
    v: torch.Tensor,
    inplace: bool = False,
) -> "cvcuda.Tensor":
    cvcuda = _import_cvcuda()

    if inplace:
        raise ValueError("inplace is not supported for cvcuda.Tensor")

    # Load CUDA runtime for memory copy
    try:
        cudart = ctypes.CDLL("libcudart.so")
    except OSError:
        cudart = ctypes.CDLL("libcudart.so.12")
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cudart.cudaMemcpy.restype = ctypes.c_int
    CUDA_MEMCPY_D2D = 3  # cudaMemcpyDeviceToDevice

    num_erasing_areas = 1
    num_channels = image.shape[3]  # NHWC layout

    # Create CV-CUDA tensors with proper compound types
    cv_anchor = cvcuda.Tensor((num_erasing_areas,), cvcuda.Type._2S32, "N")
    cv_erasing = cvcuda.Tensor((num_erasing_areas,), cvcuda.Type._3S32, "N")
    cv_imgIdx = cvcuda.Tensor((num_erasing_areas,), cvcuda.Type.S32, "N")
    # Values tensor - 4 floats per erasing area (CV-CUDA standard, supports up to RGBA)
    cv_values = cvcuda.Tensor((num_erasing_areas * 4,), cvcuda.Type.F32, "N")

    # Create source torch tensors with the data
    anchor_src = torch.tensor([j, i], dtype=torch.int32, device="cuda")
    # The third value is a bitmask for which channels to fill: 7 = 0b111 = RGB all channels
    channel_mask = (1 << num_channels) - 1  # e.g., 3 channels -> 0b111 = 7
    erasing_src = torch.tensor([w, h, channel_mask], dtype=torch.int32, device="cuda")
    imgIdx_src = torch.tensor([0], dtype=torch.int32, device="cuda")

    # Get fill values for erasing - need 4 floats per erasing area (CV-CUDA format)
    v_flat = v.flatten().to(dtype=torch.float32, device="cuda")
    # Expand to 4 values (CV-CUDA always expects 4)
    if v_flat.numel() == 1:
        # Single value - replicate to all 4 slots
        values_src = v_flat.expand(4).contiguous()
    elif v_flat.numel() >= 4:
        # Has enough values, take first 4
        values_src = v_flat[:4].contiguous()
    else:
        # Has fewer than 4 values, pad with zeros
        padding = torch.zeros(4 - v_flat.numel(), dtype=torch.float32, device="cuda")
        values_src = torch.cat([v_flat, padding])

    # Copy data from torch tensors to CV-CUDA tensors using cudaMemcpy
    cudart.cudaMemcpy(
        cv_anchor.cuda().__cuda_array_interface__["data"][0],
        anchor_src.data_ptr(),
        8,
        CUDA_MEMCPY_D2D,  # 2 x int32 = 8 bytes
    )
    cudart.cudaMemcpy(
        cv_erasing.cuda().__cuda_array_interface__["data"][0],
        erasing_src.data_ptr(),
        12,
        CUDA_MEMCPY_D2D,  # 3 x int32 = 12 bytes
    )
    cudart.cudaMemcpy(
        cv_imgIdx.cuda().__cuda_array_interface__["data"][0],
        imgIdx_src.data_ptr(),
        4,
        CUDA_MEMCPY_D2D,  # 1 x int32 = 4 bytes
    )
    cudart.cudaMemcpy(
        cv_values.cuda().__cuda_array_interface__["data"][0],
        values_src.data_ptr(),
        16,
        CUDA_MEMCPY_D2D,  # 4 x float32 = 16 bytes
    )

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
    _register_kernel_internal(erase, _import_cvcuda().Tensor)(_erase_image_cvcuda)


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
