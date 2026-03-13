import io
from types import SimpleNamespace
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

    # the v tensor is random if it has spatial dimensions > 1x1
    is_random_fill = v.shape[-2:] != (1, 1)

    # allocate any space for standard torch tensors
    mask = (1 << image.shape[3]) - 1
    src_anchor = torch.tensor([[j, i]], dtype=torch.int32, device="cuda")
    src_erasing = torch.tensor([[w, h, mask]], dtype=torch.int32, device="cuda")
    src_idx = torch.tensor([0], dtype=torch.int32, device="cuda")

    # allocate the fill values based on if random or not
    # use zeros for random fill since we have to pass the tensor to the kernel anyway
    if is_random_fill:
        src_vals = torch.zeros(4, device="cuda", dtype=torch.float32)
    # CV-CUDA requires that the fill values is a flat size 4 tensor
    # so we need to flatten the fill values and pad with zeros if needed
    else:
        v_flat = v.flatten().to(dtype=torch.float32, device="cuda")
        if v_flat.numel() == 1:
            src_vals = v_flat.expand(4).contiguous()
        else:
            if v_flat.numel() >= 4:
                src_vals = v_flat[:4]
            else:
                pad_len = 4 - v_flat.numel()
                src_vals = torch.cat([v_flat, torch.zeros(pad_len, device="cuda", dtype=torch.float32)])
            src_vals = src_vals.contiguous()

    # the simple tensors can be read directly by CV-CUDA
    cv_imgIdx = cvcuda.as_tensor(
        src_idx.reshape(
            1,
        ),
        "N",
    )
    cv_values = cvcuda.as_tensor(
        src_vals.reshape(
            1 * 4,
        ),
        "N",
    )

    # packed types (_2S32, _3S32) need to be copied into pre-allocated tensors
    # torch does not support these packed types directly, so we create a helper function
    # which will enable torch copy into the data directly (by overriding type/strides info)
    def _to_torch(cv_tensor: cvcuda.Tensor, shape: tuple[int, ...], typestr: str) -> torch.Tensor:
        iface = cv_tensor.cuda().__cuda_array_interface__
        iface.update(shape=shape, typestr=typestr, strides=None)
        return torch.as_tensor(SimpleNamespace(__cuda_array_interface__=iface), device="cuda")

    # allocate the data for packed types
    cv_anchor = cvcuda.Tensor((1,), cvcuda.Type._2S32, "N")
    cv_erasing = cvcuda.Tensor((1,), cvcuda.Type._3S32, "N")

    # do a memcpy with torch, pretending data is scalar type contiguous
    _to_torch(cv_anchor, (1, 2), "<i4").copy_(src_anchor)
    _to_torch(cv_erasing, (1, 3), "<i4").copy_(src_erasing)

    # derive seed from torch's RNG so CV-CUDA is deterministic when user sets torch.manual_seed()
    seed = 0
    if is_random_fill:
        seed = int(torch.randint(0, 2147483648, (1,)).item())

    return cvcuda.erase(
        src=image,
        anchor=cv_anchor,
        erasing=cv_erasing,
        values=cv_values,
        imgIdx=cv_imgIdx,
        random=is_random_fill,
        seed=seed,
    )


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
