from enum import Enum
from typing import List, Union
from warnings import warn

import torch

from ..extension import _load_library
from ..utils import _log_api_usage_once


try:
    _load_library("image")
except (ImportError, OSError) as e:
    warn(
        f"Failed to load image Python extension: '{e}'"
        f"If you don't plan on using image functionality from `torchvision.io`, you can ignore this warning. "
        f"Otherwise, there might be something wrong with your environment. "
        f"Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?"
    )


class ImageReadMode(Enum):
    """
    Support for various modes while reading images.

    Use ``ImageReadMode.UNCHANGED`` for loading the image as-is,
    ``ImageReadMode.GRAY`` for converting to grayscale,
    ``ImageReadMode.GRAY_ALPHA`` for grayscale with transparency,
    ``ImageReadMode.RGB`` for RGB and ``ImageReadMode.RGB_ALPHA`` for
    RGB with transparency.
    """

    UNCHANGED = 0
    GRAY = 1
    GRAY_ALPHA = 2
    RGB = 3
    RGB_ALPHA = 4


def read_file(path: str) -> torch.Tensor:
    """
    Reads and outputs the bytes contents of a file as a uint8 Tensor
    with one dimension.

    Args:
        path (str or ``pathlib.Path``): the path to the file to be read

    Returns:
        data (Tensor)
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(read_file)
    data = torch.ops.image.read_file(str(path))
    return data


def write_file(filename: str, data: torch.Tensor) -> None:
    """
    Writes the contents of an uint8 tensor with one dimension to a
    file.

    Args:
        filename (str or ``pathlib.Path``): the path to the file to be written
        data (Tensor): the contents to be written to the output file
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(write_file)
    torch.ops.image.write_file(str(filename), data)


def decode_png(
    input: torch.Tensor,
    mode: ImageReadMode = ImageReadMode.UNCHANGED,
    apply_exif_orientation: bool = False,
) -> torch.Tensor:
    """
    Decodes a PNG image into a 3 dimensional RGB or grayscale Tensor.
    Optionally converts the image to the desired format.
    The values of the output tensor are uint8 in [0, 255].

    Args:
        input (Tensor[1]): a one dimensional uint8 tensor containing
            the raw bytes of the PNG image.
        mode (ImageReadMode): the read mode used for optionally
            converting the image. Default: ``ImageReadMode.UNCHANGED``.
            See `ImageReadMode` class for more information on various
            available modes.
        apply_exif_orientation (bool): apply EXIF orientation transformation to the output tensor.
            Default: False.

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(decode_png)
    output = torch.ops.image.decode_png(input, mode.value, False, apply_exif_orientation)
    return output


def encode_png(input: torch.Tensor, compression_level: int = 6) -> torch.Tensor:
    """
    Takes an input tensor in CHW layout and returns a buffer with the contents
    of its corresponding PNG file.

    Args:
        input (Tensor[channels, image_height, image_width]): int8 image tensor of
            ``c`` channels, where ``c`` must 3 or 1.
        compression_level (int): Compression factor for the resulting file, it must be a number
            between 0 and 9. Default: 6

    Returns:
        Tensor[1]: A one dimensional int8 tensor that contains the raw bytes of the
            PNG file.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(encode_png)
    output = torch.ops.image.encode_png(input, compression_level)
    return output


def write_png(input: torch.Tensor, filename: str, compression_level: int = 6):
    """
    Takes an input tensor in CHW layout (or HW in the case of grayscale images)
    and saves it in a PNG file.

    Args:
        input (Tensor[channels, image_height, image_width]): int8 image tensor of
            ``c`` channels, where ``c`` must be 1 or 3.
        filename (str or ``pathlib.Path``): Path to save the image.
        compression_level (int): Compression factor for the resulting file, it must be a number
            between 0 and 9. Default: 6
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(write_png)
    output = encode_png(input, compression_level)
    write_file(filename, output)


def decode_jpeg(
    input: torch.Tensor,
    mode: ImageReadMode = ImageReadMode.UNCHANGED,
    device: str = "cpu",
    apply_exif_orientation: bool = False,
) -> torch.Tensor:
    """
    Decodes a JPEG image into a 3 dimensional RGB or grayscale Tensor.
    Optionally converts the image to the desired format.
    The values of the output tensor are uint8 between 0 and 255.

    Args:
        input (Tensor[1]): a one dimensional uint8 tensor containing
            the raw bytes of the JPEG image. This tensor must be on CPU,
            regardless of the ``device`` parameter.
        mode (ImageReadMode): the read mode used for optionally
            converting the image. The supported modes are: ``ImageReadMode.UNCHANGED``,
            ``ImageReadMode.GRAY`` and ``ImageReadMode.RGB``
            Default: ``ImageReadMode.UNCHANGED``.
            See ``ImageReadMode`` class for more information on various
            available modes.
        device (str or torch.device): The device on which the decoded image will
            be stored. If a cuda device is specified, the image will be decoded
            with `nvjpeg <https://developer.nvidia.com/nvjpeg>`_. This is only
            supported for CUDA version >= 10.1

            .. betastatus:: device parameter

            .. warning::
                There is a memory leak in the nvjpeg library for CUDA versions < 11.6.
                Make sure to rely on CUDA 11.6 or above before using ``device="cuda"``.
        apply_exif_orientation (bool): apply EXIF orientation transformation to the output tensor.
            Default: False. Only implemented for JPEG format on CPU.

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(decode_jpeg)
    device = torch.device(device)
    if device.type == "cuda":
        output = torch.ops.image.decode_jpeg_cuda(input, mode.value, device)
    else:
        output = torch.ops.image.decode_jpeg(input, mode.value, apply_exif_orientation)
    return output


def encode_jpeg(
    input: Union[torch.Tensor, List[torch.Tensor]], quality: int = 75
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Takes a (list of) input tensor(s) in CHW layout and returns a (list of) buffer(s) with the contents
    of the corresponding JPEG file(s).

    .. note::
        Passing a list of CUDA tensors is more efficient than repeated individual calls to ``encode_jpeg``.
        For CPU tensors the performance is equivalent.

    Args:
        input (Tensor[channels, image_height, image_width] or List[Tensor[channels, image_height, image_width]]):
            (list of) uint8 image tensor(s) of ``c`` channels, where ``c`` must be 1 or 3
        quality (int): Quality of the resulting JPEG file(s). Must be a number between
            1 and 100. Default: 75

    Returns:
        output (Tensor[1] or list[Tensor[1]]): A (list of) one dimensional uint8 tensor(s) that contain the raw bytes of the JPEG file.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(encode_jpeg)
    if quality < 1 or quality > 100:
        raise ValueError("Image quality should be a positive number between 1 and 100")
    if isinstance(input, list):
        if not input:
            raise ValueError("encode_jpeg requires at least one input tensor when a list is passed")
        if input[0].device.type == "cuda":
            return torch.ops.image.encode_jpegs_cuda(input, quality)
        else:
            return [torch.ops.image.encode_jpeg(image, quality) for image in input]
    else:  # single input tensor
        if input.device.type == "cuda":
            return torch.ops.image.encode_jpegs_cuda([input], quality)[0]
        else:
            return torch.ops.image.encode_jpeg(input, quality)


def write_jpeg(input: torch.Tensor, filename: str, quality: int = 75):
    """
    Takes an input tensor in CHW layout and saves it in a JPEG file.

    Args:
        input (Tensor[channels, image_height, image_width]): int8 image tensor of ``c``
            channels, where ``c`` must be 1 or 3.
        filename (str or ``pathlib.Path``): Path to save the image.
        quality (int): Quality of the resulting JPEG file, it must be a number
            between 1 and 100. Default: 75
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(write_jpeg)
    output = encode_jpeg(input, quality)
    assert isinstance(output, torch.Tensor)  # Needed for torchscript
    write_file(filename, output)


def decode_image(
    input: torch.Tensor,
    mode: ImageReadMode = ImageReadMode.UNCHANGED,
    apply_exif_orientation: bool = False,
) -> torch.Tensor:
    """
    Detect whether an image is a JPEG, PNG or GIF and performs the appropriate
    operation to decode the image into a 3 dimensional RGB or grayscale Tensor.

    Optionally converts the image to the desired format.
    The values of the output tensor are uint8 in [0, 255].

    Args:
        input (Tensor): a one dimensional uint8 tensor containing the raw bytes of the
            PNG or JPEG image.
        mode (ImageReadMode): the read mode used for optionally converting the image.
            Default: ``ImageReadMode.UNCHANGED``.
            See ``ImageReadMode`` class for more information on various
            available modes. Ignored for GIFs.
        apply_exif_orientation (bool): apply EXIF orientation transformation to the output tensor.
            Ignored for GIFs. Default: False.

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(decode_image)
    output = torch.ops.image.decode_image(input, mode.value, apply_exif_orientation)
    return output


def read_image(
    path: str,
    mode: ImageReadMode = ImageReadMode.UNCHANGED,
    apply_exif_orientation: bool = False,
) -> torch.Tensor:
    """
    Reads a JPEG, PNG or GIF image into a 3 dimensional RGB or grayscale Tensor.
    Optionally converts the image to the desired format.
    The values of the output tensor are uint8 in [0, 255].

    Args:
        path (str or ``pathlib.Path``): path of the JPEG, PNG or GIF image.
        mode (ImageReadMode): the read mode used for optionally converting the image.
            Default: ``ImageReadMode.UNCHANGED``.
            See ``ImageReadMode`` class for more information on various
            available modes. Ignored for GIFs.
        apply_exif_orientation (bool): apply EXIF orientation transformation to the output tensor.
            Ignored for GIFs. Default: False.

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(read_image)
    data = read_file(path)
    return decode_image(data, mode, apply_exif_orientation=apply_exif_orientation)


def _read_png_16(path: str, mode: ImageReadMode = ImageReadMode.UNCHANGED) -> torch.Tensor:
    data = read_file(path)
    return torch.ops.image.decode_png(data, mode.value, True)


def decode_gif(input: torch.Tensor) -> torch.Tensor:
    """
    Decode a GIF image into a 3 or 4 dimensional RGB Tensor.

    The values of the output tensor are uint8 between 0 and 255.
    The output tensor has shape ``(C, H, W)`` if there is only one image in the
    GIF, and ``(N, C, H, W)`` if there are ``N`` images.

    Args:
        input (Tensor[1]): a one dimensional contiguous uint8 tensor containing
            the raw bytes of the GIF image.

    Returns:
        output (Tensor[image_channels, image_height, image_width] or Tensor[num_images, image_channels, image_height, image_width])
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(decode_gif)
    return torch.ops.image.decode_gif(input)
