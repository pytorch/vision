import torch
from enum import Enum

from .._internally_replaced_utils import _get_extension_path


try:
    lib_path = _get_extension_path('image')
    torch.ops.load_library(lib_path)
except (ImportError, OSError):
    pass


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
        path (str): the path to the file to be read

    Returns:
        data (Tensor)
    """
    data = torch.ops.image.read_file(path)
    return data


def write_file(filename: str, data: torch.Tensor) -> None:
    """
    Writes the contents of a uint8 tensor with one dimension to a
    file.

    Args:
        filename (str): the path to the file to be written
        data (Tensor): the contents to be written to the output file
    """
    torch.ops.image.write_file(filename, data)


def decode_png(input: torch.Tensor, mode: ImageReadMode = ImageReadMode.UNCHANGED) -> torch.Tensor:
    """
    Decodes a PNG image into a 3 dimensional RGB Tensor.
    Optionally converts the image to the desired format.
    The values of the output tensor are uint8 between 0 and 255.

    Args:
        input (Tensor[1]): a one dimensional uint8 tensor containing
            the raw bytes of the PNG image.
        mode (ImageReadMode): the read mode used for optionally
            converting the image. Default: ``ImageReadMode.UNCHANGED``.
            See `ImageReadMode` class for more information on various
            available modes.

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """
    output = torch.ops.image.decode_png(input, mode.value)
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
    output = torch.ops.image.encode_png(input, compression_level)
    return output


def write_png(input: torch.Tensor, filename: str, compression_level: int = 6):
    """
    Takes an input tensor in CHW layout (or HW in the case of grayscale images)
    and saves it in a PNG file.

    Args:
        input (Tensor[channels, image_height, image_width]): int8 image tensor of
            ``c`` channels, where ``c`` must be 1 or 3.
        filename (str): Path to save the image.
        compression_level (int): Compression factor for the resulting file, it must be a number
            between 0 and 9. Default: 6
    """
    output = encode_png(input, compression_level)
    write_file(filename, output)


def decode_jpeg(input: torch.Tensor, mode: ImageReadMode = ImageReadMode.UNCHANGED,
                device: str = 'cpu') -> torch.Tensor:
    """
    Decodes a JPEG image into a 3 dimensional RGB Tensor.
    Optionally converts the image to the desired format.
    The values of the output tensor are uint8 between 0 and 255.

    Args:
        input (Tensor[1]): a one dimensional uint8 tensor containing
            the raw bytes of the JPEG image. This tensor must be on CPU,
            regardless of the ``device`` parameter.
        mode (ImageReadMode): the read mode used for optionally
            converting the image. Default: ``ImageReadMode.UNCHANGED``.
            See ``ImageReadMode`` class for more information on various
            available modes.
        device (str or torch.device): The device on which the decoded image will
            be stored. If a cuda device is specified, the image will be decoded
            with `nvjpeg <https://developer.nvidia.com/nvjpeg>`_. This is only
            supported for CUDA version >= 10.1

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """
    device = torch.device(device)
    if device.type == 'cuda':
        output = torch.ops.image.decode_jpeg_cuda(input, mode.value, device)
    else:
        output = torch.ops.image.decode_jpeg(input, mode.value)
    return output


def encode_jpeg(input: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """
    Takes an input tensor in CHW layout and returns a buffer with the contents
    of its corresponding JPEG file.

    Args:
        input (Tensor[channels, image_height, image_width])): int8 image tensor of
            ``c`` channels, where ``c`` must be 1 or 3.
        quality (int): Quality of the resulting JPEG file, it must be a number between
            1 and 100. Default: 75

    Returns:
        output (Tensor[1]): A one dimensional int8 tensor that contains the raw bytes of the
            JPEG file.
    """
    if quality < 1 or quality > 100:
        raise ValueError('Image quality should be a positive number '
                         'between 1 and 100')

    output = torch.ops.image.encode_jpeg(input, quality)
    return output


def write_jpeg(input: torch.Tensor, filename: str, quality: int = 75):
    """
    Takes an input tensor in CHW layout and saves it in a JPEG file.

    Args:
        input (Tensor[channels, image_height, image_width]): int8 image tensor of ``c``
            channels, where ``c`` must be 1 or 3.
        filename (str): Path to save the image.
        quality (int): Quality of the resulting JPEG file, it must be a number
            between 1 and 100. Default: 75
    """
    output = encode_jpeg(input, quality)
    write_file(filename, output)


def decode_image(input: torch.Tensor, mode: ImageReadMode = ImageReadMode.UNCHANGED) -> torch.Tensor:
    """
    Detects whether an image is a JPEG or PNG and performs the appropriate
    operation to decode the image into a 3 dimensional RGB Tensor.

    Optionally converts the image to the desired format.
    The values of the output tensor are uint8 between 0 and 255.

    Args:
        input (Tensor): a one dimensional uint8 tensor containing the raw bytes of the
            PNG or JPEG image.
        mode (ImageReadMode): the read mode used for optionally converting the image.
            Default: ``ImageReadMode.UNCHANGED``.
            See ``ImageReadMode`` class for more information on various
            available modes.

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """
    output = torch.ops.image.decode_image(input, mode.value)
    return output


def read_image(path: str, mode: ImageReadMode = ImageReadMode.UNCHANGED) -> torch.Tensor:
    """
    Reads a JPEG or PNG image into a 3 dimensional RGB Tensor.
    Optionally converts the image to the desired format.
    The values of the output tensor are uint8 between 0 and 255.

    Args:
        path (str): path of the JPEG or PNG image.
        mode (ImageReadMode): the read mode used for optionally converting the image.
            Default: ``ImageReadMode.UNCHANGED``.
            See ``ImageReadMode`` class for more information on various
            available modes.

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """
    data = read_file(path)
    return decode_image(data, mode)
