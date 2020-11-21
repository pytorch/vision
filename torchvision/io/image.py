import torch

import os
import os.path as osp
import importlib.machinery

from enum import Enum

_HAS_IMAGE_OPT = False

try:
    lib_dir = osp.abspath(osp.join(osp.dirname(__file__), ".."))

    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES
    )

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)  # type: ignore[arg-type]
    ext_specs = extfinder.find_spec("image")

    if os.name == 'nt':
        # Load the image extension using LoadLibraryExW
        import ctypes
        import sys

        kernel32 = ctypes.WinDLL('kernel32.dll', use_last_error=True)
        with_load_library_flags = hasattr(kernel32, 'AddDllDirectory')
        prev_error_mode = kernel32.SetErrorMode(0x0001)

        kernel32.LoadLibraryW.restype = ctypes.c_void_p
        if with_load_library_flags:
            kernel32.LoadLibraryExW.restype = ctypes.c_void_p

        if ext_specs is not None:
            res = kernel32.LoadLibraryExW(ext_specs.origin, None, 0x00001100)
            if res is None:
                err = ctypes.WinError(ctypes.get_last_error())
                err.strerror += (f' Error loading "{ext_specs.origin}" or any or '
                                 'its dependencies.')
                raise err

        kernel32.SetErrorMode(prev_error_mode)

    if ext_specs is not None:
        torch.ops.load_library(ext_specs.origin)
        _HAS_IMAGE_OPT = True
except (ImportError, OSError):
    pass


class ImageReadMode(Enum):
    UNCHANGED = 0
    GRAY = 1
    GRAY_ALPHA = 2
    RGB = 3
    RGB_ALPHA = 4


def read_file(path: str) -> torch.Tensor:
    """
    Reads and outputs the bytes contents of a file as a uint8 Tensor
    with one dimension.

    Arguments:
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

    Arguments:
        filename (str): the path to the file to be written
        data (Tensor): the contents to be written to the output file
    """
    torch.ops.image.write_file(filename, data)


def decode_png(input: torch.Tensor, mode: ImageReadMode = ImageReadMode.UNCHANGED) -> torch.Tensor:
    """
    Decodes a PNG image into a 3 dimensional RGB Tensor.
    Optionally converts the image to the desired format.
    The values of the output tensor are uint8 between 0 and 255.

    Arguments:
        input (Tensor[1]): a one dimensional uint8 tensor containing
    the raw bytes of the PNG image.
        mode (ImageReadMode): the read mode used for optionally
    converting the image. Use `ImageReadMode.UNCHANGED` for loading
    the image as-is, `ImageReadMode.GRAY` for converting to grayscale,
    `ImageReadMode.GRAY_ALPHA` for grayscale with transparency,
    `ImageReadMode.RGB` for RGB and `ImageReadMode.RGB_ALPHA` for
     RGB with transparency. Default: `ImageReadMode.UNCHANGED`

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """
    output = torch.ops.image.decode_png(input, mode.value)
    return output


def encode_png(input: torch.Tensor, compression_level: int = 6) -> torch.Tensor:
    """
    Takes an input tensor in CHW layout and returns a buffer with the contents
    of its corresponding PNG file.

    Parameters
    ----------
    input: Tensor[channels, image_height, image_width]
        int8 image tensor of `c` channels, where `c` must 3 or 1.
    compression_level: int
        Compression factor for the resulting file, it must be a number
        between 0 and 9. Default: 6

    Returns
    -------
    output: Tensor[1]
        A one dimensional int8 tensor that contains the raw bytes of the
        PNG file.
    """
    output = torch.ops.image.encode_png(input, compression_level)
    return output


def write_png(input: torch.Tensor, filename: str, compression_level: int = 6):
    """
    Takes an input tensor in CHW layout (or HW in the case of grayscale images)
    and saves it in a PNG file.

    Parameters
    ----------
    input: Tensor[channels, image_height, image_width]
        int8 image tensor of `c` channels, where `c` must be 1 or 3.
    filename: str
        Path to save the image.
    compression_level: int
        Compression factor for the resulting file, it must be a number
        between 0 and 9. Default: 6
    """
    output = encode_png(input, compression_level)
    write_file(filename, output)


def decode_jpeg(input: torch.Tensor, mode: ImageReadMode = ImageReadMode.UNCHANGED) -> torch.Tensor:
    """
    Decodes a JPEG image into a 3 dimensional RGB Tensor.
    Optionally converts the image to the desired format.
    The values of the output tensor are uint8 between 0 and 255.

    Arguments:
        input (Tensor[1]): a one dimensional uint8 tensor containing
    the raw bytes of the JPEG image.
        mode (ImageReadMode): the read mode used for optionally
    converting the image. Use `ImageReadMode.UNCHANGED` for loading
    the image as-is, `ImageReadMode.GRAY` for converting to grayscale
    and `ImageReadMode.RGB` for RGB. Default: `ImageReadMode.UNCHANGED`

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """
    output = torch.ops.image.decode_jpeg(input, mode.value)
    return output


def encode_jpeg(input: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """
    Takes an input tensor in CHW layout and returns a buffer with the contents
    of its corresponding JPEG file.

    Parameters
    ----------
    input: Tensor[channels, image_height, image_width])
        int8 image tensor of `c` channels, where `c` must be 1 or 3.
    quality: int
        Quality of the resulting JPEG file, it must be a number between
        1 and 100. Default: 75

    Returns
    -------
    output: Tensor[1]
        A one dimensional int8 tensor that contains the raw bytes of the
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

    Parameters
    ----------
    input: Tensor[channels, image_height, image_width]
        int8 image tensor of `c` channels, where `c` must be 1 or 3.
    filename: str
        Path to save the image.
    quality: int
        Quality of the resulting JPEG file, it must be a number
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

    Parameters
    ----------
    input: Tensor
        a one dimensional uint8 tensor containing the raw bytes of the
        PNG or JPEG image.
    mode: ImageReadMode
        the read mode used for optionally converting the image. JPEG
        and PNG images have different permitted values. The default
        value is `ImageReadMode.UNCHANGED` and it keeps the image as-is.
        See `decode_jpeg()` and `decode_png()` for more information.
        Default: `ImageReadMode.UNCHANGED`

    Returns
    -------
    output: Tensor[image_channels, image_height, image_width]
    """
    output = torch.ops.image.decode_image(input, mode.value)
    return output


def read_image(path: str, mode: ImageReadMode = ImageReadMode.UNCHANGED) -> torch.Tensor:
    """
    Reads a JPEG or PNG image into a 3 dimensional RGB Tensor.
    Optionally converts the image to the desired format.
    The values of the output tensor are uint8 between 0 and 255.

    Parameters
    ----------
    path: str
        path of the JPEG or PNG image.
    mode: ImageReadMode
        the read mode used for optionally converting the image. JPEG
        and PNG images have different permitted values. The default
        value is `ImageReadMode.UNCHANGED` and it keeps the image as-is.
        See `decode_jpeg()` and `decode_png()` for more information.
        Default: `ImageReadMode.UNCHANGED`

    Returns
    -------
    output: Tensor[image_channels, image_height, image_width]
    """
    data = read_file(path)
    return decode_image(data, mode)
