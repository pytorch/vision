import torch

import os
import os.path as osp
import importlib.machinery

_HAS_IMAGE_OPT = False

try:
    lib_dir = osp.join(osp.dirname(__file__), "..")

    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES
    )

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)  # type: ignore[arg-type]
    ext_specs = extfinder.find_spec("image")
    if ext_specs is not None:
        torch.ops.load_library(ext_specs.origin)
        _HAS_IMAGE_OPT = True
except (ImportError, OSError):
    pass


def _read_file(path: str) -> torch.Tensor:
    if not os.path.isfile(path):
        raise ValueError("Expected a valid file path.")

    size = os.path.getsize(path)
    if size == 0:
        raise ValueError("Expected a non empty file.")
    data = torch.from_file(path, dtype=torch.uint8, size=size)
    return data


def decode_png(input: torch.Tensor) -> torch.Tensor:
    """
    Decodes a PNG image into a 3 dimensional RGB Tensor.
    The values of the output tensor are uint8 between 0 and 255.

    Arguments:
        input (Tensor[1]): a one dimensional uint8 tensor containing
    the raw bytes of the PNG image.

    Returns:
        output (Tensor[3, image_height, image_width])
    """
    output = torch.ops.image.decode_png(input)
    return output


def read_png(path: str) -> torch.Tensor:
    """
    Reads a PNG image into a 3 dimensional RGB Tensor.
    The values of the output tensor are uint8 between 0 and 255.

    Arguments:
        path (str): path of the PNG image.

    Returns:
        output (Tensor[3, image_height, image_width])
    """
    data = _read_file(path)
    return decode_png(data)


def encode_png(input: torch.Tensor, compression_level: int = 6) -> torch.Tensor:
    """
    Takes an input tensor in CHW layout and returns a buffer with the contents
    of its corresponding PNG file.
    Arguments:
        input (Tensor[channels, image_height, image_width]): int8 image tensor
    of `c` channels, where `c` must 3 or 1.
        compression_level (int): Compression factor for the resulting file, it
    must be a number between 0 and 9. Default: 6
    Returns
        output (Tensor[1]): A one dimensional int8 tensor that contains the raw
    bytes of the PNG file.
    """
    output = torch.ops.image.encode_png(input, compression_level)
    return output


def write_png(input: torch.Tensor, filename: str, compression_level: int = 6):
    """
    Takes an input tensor in CHW layout (or HW in the case of grayscale images)
    and saves it in a PNG file.
    Arguments:
        input (Tensor[channels, image_height, image_width]): int8 image tensor
    of `c` channels, where `c` must be 1 or 3.
        filename (str): Path to save the image.
        compression_level (int): Compression factor for the resulting file, it
    must be a number between 0 and 9. Default: 6
    """
    torch.ops.image.write_png(input, filename, compression_level)


def decode_jpeg(input: torch.Tensor) -> torch.Tensor:
    """
    Decodes a JPEG image into a 3 dimensional RGB Tensor.
    The values of the output tensor are uint8 between 0 and 255.
    Arguments:
        input (Tensor[1]): a one dimensional uint8 tensor containing
    the raw bytes of the JPEG image.
    Returns:
        output (Tensor[3, image_height, image_width])
    """
    output = torch.ops.image.decode_jpeg(input)
    return output


def read_jpeg(path: str) -> torch.Tensor:
    """
    Reads a JPEG image into a 3 dimensional RGB Tensor.
    The values of the output tensor are uint8 between 0 and 255.
    Arguments:
        path (str): path of the JPEG image.
    Returns:
        output (Tensor[3, image_height, image_width])
    """
    data = _read_file(path)
    return decode_jpeg(data)


def encode_jpeg(input: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """
    Takes an input tensor in CHW layout and returns a buffer with the contents
    of its corresponding JPEG file.
    Arguments:
        input (Tensor[channels, image_height, image_width]): int8 image tensor
    of `c` channels, where `c` must be 1 or 3.
        quality (int): Quality of the resulting JPEG file, it must be a number
    between 1 and 100. Default: 75
    Returns
        output (Tensor[1]): A one dimensional int8 tensor that contains the raw
    bytes of the JPEG file.
    """
    if quality < 1 or quality > 100:
        raise ValueError('Image quality should be a positive number '
                         'between 1 and 100')

    output = torch.ops.image.encode_jpeg(input, quality)
    return output


def write_jpeg(input: torch.Tensor, filename: str, quality: int = 75):
    """
    Takes an input tensor in CHW layout and saves it in a JPEG file.
    Arguments:
        input (Tensor[channels, image_height, image_width]): int8 image tensor
    of `c` channels, where `c` must be 1 or 3.
        filename (str): Path to save the image.
        quality (int): Quality of the resulting JPEG file, it must be a number
    between 1 and 100. Default: 75
    """
    if quality < 1 or quality > 100:
        raise ValueError('Image quality should be a positive number '
                         'between 1 and 100')

    torch.ops.image.write_jpeg(input, filename, quality)


def decode_image(input: torch.Tensor) -> torch.Tensor:
    """
    Detects whether an image is a JPEG or PNG and performs the appropriate
    operation to decode the image into a 3 dimensional RGB Tensor.

    The values of the output tensor are uint8 between 0 and 255.

    Arguments:
        input (Tensor): a one dimensional uint8 tensor containing
        the raw bytes of the PNG or JPEG image.
    Returns:
        output (Tensor[3, image_height, image_width])
    """
    output = torch.ops.image.decode_image(input)
    return output


def read_image(path: str) -> torch.Tensor:
    """
    Reads a JPEG or PNG image into a 3 dimensional RGB Tensor.
    The values of the output tensor are uint8 between 0 and 255.
    Arguments:
        path (str): path of the JPEG or PNG image.
    Returns:
        output (Tensor[3, image_height, image_width])
    """
    data = _read_file(path)
    return decode_image(data)
