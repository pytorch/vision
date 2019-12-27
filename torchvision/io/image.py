import torch
from torch import nn, Tensor
import os


def decode_png(input):
    # type: (Tensor) -> Tensor
    """
    Decodes a PNG image into a 3 dimensional RGB Tensor.
    The values of the output tensor are uint8 between 0 and 255.

    Arguments:
        input (Tensor[1]): a one dimensional int8 tensor containing
    the raw bytes of the PNG image.

    Returns:
        output (Tensor[image_width, image_height, 3])
    """
    if not isinstance(input, torch.Tensor) or len(input) == 0:
        raise ValueError("Expected a non empty 1-dimensional tensor.")

    if not input.dtype == torch.uint8:
        raise ValueError("Expected a torch.uint8 tensor.")
    output = torch.ops.torchvision.decode_png(input)
    return output


def read_png(path):
    # type: (str) -> Tensor
    """
    Reads a PNG image into a 3 dimensional RGB Tensor.
    The values of the output tensor are uint8 between 0 and 255.

    Arguments:
        path (str): path of the PNG image.

    Returns:
        output (Tensor[image_width, image_height, 3])
    """
    if not os.path.isfile(path):
        raise ValueError("Excepted a valid file path.")

    size = os.path.getsize(path)
    if size == 0:
        raise ValueError("Excepted a non empty file.")
    data = torch.from_file(path, dtype=torch.uint8, size=size)
    return decode_png(data)

def decode_jpeg(input):
    # type: (Tensor) -> Tensor
    """
    Decodes a JPEG image into a 3 dimensional RGB Tensor.
    The values of the output tensor are uint8 between 0 and 255.

    Arguments:
        input (Tensor[1]): a one dimensional int8 tensor containing
    the raw bytes of the JPEG image.

    Returns:
        output (Tensor[image_width, image_height, 3])
    """
    if not isinstance(input, torch.Tensor) or len(input) == 0:
        raise ValueError("Expected a non empty 1-dimensional tensor.")

    if not input.dtype == torch.uint8:
        raise ValueError("Expected a torch.uint8 tensor.")
    output = torch.ops.torchvision.decode_jpeg(input)
    return output


def read_jpeg(path):
    # type: (str) -> Tensor
    """
    Reads a JPEG image into a 3 dimensional RGB Tensor.
    The values of the output tensor are uint8 between 0 and 255.

    Arguments:
        path (str): path of the JPEG image.

    Returns:
        output (Tensor[image_width, image_height, 3])
    """
    if not os.path.isfile(path):
        raise ValueError("Excepted a valid file path.")

    size = os.path.getsize(path)
    if size == 0:
        raise ValueError("Excepted a non empty file.")
    data = torch.from_file(path, dtype=torch.uint8, size=size)
    return decode_jpeg(data)
