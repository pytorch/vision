import torch
import torchvision.transforms.functional as F


def vflip(img_tensor):
    """Vertically flip the given the Image Tensor.

    Args:
        img_tensor (Tensor): Image Tensor to be flipped in the form [C, H, W].

    Returns:
        Tensor:  Vertically flipped image Tensor.
    """
    if not F._is_tensor_image(img_tensor):
        raise TypeError('tensor is not a torch image.')

    return img_tensor.flip(-2)


def hflip(img_tensor):
    """Horizontally flip the given the Image Tensor.

    Args:
        img_tensor (Tensor): Image Tensor to be flipped in the form [C, H, W].

    Returns:
        Tensor:  Horizontally flipped image Tensor.
    """

    if not F._is_tensor_image(img_tensor):
        raise TypeError('tensor is not a torch image.')

    return img_tensor.flip(-1)
