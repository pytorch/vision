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


def crop(img, top, left, height, width):
    """Crop the given Image Tensor.
    Args:
        img (Tensor): Image to be cropped in the form [C, H, W]. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
    Returns:
        Tensor: Cropped image.
    """
    if not F._is_tensor_image(img):
        raise TypeError('tensor is not a torch image.')

    return img[..., top:top + height, left:left + width]

def to_grayscale(img, num_output_channels = 3):
    """Convert the given RGB Image Tensor to Grayscale.

    Args
        img (Tensor): Image to be converted to Grayscale in the form [C, H, W].
        num_output_channels (int): denotes the number of channels to return after conversion
    Returns:
    Tensor: Grayscale image.
    """
    if not F._is_tensor_image(img):
        raise TypeError('tensor is not a torch image.')

    if img.size()[0] != 3:
        raise TypeError('Input Image does not contain 3 Channels')

    return (0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]).to(img.dtype)
