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


def rgb_to_grayscale(img):
    """Convert the given RGB Image Tensor to Grayscale.
    For RGB to Grayscale conversion, ITU-R 601-2 luma transform is performed which
    is L = R * 0.2989 + G * 0.5870 + B * 0.1140

    Args:
        img (Tensor): Image to be converted to Grayscale in the form [C, H, W].

    Returns:
        Tensor: Grayscale image.

    """
    if img.shape[0] != 3:
        raise TypeError('Input Image does not contain 3 Channels')

    return (0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]).to(img.dtype)


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an RGB image.

    Args:
        img (Tensor): Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        Tensor: Brightness adjusted image.
    """
    if not F._is_tensor_image(img):
        raise TypeError('tensor is not a torch image.')

    return _blend(img, 0, brightness_factor)


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an RGB image.

    Args:
        img (Tensor): Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        Tensor: Contrast adjusted image.
    """
    if not F._is_tensor_image(img):
        raise TypeError('tensor is not a torch image.')

    mean = torch.mean(rgb_to_grayscale(img).to(torch.float))

    return _blend(img, mean, contrast_factor)


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an RGB image.

    Args:
        img (Tensor): Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        Tensor: Saturation adjusted image.
    """
    if not F._is_tensor_image(img):
        raise TypeError('tensor is not a torch image.')

    return _blend(img, rgb_to_grayscale(img), saturation_factor)


def _blend(img1, img2, ratio):
    bound = 1 if img1.dtype.is_floating_point else 255
    return (ratio * img1 + (1 - ratio) * img2).clamp(0, bound).to(img1.dtype)
