import torch
import torchvision.transforms.functional as F
from torch import Tensor
from torch.jit.annotations import Optional, List, BroadcastingList2, Tuple


def _is_tensor_a_torch_image(input):
    return len(input.shape) == 3


def vflip(img):
    # type: (Tensor) -> Tensor
    """Vertically flip the given the Image Tensor.

    Args:
        img (Tensor): Image Tensor to be flipped in the form [C, H, W].

    Returns:
        Tensor:  Vertically flipped image Tensor.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return img.flip(-2)


def hflip(img):
    # type: (Tensor) -> Tensor
    """Horizontally flip the given the Image Tensor.

    Args:
        img (Tensor): Image Tensor to be flipped in the form [C, H, W].

    Returns:
        Tensor:  Horizontally flipped image Tensor.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return img.flip(-1)


def crop(img, top, left, height, width):
    # type: (Tensor, int, int, int, int) -> Tensor
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
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return img[..., top:top + height, left:left + width]


def rgb_to_grayscale(img):
    # type: (Tensor) -> Tensor
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
    # type: (Tensor, float) -> Tensor
    """Adjust brightness of an RGB image.

    Args:
        img (Tensor): Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        Tensor: Brightness adjusted image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return _blend(img, torch.zeros_like(img), brightness_factor)


def adjust_contrast(img, contrast_factor):
    # type: (Tensor, float) -> Tensor
    """Adjust contrast of an RGB image.

    Args:
        img (Tensor): Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        Tensor: Contrast adjusted image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    mean = torch.mean(rgb_to_grayscale(img).to(torch.float))

    return _blend(img, mean, contrast_factor)


def adjust_saturation(img, saturation_factor):
    # type: (Tensor, float) -> Tensor
    """Adjust color saturation of an RGB image.

    Args:
        img (Tensor): Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        Tensor: Saturation adjusted image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return _blend(img, rgb_to_grayscale(img), saturation_factor)


def center_crop(img, output_size):
    # type: (Tensor, BroadcastingList2[int]) -> Tensor
    """Crop the Image Tensor and resize it to desired size.

    Args:
        img (Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions

    Returns:
            Tensor: Cropped image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    _, image_width, image_height = img.size()
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))

    return crop(img, crop_top, crop_left, crop_height, crop_width)


def five_crop(img, size):
    # type: (Tensor, BroadcastingList2[int]) -> List[Tensor]
    """Crop the given Image Tensor into four corners and the central crop.
    .. Note::
        This transform returns a List of Tensors and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.

    Returns:
       List: List (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    _, image_width, image_height = img.size()
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    tl = crop(img, 0, 0, crop_width, crop_height)
    tr = crop(img, image_width - crop_width, 0, image_width, crop_height)
    bl = crop(img, 0, image_height - crop_height, crop_width, image_height)
    br = crop(img, image_width - crop_width, image_height - crop_height, image_width, image_height)
    center = center_crop(img, (crop_height, crop_width))

    return [tl, tr, bl, br, center]


def ten_crop(img, size, vertical_flip=False):
    # type: (Tensor, BroadcastingList2[int], bool) -> List[Tensor]
    """Crop the given Image Tensor into four corners and the central crop plus the
        flipped version of these (horizontal flipping is used by default).
    .. Note::
        This transform returns a List of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
       size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
       vertical_flip (bool): Use vertical flipping instead of horizontal

    Returns:
       List: List (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip)
                Corresponding top left, top right, bottom left, bottom right and center crop
                and same for the flipped image's tensor.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
    first_five = five_crop(img, size)

    if vertical_flip:
        img = vflip(img)
    else:
        img = hflip(img)

    second_five = five_crop(img, size)

    return first_five + second_five


def _blend(img1, img2, ratio):
    # type: (Tensor, Tensor, float) -> Tensor
    bound = 1 if img1.dtype in [torch.half, torch.float32, torch.float64] else 255
    return (ratio * img1 + (1 - ratio) * img2).clamp(0, bound).to(img1.dtype)
