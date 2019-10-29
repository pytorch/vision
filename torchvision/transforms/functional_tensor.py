import numbers
import torch
from PIL import Image


def _is_tensor_image(img):
    if not isinstance(img, torch.Tensor):
        raise TypeError("expected torch.Tensor, got {}".format(type(img)))
    return img.ndimension() in [3, 4]


_PIL_TO_TORCH_INTERP_MODE = {
    Image.NEAREST: "nearest",
    Image.BILINEAR: "bilinear"
}


def vflip(img_tensor):
    """Vertically flip the given the Image Tensor.

    Args:
        img_tensor (Tensor): Image Tensor to be flipped in the form [C, H, W].

    Returns:
        Tensor:  Vertically flipped image Tensor.
    """
    if not _is_tensor_image(img_tensor):
        raise TypeError('tensor is not a torch image.')

    return img_tensor.flip(-2)


def hflip(img_tensor):
    """Horizontally flip the given the Image Tensor.

    Args:
        img_tensor (Tensor): Image Tensor to be flipped in the form [C, H, W].

    Returns:
        Tensor:  Horizontally flipped image Tensor.
    """

    if not _is_tensor_image(img_tensor):
        raise TypeError('tensor is not a torch image.')

    return img_tensor.flip(-1)


def resize(img, size, interpolation=None):
    r"""Resize the input Image to the given size.

    Args:
        img (torch.Tensor): Image to be resized. Can be 3d or 4d (for batches of images or videos)
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        torch.Tensor: Resized image.
    """

    if interpolation is None:
        interpolation = Image.BILINEAR

    if not _is_tensor_image(img):
        raise TypeError('tensor is not a torch image.')
    if not (isinstance(size, int) or (isinstance(size, (tuple, list)) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        h, w = img.shape[-2:]
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        size = (oh, ow)

    interpolation_mode = _PIL_TO_TORCH_INTERP_MODE[interpolation]

    # interpolate expects batch of images for now, so should adapt input to 4D if necessary
    should_squeeze = False
    if img.ndim == 3:
        should_squeeze = True
        img = image[None]
    res = torch.nn.functional.interpolate(
        img, size=size, mode=interpolation_mode, align_corners=False
    )
    if should_squeeze:
        res = res[0]

    return res


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
    if not _is_tensor_image(img):
        raise TypeError('tensor is not a torch image.')

    return img[..., top:top + height, left:left + width]


def center_crop(img, output_size):
    """Crop the given Image Tensor and resize it to desired size.

        Args:
            img (Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
        Returns:
            Tensor: Cropped image.
        """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    image_height, image_width = img.shape[-2:]
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return crop(img, crop_top, crop_left, crop_height, crop_width)


def resized_crop(img, top, left, height, width, size, interpolation=Image.BILINEAR):
    """Crop the given Image Tensor and resize it to desired size.

    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.

    Args:
        img (Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        size (sequence or int): Desired output size. Same semantics as ``resize``.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``.
    Returns:
        Tensor: Cropped image.
    """
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation)
    return img


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
    if not _is_tensor_image(img):
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
    if not _is_tensor_image(img):
        raise TypeError('tensor is not a torch image.')

    mean = torch.mean(_rgb_to_grayscale(img).to(torch.float))

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
    if not _is_tensor_image(img):
        raise TypeError('tensor is not a torch image.')

    return _blend(img, _rgb_to_grayscale(img), saturation_factor)


def _blend(img1, img2, ratio):
    bound = 1 if img1.dtype.is_floating_point else 255
    return (ratio * img1 + (1 - ratio) * img2).clamp(0, bound).to(img1.dtype)


def _rgb_to_grayscale(img):
    # ITU-R 601-2 luma transform, as used in PIL.
    return (0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]).to(img.dtype)
