import warnings
from typing import Optional, Dict, Tuple

import torch
from torch import Tensor
from torch.nn.functional import grid_sample, conv2d, interpolate, pad as torch_pad
from torch.jit.annotations import List, BroadcastingList2


def _is_tensor_a_torch_image(x: Tensor) -> bool:
    return x.ndim >= 2


def _get_image_size(img: Tensor) -> List[int]:
    """Returns (w, h) of tensor image"""
    if _is_tensor_a_torch_image(img):
        return [img.shape[-1], img.shape[-2]]
    raise TypeError("Unexpected input type")


def _get_image_num_channels(img: Tensor) -> int:
    if img.ndim == 2:
        return 1
    elif img.ndim > 2:
        return img.shape[-3]

    raise TypeError("Input ndim should be 2 or more. Got {}".format(img.ndim))


def _max_value(dtype: torch.dtype) -> float:
    # TODO: replace this method with torch.iinfo when it gets torchscript support.
    # https://github.com/pytorch/pytorch/issues/41492

    a = torch.tensor(2, dtype=dtype)
    signed = 1 if torch.tensor(0, dtype=dtype).is_signed() else 0
    bits = 1
    max_value = torch.tensor(-signed, dtype=torch.long)
    while True:
        next_value = a.pow(bits - signed).sub(1)
        if next_value > max_value:
            max_value = next_value
            bits *= 2
        else:
            return max_value.item()
    return max_value.item()


def convert_image_dtype(image: torch.Tensor, dtype: torch.dtype = torch.float) -> torch.Tensor:
    """PRIVATE METHOD. Convert a tensor image to the given ``dtype`` and scale the values accordingly

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        image (torch.Tensor): Image to be converted
        dtype (torch.dtype): Desired data type of the output

    Returns:
        (torch.Tensor): Converted image

    .. note::

        When converting from a smaller to a larger integer ``dtype`` the maximum values are **not** mapped exactly.
        If converted back and forth, this mismatch has no effect.

    Raises:
        RuntimeError: When trying to cast :class:`torch.float32` to :class:`torch.int32` or :class:`torch.int64` as
            well as for trying to cast :class:`torch.float64` to :class:`torch.int64`. These conversions might lead to
            overflow errors since the floating point ``dtype`` cannot store consecutive integers over the whole range
            of the integer ``dtype``.
    """
    if image.dtype == dtype:
        return image

    # TODO: replace with image.dtype.is_floating_point when torchscript supports it
    if torch.empty(0, dtype=image.dtype).is_floating_point():

        # TODO: replace with dtype.is_floating_point when torchscript supports it
        if torch.tensor(0, dtype=dtype).is_floating_point():
            return image.to(dtype)

        # float to int
        if (image.dtype == torch.float32 and dtype in (torch.int32, torch.int64)) or (
            image.dtype == torch.float64 and dtype == torch.int64
        ):
            msg = f"The cast from {image.dtype} to {dtype} cannot be performed safely."
            raise RuntimeError(msg)

        # https://github.com/pytorch/vision/pull/2078#issuecomment-612045321
        # For data in the range 0-1, (float * 255).to(uint) is only 255
        # when float is exactly 1.0.
        # `max + 1 - epsilon` provides more evenly distributed mapping of
        # ranges of floats to ints.
        eps = 1e-3
        max_val = _max_value(dtype)
        result = image.mul(max_val + 1.0 - eps)
        return result.to(dtype)
    else:
        input_max = _max_value(image.dtype)
        output_max = _max_value(dtype)

        # int to float
        # TODO: replace with dtype.is_floating_point when torchscript supports it
        if torch.tensor(0, dtype=dtype).is_floating_point():
            image = image.to(dtype)
            return image / input_max

        # int to int
        if input_max > output_max:
            # factor should be forced to int for torch jit script
            # otherwise factor is a float and image // factor can produce different results
            factor = int((input_max + 1) // (output_max + 1))
            image = image // factor
            return image.to(dtype)
        else:
            # factor should be forced to int for torch jit script
            # otherwise factor is a float and image * factor can produce different results
            factor = int((output_max + 1) // (input_max + 1))
            image = image.to(dtype)
            return image * factor


def vflip(img: Tensor) -> Tensor:
    """PRIVATE METHOD. Vertically flip the given the Image Tensor.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (Tensor): Image Tensor to be flipped in the form [..., C, H, W].

    Returns:
        Tensor:  Vertically flipped image Tensor.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return img.flip(-2)


def hflip(img: Tensor) -> Tensor:
    """PRIVATE METHOD. Horizontally flip the given the Image Tensor.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (Tensor): Image Tensor to be flipped in the form [..., C, H, W].

    Returns:
        Tensor:  Horizontally flipped image Tensor.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return img.flip(-1)


def crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    """PRIVATE METHOD. Crop the given Image Tensor.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (Tensor): Image to be cropped in the form [..., H, W]. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        Tensor: Cropped image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError("tensor is not a torch image.")

    return img[..., top:top + height, left:left + width]


def rgb_to_grayscale(img: Tensor, num_output_channels: int = 1) -> Tensor:
    """PRIVATE METHOD. Convert the given RGB Image Tensor to Grayscale.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    For RGB to Grayscale conversion, ITU-R 601-2 luma transform is performed which
    is L = R * 0.2989 + G * 0.5870 + B * 0.1140

    Args:
        img (Tensor): Image to be converted to Grayscale in the form [C, H, W].
        num_output_channels (int): number of channels of the output image. Value can be 1 or 3. Default, 1.

    Returns:
        Tensor: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel

            if num_output_channels = 3 : returned image is 3 channel with r = g = b

    """
    if img.ndim < 3:
        raise TypeError("Input image tensor should have at least 3 dimensions, but found {}".format(img.ndim))
    c = img.shape[-3]
    if c != 3:
        raise TypeError("Input image tensor should 3 channels, but found {}".format(c))

    if num_output_channels not in (1, 3):
        raise ValueError('num_output_channels should be either 1 or 3')

    r, g, b = img.unbind(dim=-3)
    # This implementation closely follows the TF one:
    # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)

    if num_output_channels == 3:
        return l_img.expand(img.shape)

    return l_img


def adjust_brightness(img: Tensor, brightness_factor: float) -> Tensor:
    """PRIVATE METHOD. Adjust brightness of an RGB image.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (Tensor): Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        Tensor: Brightness adjusted image.
    """
    if brightness_factor < 0:
        raise ValueError('brightness_factor ({}) is not non-negative.'.format(brightness_factor))

    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return _blend(img, torch.zeros_like(img), brightness_factor)


def adjust_contrast(img: Tensor, contrast_factor: float) -> Tensor:
    """PRIVATE METHOD. Adjust contrast of an RGB image.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (Tensor): Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        Tensor: Contrast adjusted image.
    """
    if contrast_factor < 0:
        raise ValueError('contrast_factor ({}) is not non-negative.'.format(contrast_factor))

    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    mean = torch.mean(rgb_to_grayscale(img).to(dtype), dim=(-3, -2, -1), keepdim=True)

    return _blend(img, mean, contrast_factor)


def adjust_hue(img: Tensor, hue_factor: float) -> Tensor:
    """PRIVATE METHOD. Adjust hue of an image.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        img (Tensor): Image to be adjusted. Image type is either uint8 or float.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
         Tensor: Hue adjusted image.
    """
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))

    if not (isinstance(img, torch.Tensor) and _is_tensor_a_torch_image(img)):
        raise TypeError('Input img should be Tensor image')

    orig_dtype = img.dtype
    if img.dtype == torch.uint8:
        img = img.to(dtype=torch.float32) / 255.0

    img = _rgb2hsv(img)
    h, s, v = img.unbind(dim=-3)
    h = (h + hue_factor) % 1.0
    img = torch.stack((h, s, v), dim=-3)
    img_hue_adj = _hsv2rgb(img)

    if orig_dtype == torch.uint8:
        img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)

    return img_hue_adj


def adjust_saturation(img: Tensor, saturation_factor: float) -> Tensor:
    """PRIVATE METHOD. Adjust color saturation of an RGB image.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (Tensor): Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. Can be any
            non negative number. 0 gives a black and white image, 1 gives the
            original image while 2 enhances the saturation by a factor of 2.

    Returns:
        Tensor: Saturation adjusted image.
    """
    if saturation_factor < 0:
        raise ValueError('saturation_factor ({}) is not non-negative.'.format(saturation_factor))

    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return _blend(img, rgb_to_grayscale(img), saturation_factor)


def adjust_gamma(img: Tensor, gamma: float, gain: float = 1) -> Tensor:
    r"""PRIVATE METHOD. Adjust gamma of an RGB image.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:

    .. math::
        `I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}`

    See `Gamma Correction`_ for more details.

    .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction

    Args:
        img (Tensor): Tensor of RBG values to be adjusted.
        gamma (float): Non negative real number, same as :math:`\gamma` in the equation.
            gamma larger than 1 make the shadows darker,
            while gamma smaller than 1 make dark regions lighter.
        gain (float): The constant multiplier.
    """

    if not isinstance(img, torch.Tensor):
        raise TypeError('Input img should be a Tensor.')

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    result = img
    dtype = img.dtype
    if not torch.is_floating_point(img):
        result = convert_image_dtype(result, torch.float32)

    result = (gain * result ** gamma).clamp(0, 1)

    result = convert_image_dtype(result, dtype)
    result = result.to(dtype)
    return result


def center_crop(img: Tensor, output_size: BroadcastingList2[int]) -> Tensor:
    """DEPRECATED. Crop the Image Tensor and resize it to desired size.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    .. warning::

        This method is deprecated and will be removed in future releases.
        Please, use ``F.center_crop`` instead.

    Args:
        img (Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions

    Returns:
            Tensor: Cropped image.
    """
    warnings.warn(
        "This method is deprecated and will be removed in future releases. "
        "Please, use ``F.center_crop`` instead."
    )

    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    _, image_width, image_height = img.size()
    crop_height, crop_width = output_size
    # crop_top = int(round((image_height - crop_height) / 2.))
    # Result can be different between python func and scripted func
    # Temporary workaround:
    crop_top = int((image_height - crop_height + 1) * 0.5)
    # crop_left = int(round((image_width - crop_width) / 2.))
    # Result can be different between python func and scripted func
    # Temporary workaround:
    crop_left = int((image_width - crop_width + 1) * 0.5)

    return crop(img, crop_top, crop_left, crop_height, crop_width)


def five_crop(img: Tensor, size: BroadcastingList2[int]) -> List[Tensor]:
    """DEPRECATED. Crop the given Image Tensor into four corners and the central crop.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    .. warning::

        This method is deprecated and will be removed in future releases.
        Please, use ``F.five_crop`` instead.

    .. Note::

        This transform returns a List of Tensors and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
        img (Tensor): Image to be cropped.
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.

    Returns:
       List: List (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    warnings.warn(
        "This method is deprecated and will be removed in future releases. "
        "Please, use ``F.five_crop`` instead."
    )

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


def ten_crop(img: Tensor, size: BroadcastingList2[int], vertical_flip: bool = False) -> List[Tensor]:
    """DEPRECATED. Crop the given Image Tensor into four corners and the central crop plus the
        flipped version of these (horizontal flipping is used by default).

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    .. warning::

        This method is deprecated and will be removed in future releases.
        Please, use ``F.ten_crop`` instead.

    .. Note::

        This transform returns a List of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
        img (Tensor): Image to be cropped.
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        vertical_flip (bool): Use vertical flipping instead of horizontal

    Returns:
       List: List (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip)
                Corresponding top left, top right, bottom left, bottom right and center crop
                and same for the flipped image's tensor.
    """
    warnings.warn(
        "This method is deprecated and will be removed in future releases. "
        "Please, use ``F.ten_crop`` instead."
    )

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


def _blend(img1: Tensor, img2: Tensor, ratio: float) -> Tensor:
    bound = 1.0 if img1.is_floating_point() else 255.0
    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)


def _rgb2hsv(img):
    r, g, b = img.unbind(dim=-3)

    # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
    # src/libImaging/Convert.c#L330
    maxc = torch.max(img, dim=-3).values
    minc = torch.min(img, dim=-3).values

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occuring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = (hr + hg + hb)
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc), dim=-3)


def _hsv2rgb(img):
    h, s, v = img.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)

    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)

    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)

    return torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=img.dtype), a4)


def _pad_symmetric(img: Tensor, padding: List[int]) -> Tensor:
    # padding is left, right, top, bottom

    # crop if needed
    if padding[0] < 0 or padding[1] < 0 or padding[2] < 0 or padding[3] < 0:
        crop_left, crop_right, crop_top, crop_bottom = [-min(x, 0) for x in padding]
        img = img[..., crop_top:img.shape[-2] - crop_bottom, crop_left:img.shape[-1] - crop_right]
        padding = [max(x, 0) for x in padding]

    in_sizes = img.size()

    x_indices = [i for i in range(in_sizes[-1])]  # [0, 1, 2, 3, ...]
    left_indices = [i for i in range(padding[0] - 1, -1, -1)]  # e.g. [3, 2, 1, 0]
    right_indices = [-(i + 1) for i in range(padding[1])]  # e.g. [-1, -2, -3]
    x_indices = torch.tensor(left_indices + x_indices + right_indices)

    y_indices = [i for i in range(in_sizes[-2])]
    top_indices = [i for i in range(padding[2] - 1, -1, -1)]
    bottom_indices = [-(i + 1) for i in range(padding[3])]
    y_indices = torch.tensor(top_indices + y_indices + bottom_indices)

    ndim = img.ndim
    if ndim == 3:
        return img[:, y_indices[:, None], x_indices[None, :]]
    elif ndim == 4:
        return img[:, :, y_indices[:, None], x_indices[None, :]]
    else:
        raise RuntimeError("Symmetric padding of N-D tensors are not supported yet")


def pad(img: Tensor, padding: List[int], fill: int = 0, padding_mode: str = "constant") -> Tensor:
    r"""PRIVATE METHOD. Pad the given Tensor Image on all sides with specified padding mode and fill value.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (Tensor): Image to be padded.
        padding (int or tuple or list): Padding on each border. If a single int is provided this
            is used to pad all borders. If a tuple or list of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple or list of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively. In torchscript mode padding as single int is not supported, use a tuple or
            list of length 1: ``[padding, ]``.
        fill (int): Pixel fill value for constant fill. Default is 0.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Should be: constant, edge or reflect. Default is constant.
            Mode symmetric is not yet supported for Tensor inputs.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value on the edge of the image

            - reflect: pads with reflection of image (without repeating the last value on the edge)

                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image (repeating the last value on the edge)

                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        Tensor: Padded image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError("tensor is not a torch image.")

    if not isinstance(padding, (int, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (int, float)):
        raise TypeError("Got inappropriate fill arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, tuple):
        padding = list(padding)

    if isinstance(padding, list) and len(padding) not in [1, 2, 4]:
        raise ValueError("Padding must be an int or a 1, 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
        raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

    if isinstance(padding, int):
        if torch.jit.is_scripting():
            # This maybe unreachable
            raise ValueError("padding can't be an int while torchscripting, set it as a list [value, ]")
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif len(padding) == 1:
        pad_left = pad_right = pad_top = pad_bottom = padding[0]
    elif len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    else:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    p = [pad_left, pad_right, pad_top, pad_bottom]

    if padding_mode == "edge":
        # remap padding_mode str
        padding_mode = "replicate"
    elif padding_mode == "symmetric":
        # route to another implementation
        return _pad_symmetric(img, p)

    need_squeeze = False
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if (padding_mode != "constant") and img.dtype not in (torch.float32, torch.float64):
        # Here we temporary cast input tensor to float
        # until pytorch issue is resolved :
        # https://github.com/pytorch/pytorch/issues/40763
        need_cast = True
        img = img.to(torch.float32)

    img = torch_pad(img, p, mode=padding_mode, value=float(fill))

    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        img = img.to(out_dtype)

    return img


def resize(img: Tensor, size: List[int], interpolation: int = 2) -> Tensor:
    r"""PRIVATE METHOD. Resize the input Tensor to the given size.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (Tensor): Image to be resized.
        size (int or tuple or list): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`.
            In torchscript mode padding as a single int is not supported, use a tuple or
            list of length 1: ``[size, ]``.
        interpolation (int, optional): Desired interpolation. Default is bilinear (=2). Other supported values:
            nearest(=0) and bicubic(=3).

    Returns:
        Tensor: Resized image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError("tensor is not a torch image.")

    if not isinstance(size, (int, tuple, list)):
        raise TypeError("Got inappropriate size arg")
    if not isinstance(interpolation, int):
        raise TypeError("Got inappropriate interpolation arg")

    _interpolation_modes = {
        0: "nearest",
        2: "bilinear",
        3: "bicubic",
    }

    if interpolation not in _interpolation_modes:
        raise ValueError("This interpolation mode is unsupported with Tensor input")

    if isinstance(size, tuple):
        size = list(size)

    if isinstance(size, list) and len(size) not in [1, 2]:
        raise ValueError("Size must be an int or a 1 or 2 element tuple/list, not a "
                         "{} element tuple/list".format(len(size)))

    w, h = _get_image_size(img)

    if isinstance(size, int):
        size_w, size_h = size, size
    elif len(size) < 2:
        size_w, size_h = size[0], size[0]
    else:
        size_w, size_h = size[1], size[0]  # Convention (h, w)

    if isinstance(size, int) or len(size) < 2:
        if w < h:
            size_h = int(size_w * h / w)
        else:
            size_w = int(size_h * w / h)

        if (w <= h and w == size_w) or (h <= w and h == size_h):
            return img

    # make image NCHW
    need_squeeze = False
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    mode = _interpolation_modes[interpolation]

    out_dtype = img.dtype
    need_cast = False
    if img.dtype not in (torch.float32, torch.float64):
        need_cast = True
        img = img.to(torch.float32)

    # Define align_corners to avoid warnings
    align_corners = False if mode in ["bilinear", "bicubic"] else None

    img = interpolate(img, size=[size_h, size_w], mode=mode, align_corners=align_corners)

    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if mode == "bicubic":
            img = img.clamp(min=0, max=255)
        img = img.to(out_dtype)

    return img


def _assert_grid_transform_inputs(
        img: Tensor,
        matrix: Optional[List[float]],
        resample: int,
        fillcolor: Optional[int],
        _interpolation_modes: Dict[int, str],
        coeffs: Optional[List[float]] = None,
):
    if not (isinstance(img, torch.Tensor) and _is_tensor_a_torch_image(img)):
        raise TypeError("Input img should be Tensor Image")

    if matrix is not None and not isinstance(matrix, list):
        raise TypeError("Argument matrix should be a list")

    if matrix is not None and len(matrix) != 6:
        raise ValueError("Argument matrix should have 6 float values")

    if coeffs is not None and len(coeffs) != 8:
        raise ValueError("Argument coeffs should have 8 float values")

    if fillcolor is not None:
        warnings.warn("Argument fill/fillcolor is not supported for Tensor input. Fill value is zero")

    if resample not in _interpolation_modes:
        raise ValueError("Resampling mode '{}' is unsupported with Tensor input".format(resample))


def _cast_squeeze_in(img: Tensor, req_dtype: torch.dtype) -> Tuple[Tensor, bool, bool, torch.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype != req_dtype:
        need_cast = True
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype):
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        # it is better to round before cast
        img = torch.round(img).to(out_dtype)

    return img


def _apply_grid_transform(img: Tensor, grid: Tensor, mode: str) -> Tensor:

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, grid.dtype)

    if img.shape[0] > 1:
        # Apply same grid to a batch of images
        grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])
    img = grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


def _gen_affine_grid(
        theta: Tensor, w: int, h: int, ow: int, oh: int,
) -> Tensor:
    # https://github.com/pytorch/pytorch/blob/74b65c32be68b15dc7c9e8bb62459efbfbde33d8/aten/src/ATen/native/
    # AffineGridGenerator.cpp#L18
    # Difference with AffineGridGenerator is that:
    # 1) we normalize grid values after applying theta
    # 2) we can normalize by other image size, such that it covers "extend" option like in PIL.Image.rotate

    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3, dtype=theta.dtype, device=theta.device)
    base_grid[..., 0].copy_(torch.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow))
    base_grid[..., 1].copy_(torch.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh).unsqueeze_(-1))
    base_grid[..., 2].fill_(1)

    rescaled_theta = theta.transpose(1, 2) / torch.tensor([0.5 * w, 0.5 * h], dtype=theta.dtype, device=theta.device)
    output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
    return output_grid.view(1, oh, ow, 2)


def affine(
        img: Tensor, matrix: List[float], resample: int = 0, fillcolor: Optional[int] = None
) -> Tensor:
    """PRIVATE METHOD. Apply affine transformation on the Tensor image keeping image center invariant.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (Tensor): image to be rotated.
        matrix (list of floats): list of 6 float values representing inverse matrix for affine transformation.
        resample (int, optional): An optional resampling filter. Default is nearest (=0). Other supported values:
            bilinear(=2).
        fillcolor (int, optional): this option is not supported for Tensor input. Fill value for the area outside the
            transform in the output image is always 0.

    Returns:
        Tensor: Transformed image.
    """
    _interpolation_modes = {
        0: "nearest",
        2: "bilinear",
    }

    _assert_grid_transform_inputs(img, matrix, resample, fillcolor, _interpolation_modes)

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    theta = torch.tensor(matrix, dtype=dtype, device=img.device).reshape(1, 2, 3)
    shape = img.shape
    # grid will be generated on the same device as theta and img
    grid = _gen_affine_grid(theta, w=shape[-1], h=shape[-2], ow=shape[-1], oh=shape[-2])
    mode = _interpolation_modes[resample]
    return _apply_grid_transform(img, grid, mode)


def _compute_output_size(matrix: List[float], w: int, h: int) -> Tuple[int, int]:

    # Inspired of PIL implementation:
    # https://github.com/python-pillow/Pillow/blob/11de3318867e4398057373ee9f12dcb33db7335c/src/PIL/Image.py#L2054

    # pts are Top-Left, Top-Right, Bottom-Left, Bottom-Right points.
    pts = torch.tensor([
        [-0.5 * w, -0.5 * h, 1.0],
        [-0.5 * w, 0.5 * h, 1.0],
        [0.5 * w, 0.5 * h, 1.0],
        [0.5 * w, -0.5 * h, 1.0],
    ])
    theta = torch.tensor(matrix, dtype=torch.float).reshape(1, 2, 3)
    new_pts = pts.view(1, 4, 3).bmm(theta.transpose(1, 2)).view(4, 2)
    min_vals, _ = new_pts.min(dim=0)
    max_vals, _ = new_pts.max(dim=0)

    # Truncate precision to 1e-4 to avoid ceil of Xe-15 to 1.0
    tol = 1e-4
    cmax = torch.ceil((max_vals / tol).trunc_() * tol)
    cmin = torch.floor((min_vals / tol).trunc_() * tol)
    size = cmax - cmin
    return int(size[0]), int(size[1])


def rotate(
        img: Tensor, matrix: List[float], resample: int = 0, expand: bool = False, fill: Optional[int] = None
) -> Tensor:
    """PRIVATE METHOD. Rotate the Tensor image by angle.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (Tensor): image to be rotated.
        matrix (list of floats): list of 6 float values representing inverse matrix for rotation transformation.
            Translation part (``matrix[2]`` and ``matrix[5]``) should be in pixel coordinates.
        resample (int, optional): An optional resampling filter. Default is nearest (=0). Other supported values:
            bilinear(=2).
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        fill (n-tuple or int or float): this option is not supported for Tensor input.
            Fill value for the area outside the transform in the output image is always 0.

    Returns:
        Tensor: Rotated image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """
    _interpolation_modes = {
        0: "nearest",
        2: "bilinear",
    }

    _assert_grid_transform_inputs(img, matrix, resample, fill, _interpolation_modes)
    w, h = img.shape[-1], img.shape[-2]
    ow, oh = _compute_output_size(matrix, w, h) if expand else (w, h)
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    theta = torch.tensor(matrix, dtype=dtype, device=img.device).reshape(1, 2, 3)
    # grid will be generated on the same device as theta and img
    grid = _gen_affine_grid(theta, w=w, h=h, ow=ow, oh=oh)
    mode = _interpolation_modes[resample]

    return _apply_grid_transform(img, grid, mode)


def _perspective_grid(coeffs: List[float], ow: int, oh: int, dtype: torch.dtype, device: torch.device):
    # https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/
    # src/libImaging/Geometry.c#L394

    #
    # x_out = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / (coeffs[6] * x + coeffs[7] * y + 1)
    # y_out = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / (coeffs[6] * x + coeffs[7] * y + 1)
    #
    theta1 = torch.tensor([[
        [coeffs[0], coeffs[1], coeffs[2]],
        [coeffs[3], coeffs[4], coeffs[5]]
    ]], dtype=dtype, device=device)
    theta2 = torch.tensor([[
        [coeffs[6], coeffs[7], 1.0],
        [coeffs[6], coeffs[7], 1.0]
    ]], dtype=dtype, device=device)

    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3, dtype=dtype, device=device)
    base_grid[..., 0].copy_(torch.linspace(d, ow * 1.0 + d - 1.0, steps=ow))
    base_grid[..., 1].copy_(torch.linspace(d, oh * 1.0 + d - 1.0, steps=oh).unsqueeze_(-1))
    base_grid[..., 2].fill_(1)

    rescaled_theta1 = theta1.transpose(1, 2) / torch.tensor([0.5 * ow, 0.5 * oh], dtype=dtype, device=device)
    output_grid1 = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta1)
    output_grid2 = base_grid.view(1, oh * ow, 3).bmm(theta2.transpose(1, 2))

    output_grid = output_grid1 / output_grid2 - 1.0
    return output_grid.view(1, oh, ow, 2)


def perspective(
        img: Tensor, perspective_coeffs: List[float], interpolation: int = 2, fill: Optional[int] = None
) -> Tensor:
    """PRIVATE METHOD. Perform perspective transform of the given Tensor image.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (Tensor): Image to be transformed.
        perspective_coeffs (list of float): perspective transformation coefficients.
        interpolation (int): Interpolation type. Default, ``PIL.Image.BILINEAR``.
        fill (n-tuple or int or float): this option is not supported for Tensor input. Fill value for the area
            outside the transform in the output image is always 0.

    Returns:
        Tensor: transformed image.
    """
    if not (isinstance(img, torch.Tensor) and _is_tensor_a_torch_image(img)):
        raise TypeError('Input img should be Tensor Image')

    _interpolation_modes = {
        0: "nearest",
        2: "bilinear",
    }

    _assert_grid_transform_inputs(
        img,
        matrix=None,
        resample=interpolation,
        fillcolor=fill,
        _interpolation_modes=_interpolation_modes,
        coeffs=perspective_coeffs
    )

    ow, oh = img.shape[-1], img.shape[-2]
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    grid = _perspective_grid(perspective_coeffs, ow=ow, oh=oh, dtype=dtype, device=img.device)
    mode = _interpolation_modes[interpolation]

    return _apply_grid_transform(img, grid, mode)


def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def _get_gaussian_kernel2d(
        kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d


def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: List[float]) -> Tensor:
    """PRIVATE METHOD. Performs Gaussian blurring on the img by given kernel.

    .. warning::

        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (Tensor): Image to be blurred
        kernel_size (sequence of int or int): Kernel size of the Gaussian kernel ``(kx, ky)``.
        sigma (sequence of float or float, optional): Standard deviation of the Gaussian kernel ``(sx, sy)``.

    Returns:
        Tensor: An image that is blurred using gaussian kernel of given parameters
    """
    if not (isinstance(img, torch.Tensor) or _is_tensor_a_torch_image(img)):
        raise TypeError('img should be Tensor Image. Got {}'.format(type(img)))

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, kernel.dtype)

    # padding = (left, right, top, bottom)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img = torch_pad(img, padding, mode="reflect")
    img = conv2d(img, kernel, groups=img.shape[-3])

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img
