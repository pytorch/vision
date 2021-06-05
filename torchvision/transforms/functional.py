import math
import numbers
import warnings
from enum import Enum

import numpy as np
from PIL import Image

import torch
from torch import Tensor
from typing import List, Tuple, Any, Optional

try:
    import accimage
except ImportError:
    accimage = None

from . import functional_pil as F_pil
from . import functional_tensor as F_t


class InterpolationMode(Enum):
    """Interpolation modes
    Available interpolation methods are ``nearest``, ``bilinear``, ``bicubic``, ``box``, ``hamming``, and ``lanczos``.
    """
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


# TODO: Once torchscript supports Enums with staticmethod
# this can be put into InterpolationMode as staticmethod
def _interpolation_modes_from_int(i: int) -> InterpolationMode:
    inverse_modes_mapping = {
        0: InterpolationMode.NEAREST,
        2: InterpolationMode.BILINEAR,
        3: InterpolationMode.BICUBIC,
        4: InterpolationMode.BOX,
        5: InterpolationMode.HAMMING,
        1: InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[i]


pil_modes_mapping = {
    InterpolationMode.NEAREST: 0,
    InterpolationMode.BILINEAR: 2,
    InterpolationMode.BICUBIC: 3,
    InterpolationMode.BOX: 4,
    InterpolationMode.HAMMING: 5,
    InterpolationMode.LANCZOS: 1,
}

_is_pil_image = F_pil._is_pil_image


def _get_image_size(img: Tensor) -> List[int]:
    """Returns image size as [w, h]
    """
    if isinstance(img, torch.Tensor):
        return F_t._get_image_size(img)

    return F_pil._get_image_size(img)


def _get_image_num_channels(img: Tensor) -> int:
    """Returns number of image channels
    """
    if isinstance(img, torch.Tensor):
        return F_t._get_image_num_channels(img)

    return F_pil._get_image_num_channels(img)


@torch.jit.unused
def _is_numpy(img: Any) -> bool:
    return isinstance(img, np.ndarray)


@torch.jit.unused
def _is_numpy_image(img: Any) -> bool:
    return img.ndim in {2, 3}


def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    This function does not support torchscript.

    See :class:`~torchvision.transforms.ToTensor` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not(F_pil._is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

    default_float_dtype = torch.get_default_dtype()

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.to(dtype=default_float_dtype).div(255)
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic).to(dtype=default_float_dtype)

    # handle PIL Image
    mode_to_nptype = {'I': np.int32, 'I;16': np.int16, 'F': np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )

    if pic.mode == '1':
        img = 255 * img
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1)).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.to(dtype=default_float_dtype).div(255)
    else:
        return img


def pil_to_tensor(pic):
    """Convert a ``PIL Image`` to a tensor of the same type.
    This function does not support torchscript.

    See :class:`~torchvision.transforms.PILToTensor` for more details.

    Args:
        pic (PIL Image): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not F_pil._is_pil_image(pic):
        raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

    if accimage is not None and isinstance(pic, accimage.Image):
        # accimage format is always uint8 internally, so always return uint8 here
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.uint8)
        pic.copyto(nppic)
        return torch.as_tensor(nppic)

    # handle PIL Image
    img = torch.as_tensor(np.asarray(pic))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1))
    return img


def convert_image_dtype(image: torch.Tensor, dtype: torch.dtype = torch.float) -> torch.Tensor:
    """Convert a tensor image to the given ``dtype`` and scale the values accordingly
    This function does not support PIL Image.

    Args:
        image (torch.Tensor): Image to be converted
        dtype (torch.dtype): Desired data type of the output

    Returns:
        Tensor: Converted image

    .. note::

        When converting from a smaller to a larger integer ``dtype`` the maximum values are **not** mapped exactly.
        If converted back and forth, this mismatch has no effect.

    Raises:
        RuntimeError: When trying to cast :class:`torch.float32` to :class:`torch.int32` or :class:`torch.int64` as
            well as for trying to cast :class:`torch.float64` to :class:`torch.int64`. These conversions might lead to
            overflow errors since the floating point ``dtype`` cannot store consecutive integers over the whole range
            of the integer ``dtype``.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError('Input img should be Tensor Image')

    return F_t.convert_image_dtype(image, dtype)


def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image. This function does not support torchscript.

    See :class:`~torchvision.transforms.ToPILImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not(isinstance(pic, torch.Tensor) or isinstance(pic, np.ndarray)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    elif isinstance(pic, torch.Tensor):
        if pic.ndimension() not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndimension()))

        elif pic.ndimension() == 2:
            # if 2D image, add channel dimension (CHW)
            pic = pic.unsqueeze(0)

        # check number of channels
        if pic.shape[-3] > 4:
            raise ValueError('pic should not have > 4 channels. Got {} channels.'.format(pic.shape[-3]))

    elif isinstance(pic, np.ndarray):
        if pic.ndim not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

        elif pic.ndim == 2:
            # if 2D image, add channel dimension (HWC)
            pic = np.expand_dims(pic, 2)

        # check number of channels
        if pic.shape[-1] > 4:
            raise ValueError('pic should not have > 4 channels. Got {} channels.'.format(pic.shape[-1]))

    npimg = pic
    if isinstance(pic, torch.Tensor):
        if pic.is_floating_point() and mode != 'F':
            pic = pic.mul(255).byte()
        npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        elif npimg.dtype == np.int16:
            expected_mode = 'I;16'
        elif npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ['LA']
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError("Only modes {} are supported for 2D inputs".format(permitted_2_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'LA'

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)


def normalize(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
    """Normalize a float tensor image with mean and standard deviation.
    This transform does not support PIL Image.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Float tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if not tensor.is_floating_point():
        raise TypeError('Input tensor should be a float tensor. Got {}.'.format(tensor.dtype))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor


def resize(img: Tensor, size: List[int], interpolation: InterpolationMode = InterpolationMode.BILINEAR,
           max_size: Optional[int] = None, antialias: Optional[bool] = None) -> Tensor:
    r"""Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. warning::
        The output image might be different depending on its type: when downsampling, the interpolation of PIL images
        and tensors is slightly different, because PIL applies antialiasing. This may lead to significant differences
        in the performance of a network. Therefore, it is preferable to train and serve a model with the same input
        types.

    Args:
        img (PIL Image or Tensor): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`.

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is ``InterpolationMode.BILINEAR``. If input is Tensor, only ``InterpolationMode.NEAREST``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        max_size (int, optional): The maximum allowed for the longer edge of
            the resized image: if the longer edge of the image is greater
            than ``max_size`` after being resized according to ``size``, then
            the image is resized again so that the longer edge is equal to
            ``max_size``. As a result, ``size`` might be overruled, i.e the
            smaller edge may be shorter than ``size``. This is only supported
            if ``size`` is an int (or a sequence of length 1 in torchscript
            mode).
        antialias (bool, optional): antialias flag. If ``img`` is PIL Image, the flag is ignored and anti-alias
            is always used. If ``img`` is Tensor, the flag is False by default and can be set True for
            ``InterpolationMode.BILINEAR`` only mode.

            .. warning::
                There is no autodiff support for ``antialias=True`` option with input ``img`` as Tensor.

    Returns:
        PIL Image or Tensor: Resized image.
    """
    # Backward compatibility with integer value
    if isinstance(interpolation, int):
        warnings.warn(
            "Argument interpolation should be of type InterpolationMode instead of int. "
            "Please, use InterpolationMode enum."
        )
        interpolation = _interpolation_modes_from_int(interpolation)

    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")

    if not isinstance(img, torch.Tensor):
        if antialias is not None and not antialias:
            warnings.warn(
                "Anti-alias option is always applied for PIL Image input. Argument antialias is ignored."
            )
        pil_interpolation = pil_modes_mapping[interpolation]
        return F_pil.resize(img, size=size, interpolation=pil_interpolation, max_size=max_size)

    return F_t.resize(img, size=size, interpolation=interpolation.value, max_size=max_size, antialias=antialias)


def scale(*args, **kwargs):
    warnings.warn("The use of the transforms.Scale transform is deprecated, " +
                  "please use transforms.Resize instead.")
    return resize(*args, **kwargs)


def pad(img: Tensor, padding: List[int], fill: int = 0, padding_mode: str = "constant") -> Tensor:
    r"""Pad the given image on all sides with the given "pad" value.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means at most 2 leading dimensions for mode reflect and symmetric,
    at most 3 leading dimensions for mode edge,
    and an arbitrary number of leading dimensions for mode constant

    Args:
        img (PIL Image or Tensor): Image to be padded.
        padding (int or sequence): Padding on each border. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.

            .. note::
                In torchscript mode padding as single int is not supported, use a sequence of
                length 1: ``[padding, ]``.
        fill (number or str or tuple): Pixel fill value for constant fill. Default is 0.
            If a tuple of length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            Only number is supported for torch Tensor.
            Only int or str or tuple value is supported for PIL Image.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value at the edge of the image.
              If input a 5D torch Tensor, the last 3 dimensions will be padded instead of the last 2

            - reflect: pads with reflection of image without repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
              will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
              will result in [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        PIL Image or Tensor: Padded image.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.pad(img, padding=padding, fill=fill, padding_mode=padding_mode)

    return F_t.pad(img, padding=padding, fill=fill, padding_mode=padding_mode)


def crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    """Crop the given image at specified location and output size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        PIL Image or Tensor: Cropped image.
    """

    if not isinstance(img, torch.Tensor):
        return F_pil.crop(img, top, left, height, width)

    return F_t.crop(img, top, left, height, width)


def center_crop(img: Tensor, output_size: List[int]) -> Tensor:
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    image_width, image_height = _get_image_size(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = pad(img, padding_ltrb, fill=0)  # PIL uses fill value 0
        image_width, image_height = _get_image_size(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return crop(img, crop_top, crop_left, crop_height, crop_width)


def resized_crop(
        img: Tensor, top: int, left: int, height: int, width: int, size: List[int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR
) -> Tensor:
    """Crop the given image and resize it to desired size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.

    Args:
        img (PIL Image or Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        size (sequence or int): Desired output size. Same semantics as ``resize``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is ``InterpolationMode.BILINEAR``. If input is Tensor, only ``InterpolationMode.NEAREST``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation)
    return img


def hflip(img: Tensor) -> Tensor:
    """Horizontally flip the given image.

    Args:
        img (PIL Image or Tensor): Image to be flipped. If img
            is a Tensor, it is expected to be in [..., H, W] format,
            where ... means it can have an arbitrary number of leading
            dimensions.

    Returns:
        PIL Image or Tensor:  Horizontally flipped image.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.hflip(img)

    return F_t.hflip(img)


def _get_perspective_coeffs(
        startpoints: List[List[int]], endpoints: List[List[int]]
) -> List[float]:
    """Helper function to get the coefficients (a, b, c, d, e, f, g, h) for the perspective transforms.

    In Perspective Transform each pixel (x, y) in the original image gets transformed as,
     (x, y) -> ( (ax + by + c) / (gx + hy + 1), (dx + ey + f) / (gx + hy + 1) )

    Args:
        startpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the original image.
        endpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the transformed image.

    Returns:
        octuple (a, b, c, d, e, f, g, h) for transforming each pixel.
    """
    a_matrix = torch.zeros(2 * len(startpoints), 8, dtype=torch.float)

    for i, (p1, p2) in enumerate(zip(endpoints, startpoints)):
        a_matrix[2 * i, :] = torch.tensor([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        a_matrix[2 * i + 1, :] = torch.tensor([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    b_matrix = torch.tensor(startpoints, dtype=torch.float).view(8)
    res = torch.lstsq(b_matrix, a_matrix)[0]

    output: List[float] = res.squeeze(1).tolist()
    return output


def perspective(
        img: Tensor,
        startpoints: List[List[int]],
        endpoints: List[List[int]],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None
) -> Tensor:
    """Perform perspective transform of the given image.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): Image to be transformed.
        startpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the original image.
        endpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the transformed image.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.

            .. note::
                In torchscript mode single int/float value is not supported, please use a sequence
                of length 1: ``[value, ]``.

    Returns:
        PIL Image or Tensor: transformed Image.
    """

    coeffs = _get_perspective_coeffs(startpoints, endpoints)

    # Backward compatibility with integer value
    if isinstance(interpolation, int):
        warnings.warn(
            "Argument interpolation should be of type InterpolationMode instead of int. "
            "Please, use InterpolationMode enum."
        )
        interpolation = _interpolation_modes_from_int(interpolation)

    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")

    if not isinstance(img, torch.Tensor):
        pil_interpolation = pil_modes_mapping[interpolation]
        return F_pil.perspective(img, coeffs, interpolation=pil_interpolation, fill=fill)

    return F_t.perspective(img, coeffs, interpolation=interpolation.value, fill=fill)


def vflip(img: Tensor) -> Tensor:
    """Vertically flip the given image.

    Args:
        img (PIL Image or Tensor): Image to be flipped. If img
            is a Tensor, it is expected to be in [..., H, W] format,
            where ... means it can have an arbitrary number of leading
            dimensions.

    Returns:
        PIL Image or Tensor:  Vertically flipped image.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.vflip(img)

    return F_t.vflip(img)


def five_crop(img: Tensor, size: List[int]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Crop the given image into four corners and the central crop.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

    Returns:
       tuple: tuple (tl, tr, bl, br, center)
       Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = (size[0], size[0])

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    image_width, image_height = _get_image_size(img)
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    tl = crop(img, 0, 0, crop_height, crop_width)
    tr = crop(img, 0, image_width - crop_width, crop_height, crop_width)
    bl = crop(img, image_height - crop_height, 0, crop_height, crop_width)
    br = crop(img, image_height - crop_height, image_width - crop_width, crop_height, crop_width)

    center = center_crop(img, [crop_height, crop_width])

    return tl, tr, bl, br, center


def ten_crop(img: Tensor, size: List[int], vertical_flip: bool = False) -> List[Tensor]:
    """Generate ten cropped images from the given image.
    Crop the given image into four corners and the central crop plus the
    flipped version of these (horizontal flipping is used by default).
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        vertical_flip (bool): Use vertical flipping instead of horizontal

    Returns:
        tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip)
        Corresponding top left, top right, bottom left, bottom right and
        center crop and same for the flipped image.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = (size[0], size[0])

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    first_five = five_crop(img, size)

    if vertical_flip:
        img = vflip(img)
    else:
        img = hflip(img)

    second_five = five_crop(img, size)
    return first_five + second_five


def adjust_brightness(img: Tensor, brightness_factor: float) -> Tensor:
    """Adjust brightness of an image.

    Args:
        img (PIL Image or Tensor): Image to be adjusted.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL Image or Tensor: Brightness adjusted image.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.adjust_brightness(img, brightness_factor)

    return F_t.adjust_brightness(img, brightness_factor)


def adjust_contrast(img: Tensor, contrast_factor: float) -> Tensor:
    """Adjust contrast of an image.

    Args:
        img (PIL Image or Tensor): Image to be adjusted.
            If img is torch Tensor, it is expected to be in [..., 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL Image or Tensor: Contrast adjusted image.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.adjust_contrast(img, contrast_factor)

    return F_t.adjust_contrast(img, contrast_factor)


def adjust_saturation(img: Tensor, saturation_factor: float) -> Tensor:
    """Adjust color saturation of an image.

    Args:
        img (PIL Image or Tensor): Image to be adjusted.
            If img is torch Tensor, it is expected to be in [..., 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image or Tensor: Saturation adjusted image.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.adjust_saturation(img, saturation_factor)

    return F_t.adjust_saturation(img, saturation_factor)


def adjust_hue(img: Tensor, hue_factor: float) -> Tensor:
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        img (PIL Image or Tensor): Image to be adjusted.
            If img is torch Tensor, it is expected to be in [..., 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            If img is PIL Image mode "1", "L", "I", "F" and modes with transparency (alpha channel) are not supported.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL Image or Tensor: Hue adjusted image.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.adjust_hue(img, hue_factor)

    return F_t.adjust_hue(img, hue_factor)


def adjust_gamma(img: Tensor, gamma: float, gain: float = 1) -> Tensor:
    r"""Perform gamma correction on an image.

    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:

    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}

    See `Gamma Correction`_ for more details.

    .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction

    Args:
        img (PIL Image or Tensor): PIL Image to be adjusted.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            If img is PIL Image, modes with transparency (alpha channel) are not supported.
        gamma (float): Non negative real number, same as :math:`\gamma` in the equation.
            gamma larger than 1 make the shadows darker,
            while gamma smaller than 1 make dark regions lighter.
        gain (float): The constant multiplier.
    Returns:
        PIL Image or Tensor: Gamma correction adjusted image.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.adjust_gamma(img, gamma, gain)

    return F_t.adjust_gamma(img, gamma, gain)


def _get_inverse_affine_matrix(
        center: List[float], angle: float, translate: List[float], scale: float, shear: List[float]
) -> List[float]:
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    #
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    matrix = [d, -b, 0.0, -c, a, 0.0]
    matrix = [x / scale for x in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += cx
    matrix[5] += cy

    return matrix


def rotate(
        img: Tensor, angle: float, interpolation: InterpolationMode = InterpolationMode.NEAREST,
        expand: bool = False, center: Optional[List[int]] = None,
        fill: Optional[List[float]] = None, resample: Optional[int] = None
) -> Tensor:
    """Rotate the image by angle.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): image to be rotated.
        angle (number): rotation angle value in degrees, counter-clockwise.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (sequence, optional): Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.

            .. note::
                In torchscript mode single int/float value is not supported, please use a sequence
                of length 1: ``[value, ]``.

    Returns:
        PIL Image or Tensor: Rotated image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """
    if resample is not None:
        warnings.warn(
            "Argument resample is deprecated and will be removed since v0.10.0. Please, use interpolation instead"
        )
        interpolation = _interpolation_modes_from_int(resample)

    # Backward compatibility with integer value
    if isinstance(interpolation, int):
        warnings.warn(
            "Argument interpolation should be of type InterpolationMode instead of int. "
            "Please, use InterpolationMode enum."
        )
        interpolation = _interpolation_modes_from_int(interpolation)

    if not isinstance(angle, (int, float)):
        raise TypeError("Argument angle should be int or float")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")

    if not isinstance(img, torch.Tensor):
        pil_interpolation = pil_modes_mapping[interpolation]
        return F_pil.rotate(img, angle=angle, interpolation=pil_interpolation, expand=expand, center=center, fill=fill)

    center_f = [0.0, 0.0]
    if center is not None:
        img_size = _get_image_size(img)
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, img_size)]

    # due to current incoherence of rotation angle direction between affine and rotate implementations
    # we need to set -angle.
    matrix = _get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])
    return F_t.rotate(img, matrix=matrix, interpolation=interpolation.value, expand=expand, fill=fill)


def affine(
        img: Tensor, angle: float, translate: List[int], scale: float, shear: List[float],
        interpolation: InterpolationMode = InterpolationMode.NEAREST, fill: Optional[List[float]] = None,
        resample: Optional[int] = None, fillcolor: Optional[List[float]] = None
) -> Tensor:
    """Apply affine transformation on the image keeping image center invariant.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): image to transform.
        angle (number): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (sequence of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float or sequence): shear angle value in degrees between -180 to 180, clockwise direction.
            If a sequence is specified, the first value corresponds to a shear parallel to the x axis, while
            the second value corresponds to a shear parallel to the y axis.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.

            .. note::
                In torchscript mode single int/float value is not supported, please use a sequence
                of length 1: ``[value, ]``.
        fillcolor (sequence, int, float): deprecated argument and will be removed since v0.10.0.
            Please use the ``fill`` parameter instead.
        resample (int, optional): deprecated argument and will be removed since v0.10.0.
            Please use the ``interpolation`` parameter instead.

    Returns:
        PIL Image or Tensor: Transformed image.
    """
    if resample is not None:
        warnings.warn(
            "Argument resample is deprecated and will be removed since v0.10.0. Please, use interpolation instead"
        )
        interpolation = _interpolation_modes_from_int(resample)

    # Backward compatibility with integer value
    if isinstance(interpolation, int):
        warnings.warn(
            "Argument interpolation should be of type InterpolationMode instead of int. "
            "Please, use InterpolationMode enum."
        )
        interpolation = _interpolation_modes_from_int(interpolation)

    if fillcolor is not None:
        warnings.warn(
            "Argument fillcolor is deprecated and will be removed since v0.10.0. Please, use fill instead"
        )
        fill = fillcolor

    if not isinstance(angle, (int, float)):
        raise TypeError("Argument angle should be int or float")

    if not isinstance(translate, (list, tuple)):
        raise TypeError("Argument translate should be a sequence")

    if len(translate) != 2:
        raise ValueError("Argument translate should be a sequence of length 2")

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    if not isinstance(shear, (numbers.Number, (list, tuple))):
        raise TypeError("Shear should be either a single value or a sequence of two values")

    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")

    if isinstance(angle, int):
        angle = float(angle)

    if isinstance(translate, tuple):
        translate = list(translate)

    if isinstance(shear, numbers.Number):
        shear = [shear, 0.0]

    if isinstance(shear, tuple):
        shear = list(shear)

    if len(shear) == 1:
        shear = [shear[0], shear[0]]

    if len(shear) != 2:
        raise ValueError("Shear should be a sequence containing two values. Got {}".format(shear))

    img_size = _get_image_size(img)
    if not isinstance(img, torch.Tensor):
        # center = (img_size[0] * 0.5 + 0.5, img_size[1] * 0.5 + 0.5)
        # it is visually better to estimate the center without 0.5 offset
        # otherwise image rotated by 90 degrees is shifted vs output image of torch.rot90 or F_t.affine
        center = [img_size[0] * 0.5, img_size[1] * 0.5]
        matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
        pil_interpolation = pil_modes_mapping[interpolation]
        return F_pil.affine(img, matrix=matrix, interpolation=pil_interpolation, fill=fill)

    translate_f = [1.0 * t for t in translate]
    matrix = _get_inverse_affine_matrix([0.0, 0.0], angle, translate_f, scale, shear)
    return F_t.affine(img, matrix=matrix, interpolation=interpolation.value, fill=fill)


@torch.jit.unused
def to_grayscale(img, num_output_channels=1):
    """Convert PIL image of any mode (RGB, HSV, LAB, etc) to grayscale version of image.
    This transform does not support torch Tensor.

    Args:
        img (PIL Image): PIL Image to be converted to grayscale.
        num_output_channels (int): number of channels of the output image. Value can be 1 or 3. Default is 1.

    Returns:
        PIL Image: Grayscale version of the image.

        - if num_output_channels = 1 : returned image is single channel
        - if num_output_channels = 3 : returned image is 3 channel with r = g = b
    """
    if isinstance(img, Image.Image):
        return F_pil.to_grayscale(img, num_output_channels)

    raise TypeError("Input should be PIL Image")


def rgb_to_grayscale(img: Tensor, num_output_channels: int = 1) -> Tensor:
    """Convert RGB image to grayscale version of image.
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions

    Note:
        Please, note that this method supports only RGB images as input. For inputs in other color spaces,
        please, consider using meth:`~torchvision.transforms.functional.to_grayscale` with PIL Image.

    Args:
        img (PIL Image or Tensor): RGB Image to be converted to grayscale.
        num_output_channels (int): number of channels of the output image. Value can be 1 or 3. Default, 1.

    Returns:
        PIL Image or Tensor: Grayscale version of the image.

        - if num_output_channels = 1 : returned image is single channel
        - if num_output_channels = 3 : returned image is 3 channel with r = g = b
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.to_grayscale(img, num_output_channels)

    return F_t.rgb_to_grayscale(img, num_output_channels)


def erase(img: Tensor, i: int, j: int, h: int, w: int, v: Tensor, inplace: bool = False) -> Tensor:
    """ Erase the input Tensor Image with given value.
    This transform does not support PIL Image.

    Args:
        img (Tensor Image): Tensor image of size (C, H, W) to be erased
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the erased region.
        w (int): Width of the erased region.
        v: Erasing value.
        inplace(bool, optional): For in-place operations. By default is set False.

    Returns:
        Tensor Image: Erased image.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError('img should be Tensor Image. Got {}'.format(type(img)))

    if not inplace:
        img = img.clone()

    img[..., i:i + h, j:j + w] = v
    return img


def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: Optional[List[float]] = None) -> Tensor:
    """Performs Gaussian blurring on the image by given kernel.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): Image to be blurred
        kernel_size (sequence of ints or int): Gaussian kernel size. Can be a sequence of integers
            like ``(kx, ky)`` or a single integer for square kernels.

            .. note::
                In torchscript mode kernel_size as single int is not supported, use a sequence of
                length 1: ``[ksize, ]``.
        sigma (sequence of floats or float, optional): Gaussian kernel standard deviation. Can be a
            sequence of floats like ``(sigma_x, sigma_y)`` or a single float to define the
            same sigma in both X/Y directions. If None, then it is computed using
            ``kernel_size`` as ``sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8``.
            Default, None.

            .. note::
                In torchscript mode sigma as single float is
                not supported, use a sequence of length 1: ``[sigma, ]``.

    Returns:
        PIL Image or Tensor: Gaussian Blurred version of the image.
    """
    if not isinstance(kernel_size, (int, list, tuple)):
        raise TypeError('kernel_size should be int or a sequence of integers. Got {}'.format(type(kernel_size)))
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if len(kernel_size) != 2:
        raise ValueError('If kernel_size is a sequence its length should be 2. Got {}'.format(len(kernel_size)))
    for ksize in kernel_size:
        if ksize % 2 == 0 or ksize < 0:
            raise ValueError('kernel_size should have odd and positive integers. Got {}'.format(kernel_size))

    if sigma is None:
        sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]

    if sigma is not None and not isinstance(sigma, (int, float, list, tuple)):
        raise TypeError('sigma should be either float or sequence of floats. Got {}'.format(type(sigma)))
    if isinstance(sigma, (int, float)):
        sigma = [float(sigma), float(sigma)]
    if isinstance(sigma, (list, tuple)) and len(sigma) == 1:
        sigma = [sigma[0], sigma[0]]
    if len(sigma) != 2:
        raise ValueError('If sigma is a sequence, its length should be 2. Got {}'.format(len(sigma)))
    for s in sigma:
        if s <= 0.:
            raise ValueError('sigma should have positive values. Got {}'.format(sigma))

    t_img = img
    if not isinstance(img, torch.Tensor):
        if not F_pil._is_pil_image(img):
            raise TypeError('img should be PIL Image or Tensor. Got {}'.format(type(img)))

        t_img = to_tensor(img)

    output = F_t.gaussian_blur(t_img, kernel_size, sigma)

    if not isinstance(img, torch.Tensor):
        output = to_pil_image(output)
    return output


def invert(img: Tensor) -> Tensor:
    """Invert the colors of an RGB/grayscale image.

    Args:
        img (PIL Image or Tensor): Image to have its colors inverted.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Returns:
        PIL Image or Tensor: Color inverted image.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.invert(img)

    return F_t.invert(img)


def posterize(img: Tensor, bits: int) -> Tensor:
    """Posterize an image by reducing the number of bits for each color channel.

    Args:
        img (PIL Image or Tensor): Image to have its colors posterized.
            If img is torch Tensor, it should be of type torch.uint8 and
            it is expected to be in [..., 1 or 3, H, W] format, where ... means
            it can have an arbitrary number of leading dimensions.
            If img is PIL Image, it is expected to be in mode "L" or "RGB".
        bits (int): The number of bits to keep for each channel (0-8).
    Returns:
        PIL Image or Tensor: Posterized image.
    """
    if not (0 <= bits <= 8):
        raise ValueError('The number if bits should be between 0 and 8. Got {}'.format(bits))

    if not isinstance(img, torch.Tensor):
        return F_pil.posterize(img, bits)

    return F_t.posterize(img, bits)


def solarize(img: Tensor, threshold: float) -> Tensor:
    """Solarize an RGB/grayscale image by inverting all pixel values above a threshold.

    Args:
        img (PIL Image or Tensor): Image to have its colors inverted.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            If img is PIL Image, it is expected to be in mode "L" or "RGB".
        threshold (float): All pixels equal or above this value are inverted.
    Returns:
        PIL Image or Tensor: Solarized image.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.solarize(img, threshold)

    return F_t.solarize(img, threshold)


def adjust_sharpness(img: Tensor, sharpness_factor: float) -> Tensor:
    """Adjust the sharpness of an image.

    Args:
        img (PIL Image or Tensor): Image to be adjusted.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
        sharpness_factor (float):  How much to adjust the sharpness. Can be
            any non negative number. 0 gives a blurred image, 1 gives the
            original image while 2 increases the sharpness by a factor of 2.

    Returns:
        PIL Image or Tensor: Sharpness adjusted image.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.adjust_sharpness(img, sharpness_factor)

    return F_t.adjust_sharpness(img, sharpness_factor)


def autocontrast(img: Tensor) -> Tensor:
    """Maximize contrast of an image by remapping its
    pixels per channel so that the lowest becomes black and the lightest
    becomes white.

    Args:
        img (PIL Image or Tensor): Image on which autocontrast is applied.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Returns:
        PIL Image or Tensor: An image that was autocontrasted.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.autocontrast(img)

    return F_t.autocontrast(img)


def equalize(img: Tensor) -> Tensor:
    """Equalize the histogram of an image by applying
    a non-linear mapping to the input in order to create a uniform
    distribution of grayscale values in the output.

    Args:
        img (PIL Image or Tensor): Image on which equalize is applied.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            The tensor dtype must be ``torch.uint8`` and values are expected to be in ``[0, 255]``.
            If img is PIL Image, it is expected to be in mode "P", "L" or "RGB".

    Returns:
        PIL Image or Tensor: An image that was equalized.
    """
    if not isinstance(img, torch.Tensor):
        return F_pil.equalize(img)

    return F_t.equalize(img)
