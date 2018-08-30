import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings
import cv2

_cv2_pad_to_str = {'constant':cv2.BORDER_CONSTANT,
                   'edge':cv2.BORDER_REPLICATE,
                   'reflect':cv2.BORDER_REFLECT_101,
                   'symmetric':cv2.BORDER_REFLECT
                  }
_cv2_interpolation_to_str= {'nearest':cv2.INTER_NEAREST,
                         'bilinear':cv2.INTER_LINEAR,
                         'area':cv2.INTER_AREA,
                         'bicubic':cv2.INTER_CUBIC,
                         'lanczos':cv2.INTER_LANCZOS4}
_cv2_interpolation_from_str= {v:k for k,v in _cv2_interpolation_to_str.items()}

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    See ``ToTensor`` for more details.
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """
    if not(_is_numpy_image(pic)):
        raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))

    # handle numpy array
    img = torch.from_numpy(pic.transpose((2, 0, 1)))
    # backward compatibility
    if isinstance(img, torch.ByteTensor) or img.dtype==torch.uint8:
        return img.float().div(255)
    else:
        return img

def normalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.
    See :class:`~torchvision.transforms.Normalize` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.
    Returns:
        Tensor: Normalized Tensor image.
    """
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    # This is faster than using broadcasting, don't change without benchmarking
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor

def resize(img, size, interpolation=cv2.INTER_CUBIC):
    r"""Resize the input numpy ndarray to the given size.
    Args:
        img (numpy ndarray): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_CUBIC``
    Returns:
        PIL Image: Resized image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))
    w, h, =  size

    if isinstance(size, int):
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            output = cv2.resize(img, dsize=(ow, oh), interpolation=interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            output = cv2.resize(img, dsize=(ow, oh), interpolation=interpolation)
    else:
        output = cv2.resize(img, dsize=size[::-1], interpolation=interpolation)
    if img.shape[2]==1:
        return(output[:,:,np.newaxis])
    else:
        return(output)

def scale(*args, **kwargs):
    warnings.warn("The use of the transforms.Scale transform is deprecated, " +
                  "please use transforms.Resize instead.")
    return resize(*args, **kwargs)

def pad(img, padding, fill=0, padding_mode='constant'):
    r"""Pad the given numpy ndarray on all sides with specified padding mode and fill value.
    Args:
        img (numpy ndarray): image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value on the edge of the image
            - reflect: pads with reflection of image (without repeating the last value on the edge)
                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image (repeating the last value on the edge)
                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]
    Returns:
        Numpy image: padded image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(type(img)))
    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')
    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, collections.Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, collections.Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]
    if img.shape[2]==1:
        return(cv2.copyMakeBorder(img, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
                                 borderType=_cv2_pad_to_str[padding_mode], value=fill)[:,:,np.newaxis])
    else:
        return(cv2.copyMakeBorder(img, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
                                     borderType=_cv2_pad_to_str[padding_mode], value=fill))

def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (numpy ndarray): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        numpy ndarray: Cropped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    return img[i:i+h, j:j+w, :]

def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.shape[0:2]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)

def resized_crop(img, i, j, h, w, size, interpolation=cv2.INTER_CUBIC):
    """Crop the given numpy ndarray and resize it to desired size.
    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.
    Args:
        img (numpy ndarray): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``scale``.
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_CUBIC``.
    Returns:
        PIL Image: Cropped image.
    """
    assert _is_numpy_image(img), 'img should be numpy image'
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation=interpolation)
    return img

def hflip(img):
    """Horizontally flip the given numpy ndarray.
    Args:
        img (numpy ndarray): image to be flipped.
    Returns:
        numpy ndarray:  Horizontally flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    # img[:,::-1] is much faster, but doesn't work with torch.from_numpy()!
    if img.shape[2]==1:
        return cv2.flip(img,1)[:,:,np.newaxis]
    else:
        return cv2.flip(img, 1)

def vflip(img):
    """Vertically flip the given numpy ndarray.
    Args:
        img (numpy ndarray): Image to be flipped.
    Returns:
        numpy ndarray:  Vertically flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    if img.shape[2]==1:
        return cv2.flip(img, 0)[:,:,np.newaxis]
    else:
        return cv2.flip(img, 0)
    # img[::-1] is much faster, but doesn't work with torch.from_numpy()!
    


def five_crop(img, size):
    """Crop the given numpy ndarray into four corners and the central crop.
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
       tuple: tuple (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.shape[0:2]
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    tl = crop(img, 0, 0, crop_h, crop_w)
    tr = crop(img, w - crop_w, 0, crop_h, w)
    bl = crop(img, 0, h - crop_h, crop_w, h)
    br = crop(img, w - crop_w, h - crop_h,  h,w)
    center = center_crop(img, (crop_h, crop_w))
    return (tl, tr, bl, br, center)


def ten_crop(img, size, vertical_flip=False):
    r"""Crop the given numpy ndarray into four corners and the central crop plus the
        flipped version of these (horizontal flipping is used by default).
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
       vertical_flip (bool): Use vertical flipping instead of horizontal
    Returns:
       tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip)
                Corresponding top left, top right, bottom left, bottom right and center crop
                and same for the flipped image.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    first_five = five_crop(img, size)

    if vertical_flip:
        img = vflip(img)
    else:
        img = hflip(img)

    second_five = five_crop(img, size)
    return first_five + second_five


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        numpy ndarray: Brightness adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([ i*brightness_factor for i in range (0,256)]).clip(0,255).astype('uint8')
    # same thing but a bit slower
    # cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
    if img.shape[2]==1:
        return cv2.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv2.LUT(img, table)


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an mage.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        numpy ndarray: Contrast adjusted image.
    """
    # much faster to use the LUT construction than anything else I've tried
    # it's because you have to change dtypes multiple times
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([ (i-74)*contrast_factor+74 for i in range (0,256)]).clip(0,255).astype('uint8')
    # enhancer = ImageEnhance.Contrast(img)
    # img = enhancer.enhance(contrast_factor)
    if img.shape[2]==1:
        return cv2.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv2.LUT(img,table)


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        numpy ndarray: Saturation adjusted image.
    """
    # ~10ms slower than PIL!
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return np.array(img)


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See `Hue`_ for more details.
    .. _Hue: https://en.wikipedia.org/wiki/Hue
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        numpy ndarray: Hue adjusted image.
    """
    # After testing, found that OpenCV calculates the Hue in a call to 
    # cv2.cvtColor(..., cv2.COLOR_BGR2HSV) differently from PIL

    # This function takes 160ms! should be avoided
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    img = Image.fromarray(img)
    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return np.array(img)

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return np.array(img)


def adjust_gamma(img, gamma, gain=1):
    r"""Perform gamma correction on an image.
    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:
    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}
    See `Gamma Correction`_ for more details.
    .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        gamma (float): Non negative real number, same as :math:`\gamma` in the equation.
            gamma larger than 1 make the shadows darker,
            while gamma smaller than 1 make dark regions lighter.
        gain (float): The constant multiplier.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')
    # from here
    # https://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-opencv-python/41061351
    table = np.array([((i / 255.0) ** gamma) * 255 * gain 
                      for i in np.arange(0, 256)]).astype('uint8')
    if img.shape[2]==1:
        return cv2.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv2.LUT(img,table)


def rotate(img, angle, resample=False, expand=False, center=None):
    """Rotate the image by angle.
    Args:
        img (numpy ndarray): numpy ndarray to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    rows,cols = img.shape[0:2]
    if center is None:
        center = (cols/2, rows/2)
    M = cv2.getRotationMatrix2D(center,angle,1)
    if img.shape[2]==1:
        return cv2.warpAffine(img,M,(cols,rows))[:,:,np.newaxis]
    else:
        return cv2.warpAffine(img,M,(cols,rows))


def _get_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute matrix for affine transformation
    # We need compute affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a)*scale    -sin(a + shear)*scale     0]
    #                              [ sin(a)*scale    cos(a + shear)*scale     0]
    #                              [     0                  0          1]
    
    angle = math.radians(angle)
    shear = math.radians(shear)
    # scale = 1.0 / scale
    
    T = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0,0,1]])
    C = np.array([[1, 0, center[0]], [0, 1, center[1]], [0,0,1]])
    RSS = np.array([[math.cos(angle)*scale, -math.sin(angle+shear)*scale, 0],
                   [math.sin(angle)*scale, math.cos(angle+shear)*scale, 0],
                   [0,0,1]])
    matrix = T @ C @ RSS @ np.linalg.inv(C)
    
    return matrix[:2,:]

def affine(img, angle, translate, scale, shear, interpolation=cv2.INTER_CUBIC, mode=cv2.BORDER_CONSTANT, fillcolor=0):
    """Apply affine transformation on the image keeping image center invariant
    Args:
        img (numpy ndarray): numpy ndarray to be transformed.
        angle (float or int): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float): shear angle value in degrees between -180 to 180, clockwise direction.
        interpolation (``cv2.INTER_NEAREST` or ``cv2.INTER_LINEAR`` or ``cv2.INTER_AREA``, ``cv2.INTER_CUBIC``):
            An optional resampling filter.
            See `filters`_ for more information.
            If omitted, it is set to ``cv2.INTER_CUBIC``, for bicubic interpolation.
        mode (``cv2.BORDER_CONSTANT`` or ``cv2.BORDER_REPLICATE`` or ``cv2.BORDER_REFLECT`` or ``cv2.BORDER_REFLECT_101``)
            Method for filling in border regions. 
            Defaults to cv2.BORDER_CONSTANT, meaning areas outside the image are filled with a value (val, default 0)
        val (int): Optional fill color for the area outside the transform in the output image. Default: 0
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    output_size = img.shape[0:2]
    center = (img.shape[1] * 0.5 + 0.5, img.shape[0] * 0.5 + 0.5)
    matrix = _get_affine_matrix(center, angle, translate, scale, shear)
    
    if img.shape[2]==1:
        return cv2.warpAffine(img, matrix, output_size[::-1],interpolation, borderMode=mode, borderValue=fillcolor)[:,:,np.newaxis]
    else:
        return cv2.warpAffine(img, matrix, output_size[::-1],interpolation, borderMode=mode, borderValue=fillcolor)

def to_grayscale(img, num_output_channels=1):
    """Convert image to grayscale version of image.
    Args:
        img (numpy ndarray): Image to be converted to grayscale.
    Returns:
        numpy ndarray: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel
            if num_output_channels = 3 : returned image is 3 channel with r = g = b
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(type(img)))

    if num_output_channels==1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:,:,np.newaxis]
    elif num_output_channels==3:
        # much faster than doing cvtColor to go back to gray
        img = np.broadcast_to(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:,:,np.newaxis], img.shape) 
    return img