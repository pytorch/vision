from __future__ import division
import torch
import math
import random
from PIL import Image
import cv2
import numpy as np
import numbers
import types
import collections
import warnings
import matplotlib.pyplot as plt
from torchvision.transforms import functional
import PIL

INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}
PAD_MOD = {'constant': cv2.BORDER_CONSTANT,
           'edge': cv2.BORDER_REPLICATE,
           'reflect': cv2.BORDER_DEFAULT,
           'symmetric': cv2.BORDER_REFLECT
           }


def imshow(inps, title=None):
    """Imshow for Tensor."""
    subwindows = len(inps)
    for idx, (inp, name) in enumerate(zip(inps, title)):
        inp = inp.numpy().transpose((1, 2, 0))
        ax = plt.subplot(1, subwindows, idx+1)
        ax.axis('off')
        plt.imshow(inp)
        ax.set_title(name)
        # plt.pause(0.001)
    plt.show()
    # plt.waitforbuttonpress(-1)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(pic):
    """Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

    Args:
        pic (np.ndarray, torch.Tensor): Image to be converted to tensor, (H x W x C[RGB]).

    Returns:
        Tensor: Converted image.
    """

    if _is_numpy_image(pic):
        if len(pic.shape) == 2:
            pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor) or img.max() > 1:
            return img.float().div(255)
        else:
            return img
    elif _is_tensor_image(pic):
        return pic

    else:
        try:
            return to_tensor(np.array(pic))
        except Exception:
            raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))


def to_cv_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (str): color space and pixel depth of input data (optional).

    Returns:
        np.array: Image converted to PIL Image.
    """
    if not(_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.squeeze(np.transpose(pic.numpy(), (1, 2, 0)))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))
    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))
    return cv2.cvtColor(npimg, mode)


def normalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See ``Normalize`` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if _is_tensor_image(tensor):
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor
    elif _is_numpy_image(tensor):
        return (tensor.astype(np.float32) - 255.0 * np.array(mean))/np.array(std)
    else:
        raise RuntimeError('Undefined type')


def resize(img, size, interpolation='BILINEAR'):
    """Resize the input CV Image to the given size.

    Args:
        img (np.ndarray): Image to be resized.
        size (tuple or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (str, optional): Desired interpolation. Default is ``BILINEAR``

    Returns:
        cv Image: Resized image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        h, w, c = img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(img, dsize=(ow, oh), interpolation=INTER_MODE[interpolation])
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(img, dsize=(ow, oh), interpolation=INTER_MODE[interpolation])
    else:
        oh, ow = size
        return cv2.resize(img, dsize=(int(ow), int(oh)), interpolation=INTER_MODE[interpolation])


def pad(img, padding, fill=(0, 0, 0), padding_mode='constant'):
    """Pad the given CV Image on all sides with speficified padding mode and fill value.
    Args:
        img (np.ndarray): Image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int, tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            constant: pads with a constant value, this value is specified with fill
            edge: pads with the last value on the edge of the image
            reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        CV Image: Padded image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

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
        pad_left, pad_top, pad_right, pad_bottom = padding

    if isinstance(fill, numbers.Number):
        fill = fill,
    if padding_mode == 'constant':
        assert (len(fill) == 3 and len(img.shape) == 3) or (len(fill) == 1 and len(img.shape) == 2), \
            'channel of image is {} but length of fill is {}'.format(img.shape[-1], len(fill))

    img = cv2.copyMakeBorder(src=img, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
                             borderType=PAD_MOD[padding_mode], value=fill)
    return img


def crop(img, x, y, h, w):
    """Crop the given CV Image.

    Args:
        img (np.ndarray): Image to be cropped.
        x: Upper pixel coordinate.
        y: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.

    Returns:
        CV Image: Cropped image.
    """
    assert _is_numpy_image(img), 'img should be CV Image. Got {}'.format(type(img))
    assert h > 0 and w > 0, 'h={} and w={} should greater than 0'.format(h, w)

    x1, y1, x2, y2 = round(x), round(y), round(x+h), round(y+w)

    try:
        check_point1 = img[x1, y1, ...]
        check_point2 = img[x2-1, y2-1, ...]
    except IndexError:
        # warnings.warn('crop region is {} but image size is {}'.format((x1, y1, x2, y2), img.shape))
        img = cv2.copyMakeBorder(img, - min(0, x1), max(x2 - img.shape[0], 0),
                                 -min(0, y1), max(y2 - img.shape[1], 0), cv2.BORDER_CONSTANT, value=[0, 0, 0])
        y2 += -min(0, y1)
        y1 += -min(0, y1)
        x2 += -min(0, x1)
        x1 += -min(0, x1)

    finally:
        return img[x1:x2, y1:y2, ...].copy()


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w, _ = img.shape
    th, tw = output_size
    i = int(round((h - th) * 0.5))
    j = int(round((w - tw) * 0.5))
    return crop(img, i, j, th, tw)


def resized_crop(img, i, j, h, w, size, interpolation='BILINEAR'):
    """Crop the given CV Image and resize it to desired size. Notably used in RandomResizedCrop.

    Args:
        img (np.ndarray): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``scale``.
        interpolation (str, optional): Desired interpolation. Default is
            ``BILINEAR``.
    Returns:
        np.ndarray: Cropped image.
    """
    assert _is_numpy_image(img), 'img should be CV Image'
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation)
    return img


def hflip(img):
    """Horizontally flip the given PIL Image.

    Args:
        img (np.ndarray): Image to be flipped.

    Returns:
        np.ndarray:  Horizontall flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    return cv2.flip(img, 1)


def vflip(img):
    """Vertically flip the given PIL Image.

    Args:
        img (CV Image): Image to be flipped.

    Returns:
        PIL Image:  Vertically flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return cv2.flip(img, 0)


def five_crop(img, size):
    """Crop the given CV Image into four corners and the central crop.

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
        tuple: tuple (tl, tr, bl, br, center) corresponding top left,
            top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    h, w, _ = img.shape
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    tl = crop(img, 0, 0, crop_h, crop_w)
    tr = crop(img, 0, w - crop_w, crop_h, crop_w)
    bl = crop(img, h - crop_h, 0, crop_h, crop_w)
    br = crop(img, h - crop_h, w - crop_w, crop_h, crop_w)
    center = center_crop(img, (crop_h, crop_w))
    return (tl, tr, bl, br, center)


def ten_crop(img, size, vertical_flip=False):
    """Crop the given CV Image into four corners and the central crop plus the
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
            tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip,
                br_flip, center_flip) corresponding top left, top right,
                bottom left, bottom right and center crop and same for the
                flipped image.
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
        img (np.ndarray): CV Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        np.ndarray: Brightness adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    im = img.astype(np.float32) * brightness_factor
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.

    Args:
        img (np.ndarray): CV Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        np.ndarray: Contrast adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))
    im = img.astype(np.float32)
    mean = round(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).mean())
    im = (1-contrast_factor)*mean + contrast_factor * im
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.

    Args:
        img (np.ndarray): CV Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a gray image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        np.ndarray: Saturation adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    im = img.astype(np.float32)
    degenerate = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    im = (1-saturation_factor) * degenerate + saturation_factor * im
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See https://en.wikipedia.org/wiki/Hue for more details on Hue.

    Args:
        img (np.ndarray): CV Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        np.ndarray: Hue adjusted image.
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    im = img.astype(np.uint8)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV_FULL)
    hsv[..., 0] += np.uint8(hue_factor * 255)

    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
    return im.astype(img.dtype)


def adjust_gamma(img, gamma, gain=1):
    """Perform gamma correction on an image.

    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:

        I_out = 255 * gain * ((I_in / 255) ** gamma)

    See https://en.wikipedia.org/wiki/Gamma_correction for more details.

    Args:
        img (np.ndarray): CV Image to be adjusted.
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    im = img.astype(np.float32)
    im = 255. * gain * np.power(im / 255., gamma)
    im = im.clip(min=0., max=255.)
    return im.astype(img.dtype)


def to_grayscale(img, num_output_channels=1):
    """Convert image to grayscale version of image.

    Args:
        img (np.ndarray): Image to be converted to grayscale.

    Returns:
        CV Image:  Grayscale version of the image.
                    if num_output_channels == 1 : returned image is single channel
                    if num_output_channels == 3 : returned image is 3 channel with r == g == b
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    if num_output_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif num_output_channels == 3:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img


def rotate(img, angle, resample='BILINEAR', expand=False, center=None):
    """Rotate the image by angle.
    Args:
        img (PIL Image): PIL Image to be rotated.
        angle ({float, int}): In degrees clockwise order.
        resample ({NEAREST, BILINEAR, BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """
    imgtype = img.dtype
    if not _is_numpy_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    h, w, _ = img.shape
    point = center or (w/2, h/2)
    M = cv2.getRotationMatrix2D(point, angle=-angle, scale=1)

    if expand:
        if center is None:
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - point[0]
            M[1, 2] += (nH / 2) - point[1]

            # perform the actual rotation and return the image
            dst = cv2.warpAffine(img, M, (nW, nH))
        else:
            xx = []
            yy = []
            for point in (np.array([0, 0, 1]), np.array([w-1, 0, 1]), np.array([w-1, h-1, 1]), np.array([0, h-1, 1])):
                target = M@point
                xx.append(target[0])
                yy.append(target[1])
            nh = int(math.ceil(max(yy)) - math.floor(min(yy)))
            nw = int(math.ceil(max(xx)) - math.floor(min(xx)))
            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nw - w)/2
            M[1, 2] += (nh - h)/2
            dst = cv2.warpAffine(img, M, (nw, nh), flags=INTER_MODE[resample])
    else:
        dst = cv2.warpAffine(img, M, (w, h), flags=INTER_MODE[resample])
    return dst.astype(imgtype)


def affine6(img, angle, translate, scale, resample='BILINEAR', fillcolor=(0,0,0)):
    """Apply affine transformation on the image keeping image center invariant
    Args:
        img (np.ndarray): PIL Image to be rotated.
        angle (list or tuple): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float, or tuple): overall scale
        resample ({NEAREST, BILINEAR, BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int or tuple): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """
    rows, cols, _ = img.shape
    centery = rows * 0/5
    centerx = cols * 0.5

    alpha = math.radians(angle[0])
    beta = math.radians(angle[1])

    lambda1 = scale[0]
    lambda2 = scale[1]

    tx = translate[0]
    ty = translate[1]

    sina = math.sin(alpha)
    cosa = math.cos(alpha)
    sinb = math.sin(beta)
    cosb = math.cos(beta)

    M00 = cosb * (lambda1 * cosa ** 2 + lambda2 * sina ** 2) - sinb * (lambda2 - lambda1) * sina * cosa
    M01 = - sinb * (lambda1 * sina ** 2 + lambda2 * cosa ** 2) + cosb * (lambda2 - lambda1) * sina * cosa

    M10 = sinb * (lambda1 * cosa ** 2 + lambda2 * sina ** 2) + cosb * (lambda2 - lambda1) * sina * cosa
    M11 = + cosb * (lambda1 * sina ** 2 + lambda2 * cosa ** 2) + sinb * (lambda2 - lambda1) * sina * cosa
    M02 = centerx - M00 * centerx - M01 * centery + tx
    M12 = centery - M10 * centerx - M11 * centery + ty
    affine_matrix = np.array([[M00, M01, M02], [M10, M11, M12]], dtype=np.float32)

    dst_img = cv2.warpAffine(img, affine_matrix, (cols, rows), flags=INTER_MODE[resample],
                             borderMode=cv2.BORDER_CONSTANT, borderValue=fillcolor)
    return dst_img


def affine(img, angle=0, translate=(0, 0), scale=1, shear=0, resample='BILINEAR', fillcolor=(0,0,0)):
    """Apply affine transformation on the image keeping image center invariant
    Args:
        img (np.ndarray): PIL Image to be rotated.
        angle ({float, int}): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float): shear angle value in degrees between -180 to 180, clockwise direction.
        resample ({NEAREST, BILINEAR, BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int or tuple): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    rows, cols, _ = img.shape
    center = (cols * 0.5, rows * 0.5)
    angle = math.radians(angle)
    shear = math.radians(shear)
    M00 = math.cos(angle)*scale
    M01 = -math.sin(angle+shear)*scale
    M10 = math.sin(angle)*scale
    M11 = math.cos(angle+shear)*scale
    M02 = center[0] - center[0]*M00 - center[1]*M01 + translate[0]
    M12 = center[1] - center[0]*M10 - center[1]*M11 + translate[1]
    affine_matrix = np.array([[M00, M01, M02], [M10, M11, M12]], dtype=np.float32)
    dst_img = cv2.warpAffine(img, affine_matrix, (cols, rows), flags=INTER_MODE[resample],
                             borderMode=cv2.BORDER_CONSTANT, borderValue=fillcolor)
    return dst_img


def cv_transform(img):
    # img = resize(img, size=(100, 300))
    # img = to_tensor(img)
    # img = normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # img = pad(img, padding=(10, 10, 20, 20), fill=(255, 255, 255), padding_mode='constant')
    # img = pad(img, padding=(100, 100, 100, 100), fill=5, padding_mode='symmetric')
    # img = crop(img, -40, -20, 1000, 1000)
    # img = center_crop(img, (310, 300))
    # img = resized_crop(img, -10.3, -20, 330, 220, (500, 500))
    # img = hflip(img)
    # img = vflip(img)
    # tl, tr, bl, br, center = five_crop(img, 100)
    # img = adjust_brightness(img, 2.1)
    # img = adjust_contrast(img, 1.5)
    # img = adjust_saturation(img, 2.3)
    # img = adjust_hue(img, 0.5)
    # img = adjust_gamma(img, gamma=3, gain=0.1)
    # img = rotate(img, 10, resample='BILINEAR', expand=True, center=None)
    # img = to_grayscale(img, 3)
    img = affine(img, 10, (0, 0), 1, 0, resample='BICUBIC', fillcolor=(255,255,0))
    return to_tensor(img)


def pil_transform(img):
    # img = functional.resize(img, size=(100, 300))
    # img = functional.to_tensor(img)
    # img = functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # img = functional.pad(img, padding=(10, 10, 20, 20), fill=(255, 255, 255), padding_mode='constant')
    # img = functional.pad(img, padding=(100, 100, 100, 100), padding_mode='symmetric')
    # img = functional.crop(img, -40, -20, 1000, 1000)
    # img = functional.center_crop(img, (310, 300))
    # img = functional.resized_crop(img, -10.3, -20, 330, 220, (500, 500))
    # img = functional.hflip(img)
    # img = functional.vflip(img)
    # tl, tr, bl, br, center = functional.five_crop(img, 100)
    # img = functional.adjust_brightness(img, 2.1)
    # img = functional.adjust_contrast(img, 1.5)
    # img = functional.adjust_saturation(img, 2.3)
    # img = functional.adjust_hue(img, 0.5)
    # img = functional.adjust_gamma(img, gamma=3, gain=0.1)
    # img = functional.rotate(img, 10, resample=PIL.Image.BILINEAR, expand=True, center=None)
    # img = functional.to_grayscale(img, 3)
    img = functional.affine(img, 10, (0, 0), 1, 0, resample=PIL.Image.BICUBIC, fillcolor=(255,255,0))

    return functional.to_tensor(img)


if __name__ == '__main__':
    image_path = '../../cat.jpg'

    cvimage = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cvimage = cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB)
    cvimage = cv_transform(cvimage)

    pilimage = Image.open(image_path).convert('RGB')
    pilimage = pil_transform(pilimage)

    # sub = abs(cvimage - pilimage)

    # imshow((cvimage, pilimage, sub), ('CV', 'PIL', 'sub'))
    imshow((cvimage, pilimage), ('CV', 'PIL'))
    # imshow([pilimage], ('PIL'))