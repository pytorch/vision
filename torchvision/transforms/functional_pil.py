import numbers
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance

try:
    import accimage
except ImportError:
    accimage = None


@torch.jit.unused
def _is_pil_image(img: Any) -> bool:
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


@torch.jit.unused
def get_image_size(img: Any) -> List[int]:
    if _is_pil_image(img):
        return list(img.size)
    raise TypeError("Unexpected type {}".format(type(img)))


@torch.jit.unused
def get_image_num_channels(img: Any) -> int:
    if _is_pil_image(img):
        return 1 if img.mode == 'L' else 3
    raise TypeError("Unexpected type {}".format(type(img)))


@torch.jit.unused
def hflip(img: Image.Image) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_LEFT_RIGHT)


@torch.jit.unused
def vflip(img: Image.Image) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_TOP_BOTTOM)


@torch.jit.unused
def adjust_brightness(img: Image.Image, brightness_factor: float) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


@torch.jit.unused
def adjust_contrast(img: Image.Image, contrast_factor: float) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


@torch.jit.unused
def adjust_saturation(img: Image.Image, saturation_factor: float) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


@torch.jit.unused
def adjust_hue(img: Image.Image, hue_factor: float) -> Image.Image:
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


@torch.jit.unused
def adjust_gamma(
    img: Image.Image,
    gamma: float,
    gain: float = 1.0,
) -> Image.Image:

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    input_mode = img.mode
    img = img.convert('RGB')
    gamma_map = [(255 + 1 - 1e-3) * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
    img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

    img = img.convert(input_mode)
    return img


@torch.jit.unused
def pad(
    img: Image.Image,
    padding: Union[int, List[int], Tuple[int, ...]],
    fill: Optional[Union[float, List[float], Tuple[float, ...]]] = 0,
    padding_mode: str = "constant",
) -> Image.Image:

    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError("Got inappropriate fill arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, list):
        padding = tuple(padding)

    if isinstance(padding, tuple) and len(padding) not in [1, 2, 4]:
        raise ValueError("Padding must be an int or a 1, 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    if isinstance(padding, tuple) and len(padding) == 1:
        # Compatibility with `functional_tensor.pad`
        padding = padding[0]

    if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
        raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

    if padding_mode == "constant":
        opts = _parse_fill(fill, img, name="fill")
        if img.mode == "P":
            palette = img.getpalette()
            image = ImageOps.expand(img, border=padding, **opts)
            image.putpalette(palette)
            return image

        return ImageOps.expand(img, border=padding, **opts)
    else:
        if isinstance(padding, int):
            pad_left = pad_right = pad_top = pad_bottom = padding
        if isinstance(padding, tuple) and len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        if isinstance(padding, tuple) and len(padding) == 4:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]

        p = [pad_left, pad_top, pad_right, pad_bottom]
        cropping = -np.minimum(p, 0)

        if cropping.any():
            crop_left, crop_top, crop_right, crop_bottom = cropping
            img = img.crop((crop_left, crop_top, img.width - crop_right, img.height - crop_bottom))

        pad_left, pad_top, pad_right, pad_bottom = np.maximum(p, 0)

        if img.mode == 'P':
            palette = img.getpalette()
            img = np.asarray(img)
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)
            img = Image.fromarray(img)
            img.putpalette(palette)
            return img

        img = np.asarray(img)
        # RGB image
        if len(img.shape) == 3:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode)
        # Grayscale image
        if len(img.shape) == 2:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)

        return Image.fromarray(img)


@torch.jit.unused
def crop(
    img: Image.Image,
    top: int,
    left: int,
    height: int,
    width: int,
) -> Image.Image:

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((left, top, left + width, top + height))


@torch.jit.unused
def resize(
    img: Image.Image,
    size: Union[Sequence[int], int],
    interpolation: int = Image.BILINEAR,
    max_size: Optional[int] = None,
) -> Image.Image:

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Sequence) and len(size) in (1, 2))):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, Sequence) and len(size) == 1:
        size = size[0]
    if isinstance(size, int):
        w, h = img.size

        short, long = (w, h) if w <= h else (h, w)
        if short == size:
            return img

        new_short, new_long = size, int(size * long / short)

        if max_size is not None:
            if max_size <= size:
                raise ValueError(
                    f"max_size = {max_size} must be strictly greater than the requested "
                    f"size for the smaller edge size = {size}"
                )
            if new_long > max_size:
                new_short, new_long = int(max_size * new_short / new_long), max_size

        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
        return img.resize((new_w, new_h), interpolation)
    else:
        if max_size is not None:
            raise ValueError(
                "max_size should only be passed if size specifies the length of the smaller edge, "
                "i.e. size should be an int or a sequence of length 1 in torchscript mode."
            )
        return img.resize(size[::-1], interpolation)


@torch.jit.unused
def _parse_fill(
    fill: Optional[Union[float, List[float], Tuple[float, ...]]],
    img: Image.Image,
    name: str = "fillcolor",
) -> Dict[str, Optional[Union[float, List[float], Tuple[float, ...]]]]:

    # Process fill color for affine transforms
    num_bands = len(img.getbands())
    if fill is None:
        fill = 0
    if isinstance(fill, (int, float)) and num_bands > 1:
        fill = tuple([fill] * num_bands)
    if isinstance(fill, (list, tuple)):
        if len(fill) != num_bands:
            msg = ("The number of elements in 'fill' does not match the number of "
                   "bands of the image ({} != {})")
            raise ValueError(msg.format(len(fill), num_bands))

        fill = tuple(fill)

    return {name: fill}


@torch.jit.unused
def affine(
    img: Image.Image,
    matrix: List[float],
    interpolation: int = Image.NEAREST,
    fill: Optional[Union[float, List[float], Tuple[float, ...]]] = 0,
) -> Image.Image:

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    output_size = img.size
    opts = _parse_fill(fill, img)
    return img.transform(output_size, Image.AFFINE, matrix, interpolation, **opts)


@torch.jit.unused
def rotate(
    img: Image.Image,
    angle: float,
    interpolation: int = Image.NEAREST,
    expand: bool = False,
    center: Optional[Tuple[int, int]] = None,
    fill: Optional[Union[float, List[float], Tuple[float, ...]]] = 0,
) -> Image.Image:

    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    opts = _parse_fill(fill, img)
    return img.rotate(angle, interpolation, expand, center, **opts)


@torch.jit.unused
def perspective(
    img: Image.Image,
    perspective_coeffs: float,
    interpolation: int = Image.BICUBIC,
    fill: Optional[Union[float, List[float], Tuple[float, ...]]] = 0,
) -> Image.Image:

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    opts = _parse_fill(fill, img)

    return img.transform(img.size, Image.PERSPECTIVE, perspective_coeffs, interpolation, **opts)


@torch.jit.unused
def to_grayscale(img: Image.Image, num_output_channels: int) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if num_output_channels == 1:
        img = img.convert('L')
    elif num_output_channels == 3:
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img


@torch.jit.unused
def invert(img: Image.Image) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    return ImageOps.invert(img)


@torch.jit.unused
def posterize(img: Image.Image, bits: int) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    return ImageOps.posterize(img, bits)


@torch.jit.unused
def solarize(img: Image.Image, threshold: int) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    return ImageOps.solarize(img, threshold)


@torch.jit.unused
def adjust_sharpness(img: Image.Image, sharpness_factor: float) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(sharpness_factor)
    return img


@torch.jit.unused
def autocontrast(img: Image.Image) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    return ImageOps.autocontrast(img)


@torch.jit.unused
def equalize(img: Image.Image) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    return ImageOps.equalize(img)
