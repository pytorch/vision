import torch
try:
    import accimage
except ImportError:
    accimage = None
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION


@torch.jit.unused
def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


@torch.jit.unused
def hflip(img):
    """Horizontally flip the given PIL Image.

    Args:
        img (PIL Image): Image to be flipped.

    Returns:
        PIL Image:  Horizontally flipped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_LEFT_RIGHT)


@torch.jit.unused
def vflip(img):
    """Vertically flip the given PIL Image.

    Args:
        img (PIL Image): Image to be flipped.

    Returns:
        PIL Image:  Vertically flipped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_TOP_BOTTOM)
