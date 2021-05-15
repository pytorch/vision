import warnings

import torch
from torch import Tensor
from torch.nn.functional import grid_sample, conv2d, interpolate, pad as torch_pad
from torch.jit.annotations import BroadcastingList2
from typing import Optional, Tuple, List


def _is_tensor_a_torch_image(x: Tensor) -> bool:
    return x.ndim >= 2


def _assert_image_tensor(img):
    if not _is_tensor_a_torch_image(img):
        raise TypeError("Tensor is not a torch image.")


def _get_image_size(img: Tensor) -> List[int]:
    # Returns (w, h) of tensor image
    _assert_image_tensor(img)
    return [img.shape[-1], img.shape[-2]]


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
            break
    return max_value.item()


def _assert_channels(img: Tensor, permitted: List[int]) -> None:
    c = _get_image_num_channels(img)
    if c not in permitted:
        raise TypeError("Input image tensor permitted channel values are {}, but found {}".format(permitted, c))


def convert_image_dtype(image: torch.Tensor, dtype: torch.dtype = torch.float) -> torch.Tensor:
    if image.dtype == dtype:
        return image

    if image.is_floating_point():

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

        # int to float
        # TODO: replace with dtype.is_floating_point when torchscript supports it
        if torch.tensor(0, dtype=dtype).is_floating_point():
            image = image.to(dtype)
            return image / input_max

        output_max = _max_value(dtype)

        # int to int
        if input_max > output_max:
            # factor should be forced to int for torch jit script
            # otherwise factor is a float and image // factor can produce different results
            factor = int((input_max + 1) // (output_max + 1))
            image = torch.div(image, factor, rounding_mode='floor')
            return image.to(dtype)
        else:
            # factor should be forced to int for torch jit script
            # otherwise factor is a float and image * factor can produce different results
            factor = int((output_max + 1) // (input_max + 1))
            image = image.to(dtype)
            return image * factor


def vflip(img: Tensor) -> Tensor:
    _assert_image_tensor(img)

    return img.flip(-2)


def hflip(img: Tensor) -> Tensor:
    _assert_image_tensor(img)

    return img.flip(-1)


def crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    _assert_image_tensor(img)

    return img[..., top:top + height, left:left + width]


def rgb_to_grayscale(img: Tensor, num_output_channels: int = 1) -> Tensor:
    if img.ndim < 3:
        raise TypeError("Input image tensor should have at least 3 dimensions, but found {}".format(img.ndim))
    _assert_channels(img, [3])

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
    if brightness_factor < 0:
        raise ValueError('brightness_factor ({}) is not non-negative.'.format(brightness_factor))

    _assert_image_tensor(img)

    _assert_channels(img, [1, 3])

    return _blend(img, torch.zeros_like(img), brightness_factor)


def adjust_contrast(img: Tensor, contrast_factor: float) -> Tensor:
    if contrast_factor < 0:
        raise ValueError('contrast_factor ({}) is not non-negative.'.format(contrast_factor))

    _assert_image_tensor(img)

    _assert_channels(img, [3])

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    mean = torch.mean(rgb_to_grayscale(img).to(dtype), dim=(-3, -2, -1), keepdim=True)

    return _blend(img, mean, contrast_factor)


def adjust_hue(img: Tensor, hue_factor: float) -> Tensor:
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))

    if not (isinstance(img, torch.Tensor)):
        raise TypeError('Input img should be Tensor image')

    _assert_image_tensor(img)

    _assert_channels(img, [1, 3])
    if _get_image_num_channels(img) == 1:  # Match PIL behaviour
        return img

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
    if saturation_factor < 0:
        raise ValueError('saturation_factor ({}) is not non-negative.'.format(saturation_factor))

    _assert_image_tensor(img)

    _assert_channels(img, [3])

    return _blend(img, rgb_to_grayscale(img), saturation_factor)


def adjust_gamma(img: Tensor, gamma: float, gain: float = 1) -> Tensor:
    if not isinstance(img, torch.Tensor):
        raise TypeError('Input img should be a Tensor.')

    _assert_channels(img, [1, 3])

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    result = img
    dtype = img.dtype
    if not torch.is_floating_point(img):
        result = convert_image_dtype(result, torch.float32)

    result = (gain * result ** gamma).clamp(0, 1)

    result = convert_image_dtype(result, dtype)
    return result


def center_crop(img: Tensor, output_size: BroadcastingList2[int]) -> Tensor:
    """DEPRECATED
    """
    warnings.warn(
        "This method is deprecated and will be removed in future releases. "
        "Please, use ``F.center_crop`` instead."
    )

    _assert_image_tensor(img)

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
    """DEPRECATED
    """
    warnings.warn(
        "This method is deprecated and will be removed in future releases. "
        "Please, use ``F.five_crop`` instead."
    )

    _assert_image_tensor(img)

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
    """DEPRECATED
    """
    warnings.warn(
        "This method is deprecated and will be removed in future releases. "
        "Please, use ``F.ten_crop`` instead."
    )

    _assert_image_tensor(img)

    assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
    first_five = five_crop(img, size)

    if vertical_flip:
        img = vflip(img)
    else:
        img = hflip(img)

    second_five = five_crop(img, size)

    return first_five + second_five


def _blend(img1: Tensor, img2: Tensor, ratio: float) -> Tensor:
    ratio = float(ratio)
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
    _assert_image_tensor(img)

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


def resize(
    img: Tensor,
    size: List[int],
    interpolation: str = "bilinear",
    max_size: Optional[int] = None,
    antialias: Optional[bool] = None
) -> Tensor:
    _assert_image_tensor(img)

    if not isinstance(size, (int, tuple, list)):
        raise TypeError("Got inappropriate size arg")
    if not isinstance(interpolation, str):
        raise TypeError("Got inappropriate interpolation arg")

    if interpolation not in ["nearest", "bilinear", "bicubic"]:
        raise ValueError("This interpolation mode is unsupported with Tensor input")

    if isinstance(size, tuple):
        size = list(size)

    if isinstance(size, list):
        if len(size) not in [1, 2]:
            raise ValueError("Size must be an int or a 1 or 2 element tuple/list, not a "
                             "{} element tuple/list".format(len(size)))
        if max_size is not None and len(size) != 1:
            raise ValueError(
                "max_size should only be passed if size specifies the length of the smaller edge, "
                "i.e. size should be an int or a sequence of length 1 in torchscript mode."
            )

    if antialias is None:
        antialias = False

    if antialias and interpolation not in ["bilinear", "bicubic"]:
        raise ValueError("Antialias option is supported for bilinear and bicubic interpolation modes only")

    w, h = _get_image_size(img)

    if isinstance(size, int) or len(size) == 1:  # specified size only for the smallest edge
        short, long = (w, h) if w <= h else (h, w)
        requested_new_short = size if isinstance(size, int) else size[0]

        if short == requested_new_short:
            return img

        new_short, new_long = requested_new_short, int(requested_new_short * long / short)

        if max_size is not None:
            if max_size <= requested_new_short:
                raise ValueError(
                    f"max_size = {max_size} must be strictly greater than the requested "
                    f"size for the smaller edge size = {size}"
                )
            if new_long > max_size:
                new_short, new_long = int(max_size * new_short / new_long), max_size

        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)

    else:  # specified both h and w
        new_w, new_h = size[1], size[0]

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [torch.float32, torch.float64])

    # Define align_corners to avoid warnings
    align_corners = False if interpolation in ["bilinear", "bicubic"] else None

    if antialias:
        if interpolation == "bilinear":
            img = torch.ops.torchvision._interpolate_linear_aa(img, [new_h, new_w], align_corners=False)
        elif interpolation == "bicubic":
            img = torch.ops.torchvision._interpolate_bicubic_aa(img, [new_h, new_w], align_corners=False)
    else:
        img = interpolate(img, size=[new_h, new_w], mode=interpolation, align_corners=align_corners)

    if interpolation == "bicubic" and out_dtype == torch.uint8:
        img = img.clamp(min=0, max=255)

    img = _cast_squeeze_out(img, need_cast=need_cast, need_squeeze=need_squeeze, out_dtype=out_dtype)

    return img


def _assert_grid_transform_inputs(
        img: Tensor,
        matrix: Optional[List[float]],
        interpolation: str,
        fill: Optional[List[float]],
        supported_interpolation_modes: List[str],
        coeffs: Optional[List[float]] = None,
):

    if not (isinstance(img, torch.Tensor)):
        raise TypeError("Input img should be Tensor")

    _assert_image_tensor(img)

    if matrix is not None and not isinstance(matrix, list):
        raise TypeError("Argument matrix should be a list")

    if matrix is not None and len(matrix) != 6:
        raise ValueError("Argument matrix should have 6 float values")

    if coeffs is not None and len(coeffs) != 8:
        raise ValueError("Argument coeffs should have 8 float values")

    if fill is not None and not isinstance(fill, (int, float, tuple, list)):
        warnings.warn("Argument fill should be either int, float, tuple or list")

    # Check fill
    num_channels = _get_image_num_channels(img)
    if isinstance(fill, (tuple, list)) and (len(fill) > 1 and len(fill) != num_channels):
        msg = ("The number of elements in 'fill' cannot broadcast to match the number of "
               "channels of the image ({} != {})")
        raise ValueError(msg.format(len(fill), num_channels))

    if interpolation not in supported_interpolation_modes:
        raise ValueError("Interpolation mode '{}' is unsupported with Tensor input".format(interpolation))


def _cast_squeeze_in(img: Tensor, req_dtypes: List[torch.dtype]) -> Tuple[Tensor, bool, bool, torch.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype):
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # it is better to round before cast
            img = torch.round(img)
        img = img.to(out_dtype)

    return img


def _apply_grid_transform(img: Tensor, grid: Tensor, mode: str, fill: Optional[List[float]]) -> Tensor:

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [grid.dtype, ])

    if img.shape[0] > 1:
        # Apply same grid to a batch of images
        grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])

    # Append a dummy mask for customized fill colors, should be faster than grid_sample() twice
    if fill is not None:
        dummy = torch.ones((img.shape[0], 1, img.shape[2], img.shape[3]), dtype=img.dtype, device=img.device)
        img = torch.cat((img, dummy), dim=1)

    img = grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)

    # Fill with required color
    if fill is not None:
        mask = img[:, -1:, :, :]  # N * 1 * H * W
        img = img[:, :-1, :, :]  # N * C * H * W
        mask = mask.expand_as(img)
        len_fill = len(fill) if isinstance(fill, (tuple, list)) else 1
        fill_img = torch.tensor(fill, dtype=img.dtype, device=img.device).view(1, len_fill, 1, 1).expand_as(img)
        if mode == 'nearest':
            mask = mask < 0.5
            img[mask] = fill_img[mask]
        else:  # 'bilinear'
            img = img * mask + (1.0 - mask) * fill_img

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
    x_grid = torch.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow, device=theta.device)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh, device=theta.device).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta = theta.transpose(1, 2) / torch.tensor([0.5 * w, 0.5 * h], dtype=theta.dtype, device=theta.device)
    output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
    return output_grid.view(1, oh, ow, 2)


def affine(
        img: Tensor, matrix: List[float], interpolation: str = "nearest", fill: Optional[List[float]] = None
) -> Tensor:
    _assert_grid_transform_inputs(img, matrix, interpolation, fill, ["nearest", "bilinear"])

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    theta = torch.tensor(matrix, dtype=dtype, device=img.device).reshape(1, 2, 3)
    shape = img.shape
    # grid will be generated on the same device as theta and img
    grid = _gen_affine_grid(theta, w=shape[-1], h=shape[-2], ow=shape[-1], oh=shape[-2])
    return _apply_grid_transform(img, grid, interpolation, fill=fill)


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
    img: Tensor, matrix: List[float], interpolation: str = "nearest",
    expand: bool = False, fill: Optional[List[float]] = None
) -> Tensor:
    _assert_grid_transform_inputs(img, matrix, interpolation, fill, ["nearest", "bilinear"])
    w, h = img.shape[-1], img.shape[-2]
    ow, oh = _compute_output_size(matrix, w, h) if expand else (w, h)
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    theta = torch.tensor(matrix, dtype=dtype, device=img.device).reshape(1, 2, 3)
    # grid will be generated on the same device as theta and img
    grid = _gen_affine_grid(theta, w=w, h=h, ow=ow, oh=oh)

    return _apply_grid_transform(img, grid, interpolation, fill=fill)


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
    x_grid = torch.linspace(d, ow * 1.0 + d - 1.0, steps=ow, device=device)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(d, oh * 1.0 + d - 1.0, steps=oh, device=device).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta1 = theta1.transpose(1, 2) / torch.tensor([0.5 * ow, 0.5 * oh], dtype=dtype, device=device)
    output_grid1 = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta1)
    output_grid2 = base_grid.view(1, oh * ow, 3).bmm(theta2.transpose(1, 2))

    output_grid = output_grid1 / output_grid2 - 1.0
    return output_grid.view(1, oh, ow, 2)


def perspective(
    img: Tensor, perspective_coeffs: List[float], interpolation: str = "bilinear", fill: Optional[List[float]] = None
) -> Tensor:
    if not (isinstance(img, torch.Tensor)):
        raise TypeError('Input img should be Tensor.')

    _assert_image_tensor(img)

    _assert_grid_transform_inputs(
        img,
        matrix=None,
        interpolation=interpolation,
        fill=fill,
        supported_interpolation_modes=["nearest", "bilinear"],
        coeffs=perspective_coeffs
    )

    ow, oh = img.shape[-1], img.shape[-2]
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    grid = _perspective_grid(perspective_coeffs, ow=ow, oh=oh, dtype=dtype, device=img.device)
    return _apply_grid_transform(img, grid, interpolation, fill=fill)


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
    if not (isinstance(img, torch.Tensor)):
        raise TypeError('img should be Tensor. Got {}'.format(type(img)))

    _assert_image_tensor(img)

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [kernel.dtype, ])

    # padding = (left, right, top, bottom)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img = torch_pad(img, padding, mode="reflect")
    img = conv2d(img, kernel, groups=img.shape[-3])

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


def invert(img: Tensor) -> Tensor:

    _assert_image_tensor(img)

    if img.ndim < 3:
        raise TypeError("Input image tensor should have at least 3 dimensions, but found {}".format(img.ndim))

    _assert_channels(img, [1, 3])

    bound = torch.tensor(1 if img.is_floating_point() else 255, dtype=img.dtype, device=img.device)
    return bound - img


def posterize(img: Tensor, bits: int) -> Tensor:

    _assert_image_tensor(img)

    if img.ndim < 3:
        raise TypeError("Input image tensor should have at least 3 dimensions, but found {}".format(img.ndim))
    if img.dtype != torch.uint8:
        raise TypeError("Only torch.uint8 image tensors are supported, but found {}".format(img.dtype))

    _assert_channels(img, [1, 3])
    mask = -int(2**(8 - bits))  # JIT-friendly for: ~(2 ** (8 - bits) - 1)
    return img & mask


def solarize(img: Tensor, threshold: float) -> Tensor:

    _assert_image_tensor(img)

    if img.ndim < 3:
        raise TypeError("Input image tensor should have at least 3 dimensions, but found {}".format(img.ndim))

    _assert_channels(img, [1, 3])

    inverted_img = invert(img)
    return torch.where(img >= threshold, inverted_img, img)


def _blurred_degenerate_image(img: Tensor) -> Tensor:
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32

    kernel = torch.ones((3, 3), dtype=dtype, device=img.device)
    kernel[1, 1] = 5.0
    kernel /= kernel.sum()
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    result_tmp, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [kernel.dtype, ])
    result_tmp = conv2d(result_tmp, kernel, groups=result_tmp.shape[-3])
    result_tmp = _cast_squeeze_out(result_tmp, need_cast, need_squeeze, out_dtype)

    result = img.clone()
    result[..., 1:-1, 1:-1] = result_tmp

    return result


def adjust_sharpness(img: Tensor, sharpness_factor: float) -> Tensor:
    if sharpness_factor < 0:
        raise ValueError('sharpness_factor ({}) is not non-negative.'.format(sharpness_factor))

    _assert_image_tensor(img)

    _assert_channels(img, [1, 3])

    if img.size(-1) <= 2 or img.size(-2) <= 2:
        return img

    return _blend(img, _blurred_degenerate_image(img), sharpness_factor)


def autocontrast(img: Tensor) -> Tensor:

    _assert_image_tensor(img)

    if img.ndim < 3:
        raise TypeError("Input image tensor should have at least 3 dimensions, but found {}".format(img.ndim))

    _assert_channels(img, [1, 3])

    bound = 1.0 if img.is_floating_point() else 255.0
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32

    minimum = img.amin(dim=(-2, -1), keepdim=True).to(dtype)
    maximum = img.amax(dim=(-2, -1), keepdim=True).to(dtype)
    eq_idxs = torch.where(minimum == maximum)[0]
    minimum[eq_idxs] = 0
    maximum[eq_idxs] = bound
    scale = bound / (maximum - minimum)

    return ((img - minimum) * scale).clamp(0, bound).to(img.dtype)


def _scale_channel(img_chan):
    # TODO: we should expect bincount to always be faster than histc, but this
    # isn't always the case. Once
    # https://github.com/pytorch/pytorch/issues/53194 is fixed, remove the if
    # block and only use bincount.
    if img_chan.is_cuda:
        hist = torch.histc(img_chan.to(torch.float32), bins=256, min=0, max=255)
    else:
        hist = torch.bincount(img_chan.view(-1), minlength=256)

    nonzero_hist = hist[hist != 0]
    step = torch.div(nonzero_hist[:-1].sum(), 255, rounding_mode='floor')
    if step == 0:
        return img_chan

    lut = torch.div(
        torch.cumsum(hist, 0) + torch.div(step, 2, rounding_mode='floor'),
        step, rounding_mode='floor')
    lut = torch.nn.functional.pad(lut, [1, 0])[:-1].clamp(0, 255)

    return lut[img_chan.to(torch.int64)].to(torch.uint8)


def _equalize_single_image(img: Tensor) -> Tensor:
    return torch.stack([_scale_channel(img[c]) for c in range(img.size(0))])


def equalize(img: Tensor) -> Tensor:

    _assert_image_tensor(img)

    if not (3 <= img.ndim <= 4):
        raise TypeError("Input image tensor should have 3 or 4 dimensions, but found {}".format(img.ndim))
    if img.dtype != torch.uint8:
        raise TypeError("Only torch.uint8 image tensors are supported, but found {}".format(img.dtype))

    _assert_channels(img, [1, 3])

    if img.ndim == 3:
        return _equalize_single_image(img)

    return torch.stack([_equalize_single_image(x) for x in img])
