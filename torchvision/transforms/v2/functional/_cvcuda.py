import functools
from typing import Any

import torch
from torch.torch_version import TorchVersion

from ._meta import _validate_cvcuda_image
from ._utils import _register_kernel_internal


@functools.lru_cache(maxsize=1)
def _is_cvcuda_available() -> bool:
    try:
        import cvcuda
    except ModuleNotFoundError as error:
        if error.name != "cvcuda":
            raise
        return False
    return TorchVersion(getattr(cvcuda, "__version__", "0")) >= "0.19.0"


def _import_cvcuda():
    if not _is_cvcuda_available():
        raise ImportError("CV-CUDA >= 0.19.0 with cvcuda.tv is required for CV-CUDA transforms.")
    import cvcuda

    return cvcuda


def _is_cvcuda_tensor(inpt: Any) -> bool:
    return _is_cvcuda_available() and isinstance(inpt, _import_cvcuda().Tensor)


def _register_cvcuda_kernel(functional, kernel):
    """Register a cvcuda.tv kernel on PyTorch's current stream, without changing Torch dispatch."""

    @functools.wraps(kernel)
    def wrapper(inpt, *args, **kwargs):
        _validate_cvcuda_image(inpt)
        device = inpt.cuda().__dlpack_device__()[1]
        with torch.cuda.device(device):
            stream = _import_cvcuda().as_stream(torch.cuda.current_stream(device).cuda_stream)
            return kernel(inpt, *args, **kwargs, stream=stream)

    _register_kernel_internal(functional, _import_cvcuda().Tensor)(wrapper)
    return wrapper


if _is_cvcuda_available():
    from torchvision.transforms.v2 import functional as F

    # Augment
    _register_cvcuda_kernel(F.erase, _import_cvcuda().tv.erase)
    _register_cvcuda_kernel(F.jpeg, _import_cvcuda().tv.jpeg)

    # Color
    _register_cvcuda_kernel(F.adjust_brightness, _import_cvcuda().tv.adjust_brightness)
    _register_cvcuda_kernel(F.adjust_contrast, _import_cvcuda().tv.adjust_contrast)
    _register_cvcuda_kernel(F.adjust_gamma, _import_cvcuda().tv.adjust_gamma)
    _register_cvcuda_kernel(F.adjust_hue, _import_cvcuda().tv.adjust_hue)
    _register_cvcuda_kernel(F.adjust_saturation, _import_cvcuda().tv.adjust_saturation)
    _register_cvcuda_kernel(F.adjust_sharpness, _import_cvcuda().tv.adjust_sharpness)
    _register_cvcuda_kernel(F.autocontrast, _import_cvcuda().tv.autocontrast)
    _register_cvcuda_kernel(F.equalize, _import_cvcuda().tv.equalize)
    _register_cvcuda_kernel(F.grayscale_to_rgb, _import_cvcuda().tv.grayscale_to_rgb)
    _register_cvcuda_kernel(F.invert, _import_cvcuda().tv.invert)
    _register_cvcuda_kernel(F.permute_channels, _import_cvcuda().tv.permute_channels)
    _register_cvcuda_kernel(F.posterize, _import_cvcuda().tv.posterize)
    _register_cvcuda_kernel(F.rgb_to_grayscale, _import_cvcuda().tv.rgb_to_grayscale)
    _register_cvcuda_kernel(F.solarize, _import_cvcuda().tv.solarize)

    # Geometry
    _register_cvcuda_kernel(F.affine, _import_cvcuda().tv.affine)
    _register_cvcuda_kernel(F.center_crop, _import_cvcuda().tv.center_crop)
    _register_cvcuda_kernel(F.crop, _import_cvcuda().tv.crop)
    _register_cvcuda_kernel(F.elastic, _import_cvcuda().tv.elastic)
    _register_cvcuda_kernel(F.five_crop, _import_cvcuda().tv.five_crop)
    _register_cvcuda_kernel(F.horizontal_flip, _import_cvcuda().tv.horizontal_flip)
    _register_cvcuda_kernel(F.pad, _import_cvcuda().tv.pad)
    _register_cvcuda_kernel(F.perspective, _import_cvcuda().tv.perspective)
    _register_cvcuda_kernel(F.resize, _import_cvcuda().tv.resize)
    _register_cvcuda_kernel(F.resized_crop, _import_cvcuda().tv.resized_crop)
    _register_cvcuda_kernel(F.rotate, _import_cvcuda().tv.rotate)
    _register_cvcuda_kernel(F.ten_crop, _import_cvcuda().tv.ten_crop)
    _register_cvcuda_kernel(F.vertical_flip, _import_cvcuda().tv.vertical_flip)

    # Misc
    _register_cvcuda_kernel(F.gaussian_blur, _import_cvcuda().tv.gaussian_blur)
    _register_cvcuda_kernel(F.gaussian_noise, _import_cvcuda().tv.gaussian_noise)
    _register_cvcuda_kernel(F.normalize, _import_cvcuda().tv.normalize)
    _register_cvcuda_kernel(F.to_dtype, _import_cvcuda().tv.to_dtype)

    # Metadata
    _register_kernel_internal(F.get_dimensions, _import_cvcuda().Tensor)(F._meta._get_dimensions_image_cvcuda)
    _register_kernel_internal(F.get_num_channels, _import_cvcuda().Tensor)(F._meta._get_num_channels_image_cvcuda)
    _register_kernel_internal(F.get_size, _import_cvcuda().Tensor)(F._meta._get_size_image_cvcuda)
