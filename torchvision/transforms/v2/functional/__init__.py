from torchvision.transforms import InterpolationMode  # usort: skip

from ._utils import is_simple_tensor, register_kernel  # usort: skip

from ._meta import (
    clamp_bounding_box,
    convert_format_bounding_box,
    convert_image_dtype,
    to_dtype,
    to_dtype_image_tensor,
    to_dtype_video,
    get_dimensions_image_tensor,
    get_dimensions_image_pil,
    get_dimensions,
    get_num_frames_video,
    get_num_frames,
    get_image_num_channels,
    get_num_channels_image_tensor,
    get_num_channels_image_pil,
    get_num_channels_video,
    get_num_channels,
    get_spatial_size_bounding_box,
    get_spatial_size_image_tensor,
    get_spatial_size_image_pil,
    get_spatial_size_mask,
    get_spatial_size_video,
    get_spatial_size,
)  # usort: skip

from ._augment import erase, erase_image_pil, erase_image_tensor, erase_video
from ._color import (
    adjust_brightness,
    adjust_brightness_image_pil,
    adjust_brightness_image_tensor,
    adjust_brightness_video,
    adjust_contrast,
    adjust_contrast_image_pil,
    adjust_contrast_image_tensor,
    adjust_contrast_video,
    adjust_gamma,
    adjust_gamma_image_pil,
    adjust_gamma_image_tensor,
    adjust_gamma_video,
    adjust_hue,
    adjust_hue_image_pil,
    adjust_hue_image_tensor,
    adjust_hue_video,
    adjust_saturation,
    adjust_saturation_image_pil,
    adjust_saturation_image_tensor,
    adjust_saturation_video,
    adjust_sharpness,
    adjust_sharpness_image_pil,
    adjust_sharpness_image_tensor,
    adjust_sharpness_video,
    autocontrast,
    autocontrast_image_pil,
    autocontrast_image_tensor,
    autocontrast_video,
    equalize,
    equalize_image_pil,
    equalize_image_tensor,
    equalize_video,
    invert,
    invert_image_pil,
    invert_image_tensor,
    invert_video,
    posterize,
    posterize_image_pil,
    posterize_image_tensor,
    posterize_video,
    rgb_to_grayscale,
    rgb_to_grayscale_image_pil,
    rgb_to_grayscale_image_tensor,
    solarize,
    solarize_image_pil,
    solarize_image_tensor,
    solarize_video,
    to_grayscale,
)
from ._geometry import (
    affine,
    affine_bounding_box,
    affine_image_pil,
    affine_image_tensor,
    affine_mask,
    affine_video,
    center_crop,
    center_crop_bounding_box,
    center_crop_image_pil,
    center_crop_image_tensor,
    center_crop_mask,
    center_crop_video,
    crop,
    crop_bounding_box,
    crop_image_pil,
    crop_image_tensor,
    crop_mask,
    crop_video,
    elastic,
    elastic_bounding_box,
    elastic_image_pil,
    elastic_image_tensor,
    elastic_mask,
    elastic_transform,
    elastic_video,
    five_crop,
    five_crop_image_pil,
    five_crop_image_tensor,
    five_crop_video,
    hflip,  # TODO: Consider moving all pure alias definitions at the bottom of the file
    horizontal_flip,
    horizontal_flip_bounding_box,
    horizontal_flip_image_pil,
    horizontal_flip_image_tensor,
    horizontal_flip_mask,
    horizontal_flip_video,
    pad,
    pad_bounding_box,
    pad_image_pil,
    pad_image_tensor,
    pad_mask,
    pad_video,
    perspective,
    perspective_bounding_box,
    perspective_image_pil,
    perspective_image_tensor,
    perspective_mask,
    perspective_video,
    resize,
    resize_bounding_box,
    resize_image_pil,
    resize_image_tensor,
    resize_mask,
    resize_video,
    resized_crop,
    resized_crop_bounding_box,
    resized_crop_image_pil,
    resized_crop_image_tensor,
    resized_crop_mask,
    resized_crop_video,
    rotate,
    rotate_bounding_box,
    rotate_image_pil,
    rotate_image_tensor,
    rotate_mask,
    rotate_video,
    ten_crop,
    ten_crop_image_pil,
    ten_crop_image_tensor,
    ten_crop_video,
    vertical_flip,
    vertical_flip_bounding_box,
    vertical_flip_image_pil,
    vertical_flip_image_tensor,
    vertical_flip_mask,
    vertical_flip_video,
    vflip,
)
from ._misc import (
    gaussian_blur,
    gaussian_blur_image_pil,
    gaussian_blur_image_tensor,
    gaussian_blur_video,
    normalize,
    normalize_image_tensor,
    normalize_video,
)
from ._temporal import uniform_temporal_subsample, uniform_temporal_subsample_video
from ._type_conversion import pil_to_tensor, to_image_pil, to_image_tensor, to_pil_image

from ._deprecated import get_image_size, to_tensor  # usort: skip


def _register_builtin_kernels():
    import functools
    import inspect

    import torch
    from torchvision import datapoints

    from ._utils import _KERNEL_REGISTRY, _noop

    def default_kernel_wrapper(dispatcher, kernel):
        dispatcher_params = list(inspect.signature(dispatcher).parameters)[1:]
        kernel_params = list(inspect.signature(kernel).parameters)[1:]

        needs_args_kwargs_handling = kernel_params != dispatcher_params

        # this avoids converting list -> set at runtime below
        kernel_params = set(kernel_params)

        @functools.wraps(kernel)
        def wrapper(inpt, *args, **kwargs):
            input_type = type(inpt)

            if needs_args_kwargs_handling:
                # Convert args to kwargs to simplify further processing
                kwargs.update(dict(zip(dispatcher_params, args)))
                args = ()

                # drop parameters that are not relevant for the kernel, but have a default value
                # in the dispatcher
                for kwarg in kwargs.keys() - kernel_params:
                    del kwargs[kwarg]

                # add parameters that are passed implicitly to the dispatcher as metadata,
                # but have to be explicit for the kernel
                for kwarg in input_type.__annotations__.keys() & kernel_params:
                    kwargs[kwarg] = getattr(inpt, kwarg)

            output = kernel(inpt.as_subclass(torch.Tensor), *args, **kwargs)

            if isinstance(inpt, datapoints.BoundingBox) and isinstance(output, tuple):
                output, spatial_size = output
                metadata = dict(spatial_size=spatial_size)
            else:
                metadata = dict()

            return input_type.wrap_like(inpt, output, **metadata)

        return wrapper

    def register(dispatcher, datapoint_cls, kernel):
        _KERNEL_REGISTRY.setdefault(dispatcher, {})[datapoint_cls] = default_kernel_wrapper(dispatcher, kernel)

    register(resize, datapoints.Image, resize_image_tensor)
    register(resize, datapoints.BoundingBox, resize_bounding_box)
    register(resize, datapoints.Mask, resize_mask)
    register(resize, datapoints.Video, resize_video)

    register(adjust_brightness, datapoints.Image, adjust_brightness_image_tensor)
    register(adjust_brightness, datapoints.BoundingBox, _noop)
    register(adjust_brightness, datapoints.Mask, _noop)
    register(adjust_brightness, datapoints.Video, adjust_brightness_video)


_register_builtin_kernels()
del _register_builtin_kernels
