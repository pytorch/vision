import collections.abc

import pytest
import torchvision.prototype.transforms.functional as F
from prototype_common_utils import InfoBase, TestMark
from prototype_transforms_kernel_infos import KERNEL_INFOS
from torchvision.prototype import features

__all__ = ["DispatcherInfo", "DISPATCHER_INFOS"]


class PILKernelInfo(InfoBase):
    def __init__(
        self,
        kernel,
        *,
        # Defaults to `kernel.__name__`. Should be set if the function is exposed under a different name
        # TODO: This can probably be removed after roll-out since we shouldn't have any aliasing then
        kernel_name=None,
    ):
        super().__init__(id=kernel_name or kernel.__name__)
        self.kernel = kernel


class DispatcherInfo(InfoBase):
    _KERNEL_INFO_MAP = {info.kernel: info for info in KERNEL_INFOS}

    def __init__(
        self,
        dispatcher,
        *,
        # Dictionary of types that map to the kernel the dispatcher dispatches to.
        kernels,
        # If omitted, no PIL dispatch test will be performed.
        pil_kernel_info=None,
        # See InfoBase
        test_marks=None,
        # See InfoBase
        closeness_kwargs=None,
    ):
        super().__init__(id=dispatcher.__name__, test_marks=test_marks, closeness_kwargs=closeness_kwargs)
        self.dispatcher = dispatcher
        self.kernels = kernels
        self.pil_kernel_info = pil_kernel_info

        kernel_infos = {}
        for feature_type, kernel in self.kernels.items():
            kernel_info = self._KERNEL_INFO_MAP.get(kernel)
            if not kernel_info:
                raise pytest.UsageError(
                    f"Can't register {kernel.__name__} for type {feature_type} since there is no `KernelInfo` for it. "
                    f"Please add a `KernelInfo` for it in `prototype_transforms_kernel_infos.py`."
                )
            kernel_infos[feature_type] = kernel_info
        self.kernel_infos = kernel_infos

    def sample_inputs(self, *feature_types, filter_metadata=True):
        for feature_type in feature_types or self.kernel_infos.keys():
            kernel_info = self.kernel_infos.get(feature_type)
            if not kernel_info:
                raise pytest.UsageError(f"There is no kernel registered for type {type.__name__}")

            sample_inputs = kernel_info.sample_inputs_fn()

            if not filter_metadata:
                yield from sample_inputs
            else:
                for args_kwargs in sample_inputs:
                    for attribute in feature_type.__annotations__.keys():
                        if attribute in args_kwargs.kwargs:
                            del args_kwargs.kwargs[attribute]

                    yield args_kwargs


def xfail_jit(reason, *, condition=None):
    return TestMark(
        ("TestDispatchers", "test_scripted_smoke"),
        pytest.mark.xfail(reason=reason),
        condition=condition,
    )


def xfail_jit_python_scalar_arg(name, *, reason=None):
    return xfail_jit(
        reason or f"Python scalar int or float for `{name}` is not supported when scripting",
        condition=lambda args_kwargs: isinstance(args_kwargs.kwargs.get(name), (int, float)),
    )


def xfail_jit_tuple_instead_of_list(name, *, reason=None):
    return xfail_jit(
        reason or f"Passing a tuple instead of a list for `{name}` is not supported when scripting",
        condition=lambda args_kwargs: isinstance(args_kwargs.kwargs.get(name), tuple),
    )


def is_list_of_ints(args_kwargs):
    fill = args_kwargs.kwargs.get("fill")
    return isinstance(fill, list) and any(isinstance(scalar_fill, int) for scalar_fill in fill)


def xfail_jit_list_of_ints(name, *, reason=None):
    return xfail_jit(
        reason or f"Passing a list of integers for `{name}` is not supported when scripting",
        condition=is_list_of_ints,
    )


skip_dispatch_feature = TestMark(
    ("TestDispatchers", "test_dispatch_feature"),
    pytest.mark.skip(reason="Dispatcher doesn't support arbitrary feature dispatch."),
)


def fill_sequence_needs_broadcast(args_kwargs):
    (image_loader, *_), kwargs = args_kwargs
    try:
        fill = kwargs["fill"]
    except KeyError:
        return False

    if not isinstance(fill, collections.abc.Sequence) or len(fill) > 1:
        return False

    return image_loader.num_channels > 1


xfail_dispatch_pil_if_fill_sequence_needs_broadcast = TestMark(
    ("TestDispatchers", "test_dispatch_pil"),
    pytest.mark.xfail(
        reason="PIL kernel doesn't support sequences of length 1 for `fill` if the number of color channels is larger."
    ),
    condition=fill_sequence_needs_broadcast,
)


DISPATCHER_INFOS = [
    DispatcherInfo(
        F.horizontal_flip,
        kernels={
            features.Image: F.horizontal_flip_image_tensor,
            features.Video: F.horizontal_flip_video,
            features.BoundingBox: F.horizontal_flip_bounding_box,
            features.Mask: F.horizontal_flip_mask,
        },
        pil_kernel_info=PILKernelInfo(F.horizontal_flip_image_pil, kernel_name="horizontal_flip_image_pil"),
    ),
    DispatcherInfo(
        F.resize,
        kernels={
            features.Image: F.resize_image_tensor,
            features.Video: F.resize_video,
            features.BoundingBox: F.resize_bounding_box,
            features.Mask: F.resize_mask,
        },
        pil_kernel_info=PILKernelInfo(F.resize_image_pil),
        test_marks=[
            xfail_jit_python_scalar_arg("size"),
        ],
    ),
    DispatcherInfo(
        F.affine,
        kernels={
            features.Image: F.affine_image_tensor,
            features.Video: F.affine_video,
            features.BoundingBox: F.affine_bounding_box,
            features.Mask: F.affine_mask,
        },
        pil_kernel_info=PILKernelInfo(F.affine_image_pil),
        test_marks=[
            xfail_dispatch_pil_if_fill_sequence_needs_broadcast,
            xfail_jit_python_scalar_arg("shear"),
            xfail_jit_tuple_instead_of_list("fill"),
            # TODO: check if this is a regression since it seems that should be supported if `int` is ok
            xfail_jit_list_of_ints("fill"),
        ],
    ),
    DispatcherInfo(
        F.vertical_flip,
        kernels={
            features.Image: F.vertical_flip_image_tensor,
            features.Video: F.vertical_flip_video,
            features.BoundingBox: F.vertical_flip_bounding_box,
            features.Mask: F.vertical_flip_mask,
        },
        pil_kernel_info=PILKernelInfo(F.vertical_flip_image_pil, kernel_name="vertical_flip_image_pil"),
    ),
    DispatcherInfo(
        F.rotate,
        kernels={
            features.Image: F.rotate_image_tensor,
            features.Video: F.rotate_video,
            features.BoundingBox: F.rotate_bounding_box,
            features.Mask: F.rotate_mask,
        },
        pil_kernel_info=PILKernelInfo(F.rotate_image_pil),
        test_marks=[
            xfail_jit_tuple_instead_of_list("fill"),
            # TODO: check if this is a regression since it seems that should be supported if `int` is ok
            xfail_jit_list_of_ints("fill"),
        ],
    ),
    DispatcherInfo(
        F.crop,
        kernels={
            features.Image: F.crop_image_tensor,
            features.Video: F.crop_video,
            features.BoundingBox: F.crop_bounding_box,
            features.Mask: F.crop_mask,
        },
        pil_kernel_info=PILKernelInfo(F.crop_image_pil, kernel_name="crop_image_pil"),
    ),
    DispatcherInfo(
        F.resized_crop,
        kernels={
            features.Image: F.resized_crop_image_tensor,
            features.Video: F.resized_crop_video,
            features.BoundingBox: F.resized_crop_bounding_box,
            features.Mask: F.resized_crop_mask,
        },
        pil_kernel_info=PILKernelInfo(F.resized_crop_image_pil),
    ),
    DispatcherInfo(
        F.pad,
        kernels={
            features.Image: F.pad_image_tensor,
            features.Video: F.pad_video,
            features.BoundingBox: F.pad_bounding_box,
            features.Mask: F.pad_mask,
        },
        pil_kernel_info=PILKernelInfo(F.pad_image_pil, kernel_name="pad_image_pil"),
        test_marks=[
            TestMark(
                ("TestDispatchers", "test_dispatch_pil"),
                pytest.mark.xfail(
                    reason=(
                        "PIL kernel doesn't support sequences of length 1 for argument `fill` and "
                        "`padding_mode='constant'`, if the number of color channels is larger."
                    )
                ),
                condition=lambda args_kwargs: fill_sequence_needs_broadcast(args_kwargs)
                and args_kwargs.kwargs.get("padding_mode", "constant") == "constant",
            ),
            xfail_jit_tuple_instead_of_list("padding"),
            xfail_jit_tuple_instead_of_list("fill"),
            # TODO: check if this is a regression since it seems that should be supported if `int` is ok
            xfail_jit_list_of_ints("fill"),
        ],
    ),
    DispatcherInfo(
        F.perspective,
        kernels={
            features.Image: F.perspective_image_tensor,
            features.Video: F.perspective_video,
            features.BoundingBox: F.perspective_bounding_box,
            features.Mask: F.perspective_mask,
        },
        pil_kernel_info=PILKernelInfo(F.perspective_image_pil),
        test_marks=[
            xfail_dispatch_pil_if_fill_sequence_needs_broadcast,
        ],
    ),
    DispatcherInfo(
        F.elastic,
        kernels={
            features.Image: F.elastic_image_tensor,
            features.Video: F.elastic_video,
            features.BoundingBox: F.elastic_bounding_box,
            features.Mask: F.elastic_mask,
        },
        pil_kernel_info=PILKernelInfo(F.elastic_image_pil),
    ),
    DispatcherInfo(
        F.center_crop,
        kernels={
            features.Image: F.center_crop_image_tensor,
            features.Video: F.center_crop_video,
            features.BoundingBox: F.center_crop_bounding_box,
            features.Mask: F.center_crop_mask,
        },
        pil_kernel_info=PILKernelInfo(F.center_crop_image_pil),
        test_marks=[
            xfail_jit_python_scalar_arg("output_size"),
        ],
    ),
    DispatcherInfo(
        F.gaussian_blur,
        kernels={
            features.Image: F.gaussian_blur_image_tensor,
            features.Video: F.gaussian_blur_video,
        },
        pil_kernel_info=PILKernelInfo(F.gaussian_blur_image_pil),
        test_marks=[
            xfail_jit_python_scalar_arg("kernel_size"),
            xfail_jit_python_scalar_arg("sigma"),
        ],
    ),
    DispatcherInfo(
        F.equalize,
        kernels={
            features.Image: F.equalize_image_tensor,
            features.Video: F.equalize_video,
        },
        pil_kernel_info=PILKernelInfo(F.equalize_image_pil, kernel_name="equalize_image_pil"),
    ),
    DispatcherInfo(
        F.invert,
        kernels={
            features.Image: F.invert_image_tensor,
            features.Video: F.invert_video,
        },
        pil_kernel_info=PILKernelInfo(F.invert_image_pil, kernel_name="invert_image_pil"),
    ),
    DispatcherInfo(
        F.posterize,
        kernels={
            features.Image: F.posterize_image_tensor,
            features.Video: F.posterize_video,
        },
        pil_kernel_info=PILKernelInfo(F.posterize_image_pil, kernel_name="posterize_image_pil"),
    ),
    DispatcherInfo(
        F.solarize,
        kernels={
            features.Image: F.solarize_image_tensor,
            features.Video: F.solarize_video,
        },
        pil_kernel_info=PILKernelInfo(F.solarize_image_pil, kernel_name="solarize_image_pil"),
    ),
    DispatcherInfo(
        F.autocontrast,
        kernels={
            features.Image: F.autocontrast_image_tensor,
            features.Video: F.autocontrast_video,
        },
        pil_kernel_info=PILKernelInfo(F.autocontrast_image_pil, kernel_name="autocontrast_image_pil"),
    ),
    DispatcherInfo(
        F.adjust_sharpness,
        kernels={
            features.Image: F.adjust_sharpness_image_tensor,
            features.Video: F.adjust_sharpness_video,
        },
        pil_kernel_info=PILKernelInfo(F.adjust_sharpness_image_pil, kernel_name="adjust_sharpness_image_pil"),
    ),
    DispatcherInfo(
        F.erase,
        kernels={
            features.Image: F.erase_image_tensor,
            features.Video: F.erase_video,
        },
        pil_kernel_info=PILKernelInfo(F.erase_image_pil),
        test_marks=[
            skip_dispatch_feature,
        ],
    ),
    DispatcherInfo(
        F.adjust_brightness,
        kernels={
            features.Image: F.adjust_brightness_image_tensor,
            features.Video: F.adjust_brightness_video,
        },
        pil_kernel_info=PILKernelInfo(F.adjust_brightness_image_pil, kernel_name="adjust_brightness_image_pil"),
    ),
    DispatcherInfo(
        F.adjust_contrast,
        kernels={
            features.Image: F.adjust_contrast_image_tensor,
            features.Video: F.adjust_contrast_video,
        },
        pil_kernel_info=PILKernelInfo(F.adjust_contrast_image_pil, kernel_name="adjust_contrast_image_pil"),
    ),
    DispatcherInfo(
        F.adjust_gamma,
        kernels={
            features.Image: F.adjust_gamma_image_tensor,
            features.Video: F.adjust_gamma_video,
        },
        pil_kernel_info=PILKernelInfo(F.adjust_gamma_image_pil, kernel_name="adjust_gamma_image_pil"),
    ),
    DispatcherInfo(
        F.adjust_hue,
        kernels={
            features.Image: F.adjust_hue_image_tensor,
            features.Video: F.adjust_hue_video,
        },
        pil_kernel_info=PILKernelInfo(F.adjust_hue_image_pil, kernel_name="adjust_hue_image_pil"),
    ),
    DispatcherInfo(
        F.adjust_saturation,
        kernels={
            features.Image: F.adjust_saturation_image_tensor,
            features.Video: F.adjust_saturation_video,
        },
        pil_kernel_info=PILKernelInfo(F.adjust_saturation_image_pil, kernel_name="adjust_saturation_image_pil"),
    ),
    DispatcherInfo(
        F.five_crop,
        kernels={
            features.Image: F.five_crop_image_tensor,
            features.Video: F.five_crop_video,
        },
        pil_kernel_info=PILKernelInfo(F.five_crop_image_pil),
        test_marks=[
            xfail_jit_python_scalar_arg("size"),
            skip_dispatch_feature,
        ],
    ),
    DispatcherInfo(
        F.ten_crop,
        kernels={
            features.Image: F.ten_crop_image_tensor,
            features.Video: F.ten_crop_video,
        },
        test_marks=[
            xfail_jit_python_scalar_arg("size"),
            skip_dispatch_feature,
        ],
        pil_kernel_info=PILKernelInfo(F.ten_crop_image_pil),
    ),
    DispatcherInfo(
        F.normalize,
        kernels={
            features.Image: F.normalize_image_tensor,
            features.Video: F.normalize_video,
        },
        test_marks=[
            skip_dispatch_feature,
            xfail_jit_python_scalar_arg("mean"),
            xfail_jit_python_scalar_arg("std"),
        ],
    ),
    DispatcherInfo(
        F.convert_dtype,
        kernels={
            features.Image: F.convert_dtype_image_tensor,
            features.Video: F.convert_dtype_video,
        },
        test_marks=[
            skip_dispatch_feature,
        ],
    ),
    DispatcherInfo(
        F.uniform_temporal_subsample,
        kernels={
            features.Video: F.uniform_temporal_subsample_video,
        },
        test_marks=[
            skip_dispatch_feature,
        ],
    ),
]
