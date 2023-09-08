import pytest
import torchvision.transforms.v2.functional as F
from torchvision import tv_tensors
from transforms_v2_kernel_infos import KERNEL_INFOS
from transforms_v2_legacy_utils import InfoBase, TestMark

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
        for tv_tensor_type, kernel in self.kernels.items():
            kernel_info = self._KERNEL_INFO_MAP.get(kernel)
            if not kernel_info:
                raise pytest.UsageError(
                    f"Can't register {kernel.__name__} for type {tv_tensor_type} since there is no `KernelInfo` for it. "
                    f"Please add a `KernelInfo` for it in `transforms_v2_kernel_infos.py`."
                )
            kernel_infos[tv_tensor_type] = kernel_info
        self.kernel_infos = kernel_infos

    def sample_inputs(self, *tv_tensor_types, filter_metadata=True):
        for tv_tensor_type in tv_tensor_types or self.kernel_infos.keys():
            kernel_info = self.kernel_infos.get(tv_tensor_type)
            if not kernel_info:
                raise pytest.UsageError(f"There is no kernel registered for type {type.__name__}")

            sample_inputs = kernel_info.sample_inputs_fn()

            if not filter_metadata:
                yield from sample_inputs
                return

            import itertools

            for args_kwargs in sample_inputs:
                if hasattr(tv_tensor_type, "__annotations__"):
                    for name in itertools.chain(
                        tv_tensor_type.__annotations__.keys(),
                        # FIXME: this seems ok for conversion dispatchers, but we should probably handle this on a
                        #  per-dispatcher level. However, so far there is no option for that.
                        (f"old_{name}" for name in tv_tensor_type.__annotations__.keys()),
                    ):
                        if name in args_kwargs.kwargs:
                            del args_kwargs.kwargs[name]

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


skip_dispatch_tv_tensor = TestMark(
    ("TestDispatchers", "test_dispatch_tv_tensor"),
    pytest.mark.skip(reason="Dispatcher doesn't support arbitrary tv_tensor dispatch."),
)

multi_crop_skips = [
    TestMark(
        ("TestDispatchers", test_name),
        pytest.mark.skip(reason="Multi-crop dispatchers return a sequence of items rather than a single one."),
    )
    for test_name in ["test_pure_tensor_output_type", "test_pil_output_type", "test_tv_tensor_output_type"]
]
multi_crop_skips.append(skip_dispatch_tv_tensor)


DISPATCHER_INFOS = [
    DispatcherInfo(
        F.perspective,
        kernels={
            tv_tensors.Image: F.perspective_image,
            tv_tensors.Video: F.perspective_video,
            tv_tensors.BoundingBoxes: F.perspective_bounding_boxes,
            tv_tensors.Mask: F.perspective_mask,
        },
        pil_kernel_info=PILKernelInfo(F._perspective_image_pil),
        test_marks=[
            xfail_jit_python_scalar_arg("fill"),
        ],
    ),
    DispatcherInfo(
        F.elastic,
        kernels={
            tv_tensors.Image: F.elastic_image,
            tv_tensors.Video: F.elastic_video,
            tv_tensors.BoundingBoxes: F.elastic_bounding_boxes,
            tv_tensors.Mask: F.elastic_mask,
        },
        pil_kernel_info=PILKernelInfo(F._elastic_image_pil),
        test_marks=[xfail_jit_python_scalar_arg("fill")],
    ),
    DispatcherInfo(
        F.equalize,
        kernels={
            tv_tensors.Image: F.equalize_image,
            tv_tensors.Video: F.equalize_video,
        },
        pil_kernel_info=PILKernelInfo(F._equalize_image_pil, kernel_name="equalize_image_pil"),
    ),
    DispatcherInfo(
        F.invert,
        kernels={
            tv_tensors.Image: F.invert_image,
            tv_tensors.Video: F.invert_video,
        },
        pil_kernel_info=PILKernelInfo(F._invert_image_pil, kernel_name="invert_image_pil"),
    ),
    DispatcherInfo(
        F.posterize,
        kernels={
            tv_tensors.Image: F.posterize_image,
            tv_tensors.Video: F.posterize_video,
        },
        pil_kernel_info=PILKernelInfo(F._posterize_image_pil, kernel_name="posterize_image_pil"),
    ),
    DispatcherInfo(
        F.solarize,
        kernels={
            tv_tensors.Image: F.solarize_image,
            tv_tensors.Video: F.solarize_video,
        },
        pil_kernel_info=PILKernelInfo(F._solarize_image_pil, kernel_name="solarize_image_pil"),
    ),
    DispatcherInfo(
        F.autocontrast,
        kernels={
            tv_tensors.Image: F.autocontrast_image,
            tv_tensors.Video: F.autocontrast_video,
        },
        pil_kernel_info=PILKernelInfo(F._autocontrast_image_pil, kernel_name="autocontrast_image_pil"),
    ),
    DispatcherInfo(
        F.adjust_sharpness,
        kernels={
            tv_tensors.Image: F.adjust_sharpness_image,
            tv_tensors.Video: F.adjust_sharpness_video,
        },
        pil_kernel_info=PILKernelInfo(F._adjust_sharpness_image_pil, kernel_name="adjust_sharpness_image_pil"),
    ),
    DispatcherInfo(
        F.adjust_contrast,
        kernels={
            tv_tensors.Image: F.adjust_contrast_image,
            tv_tensors.Video: F.adjust_contrast_video,
        },
        pil_kernel_info=PILKernelInfo(F._adjust_contrast_image_pil, kernel_name="adjust_contrast_image_pil"),
    ),
    DispatcherInfo(
        F.adjust_gamma,
        kernels={
            tv_tensors.Image: F.adjust_gamma_image,
            tv_tensors.Video: F.adjust_gamma_video,
        },
        pil_kernel_info=PILKernelInfo(F._adjust_gamma_image_pil, kernel_name="adjust_gamma_image_pil"),
    ),
    DispatcherInfo(
        F.adjust_hue,
        kernels={
            tv_tensors.Image: F.adjust_hue_image,
            tv_tensors.Video: F.adjust_hue_video,
        },
        pil_kernel_info=PILKernelInfo(F._adjust_hue_image_pil, kernel_name="adjust_hue_image_pil"),
    ),
    DispatcherInfo(
        F.adjust_saturation,
        kernels={
            tv_tensors.Image: F.adjust_saturation_image,
            tv_tensors.Video: F.adjust_saturation_video,
        },
        pil_kernel_info=PILKernelInfo(F._adjust_saturation_image_pil, kernel_name="adjust_saturation_image_pil"),
    ),
    DispatcherInfo(
        F.five_crop,
        kernels={
            tv_tensors.Image: F.five_crop_image,
            tv_tensors.Video: F.five_crop_video,
        },
        pil_kernel_info=PILKernelInfo(F._five_crop_image_pil),
        test_marks=[
            xfail_jit_python_scalar_arg("size"),
            *multi_crop_skips,
        ],
    ),
    DispatcherInfo(
        F.ten_crop,
        kernels={
            tv_tensors.Image: F.ten_crop_image,
            tv_tensors.Video: F.ten_crop_video,
        },
        test_marks=[
            xfail_jit_python_scalar_arg("size"),
            *multi_crop_skips,
        ],
        pil_kernel_info=PILKernelInfo(F._ten_crop_image_pil),
    ),
    DispatcherInfo(
        F.normalize,
        kernels={
            tv_tensors.Image: F.normalize_image,
            tv_tensors.Video: F.normalize_video,
        },
        test_marks=[
            xfail_jit_python_scalar_arg("mean"),
            xfail_jit_python_scalar_arg("std"),
        ],
    ),
    DispatcherInfo(
        F.uniform_temporal_subsample,
        kernels={
            tv_tensors.Video: F.uniform_temporal_subsample_video,
        },
        test_marks=[
            skip_dispatch_tv_tensor,
        ],
    ),
    DispatcherInfo(
        F.clamp_bounding_boxes,
        kernels={tv_tensors.BoundingBoxes: F.clamp_bounding_boxes},
        test_marks=[
            skip_dispatch_tv_tensor,
        ],
    ),
]
