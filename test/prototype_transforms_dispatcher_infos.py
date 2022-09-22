import collections.abc
import dataclasses
from typing import Callable, Dict, Optional, Sequence, Type

import pytest
import torchvision.prototype.transforms.functional as F
from prototype_common_utils import BoundingBoxLoader
from prototype_transforms_kernel_infos import KERNEL_INFOS, KernelInfo, Skip
from torchvision.prototype import features

__all__ = ["DispatcherInfo", "DISPATCHER_INFOS"]

KERNEL_INFO_MAP = {info.kernel: info for info in KERNEL_INFOS}


@dataclasses.dataclass
class PILKernelInfo:
    kernel: Callable
    kernel_name: str = dataclasses.field(default=None)

    def __post_init__(self):
        self.kernel_name = self.kernel_name or self.kernel.__name__


@dataclasses.dataclass
class DispatcherInfo:
    dispatcher: Callable
    kernels: Dict[Type, Callable]
    kernel_infos: Dict[Type, KernelInfo] = dataclasses.field(default=None)
    pil_kernel_info: Optional[PILKernelInfo] = None
    method_name: str = dataclasses.field(default=None)
    skips: Sequence[Skip] = dataclasses.field(default_factory=list)
    _skips_map: Dict[str, Skip] = dataclasses.field(default=None, init=False)

    def __post_init__(self):
        self.kernel_infos = {feature_type: KERNEL_INFO_MAP[kernel] for feature_type, kernel in self.kernels.items()}
        self.method_name = self.method_name or self.dispatcher.__name__
        self._skips_map = {skip.test_name: skip for skip in self.skips}

    def sample_inputs(self, *feature_types, filter_metadata=True):
        for feature_type in feature_types or self.kernels.keys():
            if feature_type not in self.kernels:
                raise pytest.UsageError(f"There is no kernel registered for type {feature_type.__name__}")

            sample_inputs = self.kernel_infos[feature_type].sample_inputs_fn()
            if not filter_metadata:
                yield from sample_inputs
            else:
                for args_kwargs in sample_inputs:
                    for attribute in feature_type.__annotations__.keys():
                        if attribute in args_kwargs.kwargs:
                            del args_kwargs.kwargs[attribute]

                    yield args_kwargs

    def maybe_skip(self, *, test_name, args_kwargs, device):
        skip = self._skips_map.get(test_name)
        if skip and skip.condition(args_kwargs, device):
            pytest.skip(skip.reason)


def fill_sequence_needs_broadcast(args_kwargs, device):
    (image_loader, *_), kwargs = args_kwargs
    try:
        fill = kwargs["fill"]
    except KeyError:
        return False

    if not isinstance(fill, collections.abc.Sequence) or len(fill) > 1:
        return False

    return image_loader.num_channels > 1


skip_dispatch_pil_if_fill_sequence_needs_broadcast = Skip(
    "test_dispatch_pil",
    condition=fill_sequence_needs_broadcast,
    reason="PIL kernel doesn't support sequences of length 1 if the number of channels is larger.",
)

skip_dispatch_feature = Skip(
    "test_dispatch_feature",
    reason="Dispatcher doesn't support arbitrary feature dispatch.",
)


DISPATCHER_INFOS = [
    DispatcherInfo(
        F.horizontal_flip,
        kernels={
            features.Image: F.horizontal_flip_image_tensor,
            features.BoundingBox: F.horizontal_flip_bounding_box,
            features.Mask: F.horizontal_flip_mask,
        },
        pil_kernel_info=PILKernelInfo(F.horizontal_flip_image_pil, kernel_name="horizontal_flip_image_pil"),
    ),
    DispatcherInfo(
        F.resize,
        kernels={
            features.Image: F.resize_image_tensor,
            features.BoundingBox: F.resize_bounding_box,
            features.Mask: F.resize_mask,
        },
        pil_kernel_info=PILKernelInfo(F.resize_image_pil),
    ),
    DispatcherInfo(
        F.affine,
        kernels={
            features.Image: F.affine_image_tensor,
            features.BoundingBox: F.affine_bounding_box,
            features.Mask: F.affine_mask,
        },
        pil_kernel_info=PILKernelInfo(F.affine_image_pil),
        skips=[
            skip_dispatch_pil_if_fill_sequence_needs_broadcast,
        ],
    ),
    DispatcherInfo(
        F.vertical_flip,
        kernels={
            features.Image: F.vertical_flip_image_tensor,
            features.BoundingBox: F.vertical_flip_bounding_box,
            features.Mask: F.vertical_flip_mask,
        },
        pil_kernel_info=PILKernelInfo(F.vertical_flip_image_pil, kernel_name="vertical_flip_image_pil"),
    ),
    DispatcherInfo(
        F.rotate,
        kernels={
            features.Image: F.rotate_image_tensor,
            features.BoundingBox: F.rotate_bounding_box,
            features.Mask: F.rotate_mask,
        },
        pil_kernel_info=PILKernelInfo(F.rotate_image_pil),
    ),
    DispatcherInfo(
        F.crop,
        kernels={
            features.Image: F.crop_image_tensor,
            features.BoundingBox: F.crop_bounding_box,
            features.Mask: F.crop_mask,
        },
        pil_kernel_info=PILKernelInfo(F.crop_image_pil, kernel_name="crop_image_pil"),
        skips=[
            Skip(
                "test_dispatch_feature",
                condition=lambda args_kwargs, device: isinstance(args_kwargs.args[0], BoundingBoxLoader),
                reason=(
                    "F.crop expects 4 coordinates as input, but bounding box sample inputs only generate two "
                    "since that is sufficient for the kernel."
                ),
            )
        ],
    ),
    DispatcherInfo(
        F.resized_crop,
        kernels={
            features.Image: F.resized_crop_image_tensor,
            features.BoundingBox: F.resized_crop_bounding_box,
            features.Mask: F.resized_crop_mask,
        },
        pil_kernel_info=PILKernelInfo(F.resized_crop_image_pil),
    ),
    DispatcherInfo(
        F.pad,
        kernels={
            features.Image: F.pad_image_tensor,
            features.BoundingBox: F.pad_bounding_box,
            features.Mask: F.pad_mask,
        },
        skips=[
            skip_dispatch_pil_if_fill_sequence_needs_broadcast,
        ],
        pil_kernel_info=PILKernelInfo(F.pad_image_pil, kernel_name="pad_image_pil"),
    ),
    DispatcherInfo(
        F.perspective,
        kernels={
            features.Image: F.perspective_image_tensor,
            features.BoundingBox: F.perspective_bounding_box,
            features.Mask: F.perspective_mask,
        },
        skips=[
            skip_dispatch_pil_if_fill_sequence_needs_broadcast,
        ],
        pil_kernel_info=PILKernelInfo(F.perspective_image_pil),
    ),
    DispatcherInfo(
        F.elastic,
        kernels={
            features.Image: F.elastic_image_tensor,
            features.BoundingBox: F.elastic_bounding_box,
            features.Mask: F.elastic_mask,
        },
        pil_kernel_info=PILKernelInfo(F.elastic_image_pil),
    ),
    DispatcherInfo(
        F.center_crop,
        kernels={
            features.Image: F.center_crop_image_tensor,
            features.BoundingBox: F.center_crop_bounding_box,
            features.Mask: F.center_crop_mask,
        },
        pil_kernel_info=PILKernelInfo(F.center_crop_image_pil),
    ),
    DispatcherInfo(
        F.gaussian_blur,
        kernels={
            features.Image: F.gaussian_blur_image_tensor,
        },
        pil_kernel_info=PILKernelInfo(F.gaussian_blur_image_pil),
    ),
    DispatcherInfo(
        F.equalize,
        kernels={
            features.Image: F.equalize_image_tensor,
        },
        pil_kernel_info=PILKernelInfo(F.equalize_image_pil, kernel_name="equalize_image_pil"),
    ),
    DispatcherInfo(
        F.invert,
        kernels={
            features.Image: F.invert_image_tensor,
        },
        pil_kernel_info=PILKernelInfo(F.invert_image_pil, kernel_name="invert_image_pil"),
    ),
    DispatcherInfo(
        F.posterize,
        kernels={
            features.Image: F.posterize_image_tensor,
        },
        pil_kernel_info=PILKernelInfo(F.posterize_image_pil, kernel_name="posterize_image_pil"),
    ),
    DispatcherInfo(
        F.solarize,
        kernels={
            features.Image: F.solarize_image_tensor,
        },
        pil_kernel_info=PILKernelInfo(F.solarize_image_pil, kernel_name="solarize_image_pil"),
    ),
    DispatcherInfo(
        F.autocontrast,
        kernels={
            features.Image: F.autocontrast_image_tensor,
        },
        pil_kernel_info=PILKernelInfo(F.autocontrast_image_pil, kernel_name="autocontrast_image_pil"),
    ),
    DispatcherInfo(
        F.adjust_sharpness,
        kernels={
            features.Image: F.adjust_sharpness_image_tensor,
        },
        pil_kernel_info=PILKernelInfo(F.adjust_sharpness_image_pil, kernel_name="adjust_sharpness_image_pil"),
    ),
    DispatcherInfo(
        F.erase,
        kernels={
            features.Image: F.erase_image_tensor,
        },
        pil_kernel_info=PILKernelInfo(F.erase_image_pil),
        skips=[
            skip_dispatch_feature,
        ],
    ),
    DispatcherInfo(
        F.adjust_brightness,
        kernels={
            features.Image: F.adjust_brightness_image_tensor,
        },
        pil_kernel_info=PILKernelInfo(F.adjust_brightness_image_pil, kernel_name="adjust_brightness_image_pil"),
    ),
    DispatcherInfo(
        F.adjust_contrast,
        kernels={
            features.Image: F.adjust_contrast_image_tensor,
        },
        pil_kernel_info=PILKernelInfo(F.adjust_contrast_image_pil, kernel_name="adjust_contrast_image_pil"),
    ),
    DispatcherInfo(
        F.adjust_gamma,
        kernels={
            features.Image: F.adjust_gamma_image_tensor,
        },
        pil_kernel_info=PILKernelInfo(F.adjust_gamma_image_pil, kernel_name="adjust_gamma_image_pil"),
    ),
    DispatcherInfo(
        F.adjust_hue,
        kernels={
            features.Image: F.adjust_hue_image_tensor,
        },
        pil_kernel_info=PILKernelInfo(F.adjust_hue_image_pil, kernel_name="adjust_hue_image_pil"),
    ),
    DispatcherInfo(
        F.adjust_saturation,
        kernels={
            features.Image: F.adjust_saturation_image_tensor,
        },
        pil_kernel_info=PILKernelInfo(F.adjust_saturation_image_pil, kernel_name="adjust_saturation_image_pil"),
    ),
    DispatcherInfo(
        F.five_crop,
        kernels={
            features.Image: F.five_crop_image_tensor,
        },
        pil_kernel_info=PILKernelInfo(F.five_crop_image_pil),
        skips=[
            Skip(
                "test_scripted_smoke",
                condition=lambda args_kwargs, device: isinstance(args_kwargs.kwargs["size"], int),
                reason="Integer size is not supported when scripting five_crop_image_tensor.",
            ),
            skip_dispatch_feature,
        ],
    ),
    DispatcherInfo(
        F.ten_crop,
        kernels={
            features.Image: F.ten_crop_image_tensor,
        },
        pil_kernel_info=PILKernelInfo(F.ten_crop_image_pil),
        skips=[
            Skip(
                "test_scripted_smoke",
                condition=lambda args_kwargs, device: isinstance(args_kwargs.kwargs["size"], int),
                reason="Integer size is not supported when scripting ten_crop_image_tensor.",
            ),
            skip_dispatch_feature,
        ],
    ),
    DispatcherInfo(
        F.normalize,
        kernels={
            features.Image: F.normalize_image_tensor,
        },
        skips=[
            skip_dispatch_feature,
        ],
    ),
]
