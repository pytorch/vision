import dataclasses
from collections import defaultdict
from typing import Callable, Dict, List, Sequence, Type

import pytest
import torchvision.prototype.transforms.functional as F
from prototype_transforms_kernel_infos import KERNEL_INFOS, Skip
from torchvision.prototype import features

__all__ = ["DispatcherInfo", "DISPATCHER_INFOS"]

KERNEL_SAMPLE_INPUTS_FN_MAP = {info.kernel: info.sample_inputs_fn for info in KERNEL_INFOS}


def skip_python_scalar_arg_jit(name, *, reason="Python scalar int or float is not supported when scripting"):
    return Skip(
        "test_scripted_smoke",
        condition=lambda args_kwargs, device: isinstance(args_kwargs.kwargs[name], (int, float)),
        reason=reason,
    )


def skip_integer_size_jit(name="size"):
    return skip_python_scalar_arg_jit(name, reason="Integer size is not supported when scripting.")


@dataclasses.dataclass
class DispatcherInfo:
    dispatcher: Callable
    kernels: Dict[Type, Callable]
    skips: Sequence[Skip] = dataclasses.field(default_factory=list)
    _skips_map: Dict[str, List[Skip]] = dataclasses.field(default=None, init=False)

    def __post_init__(self):
        skips_map = defaultdict(list)
        for skip in self.skips:
            skips_map[skip.test_name].append(skip)
        self._skips_map = dict(skips_map)

    def sample_inputs(self, *types):
        for type in types or self.kernels.keys():
            if type not in self.kernels:
                raise pytest.UsageError(f"There is no kernel registered for type {type.__name__}")

            yield from KERNEL_SAMPLE_INPUTS_FN_MAP[self.kernels[type]]()

    def maybe_skip(self, *, test_name, args_kwargs, device):
        skips = self._skips_map.get(test_name)
        if not skips:
            return

        for skip in skips:
            if skip.condition(args_kwargs, device):
                pytest.skip(skip.reason)


DISPATCHER_INFOS = [
    DispatcherInfo(
        F.horizontal_flip,
        kernels={
            features.Image: F.horizontal_flip_image_tensor,
            features.BoundingBox: F.horizontal_flip_bounding_box,
            features.Mask: F.horizontal_flip_mask,
        },
    ),
    DispatcherInfo(
        F.resize,
        kernels={
            features.Image: F.resize_image_tensor,
            features.BoundingBox: F.resize_bounding_box,
            features.Mask: F.resize_mask,
        },
        skips=[
            skip_integer_size_jit(),
        ],
    ),
    DispatcherInfo(
        F.affine,
        kernels={
            features.Image: F.affine_image_tensor,
            features.BoundingBox: F.affine_bounding_box,
            features.Mask: F.affine_mask,
        },
        skips=[skip_python_scalar_arg_jit("shear", reason="Scalar shear is not supported by JIT")],
    ),
    DispatcherInfo(
        F.vertical_flip,
        kernels={
            features.Image: F.vertical_flip_image_tensor,
            features.BoundingBox: F.vertical_flip_bounding_box,
            features.Mask: F.vertical_flip_mask,
        },
    ),
    DispatcherInfo(
        F.rotate,
        kernels={
            features.Image: F.rotate_image_tensor,
            features.BoundingBox: F.rotate_bounding_box,
            features.Mask: F.rotate_mask,
        },
    ),
    DispatcherInfo(
        F.crop,
        kernels={
            features.Image: F.crop_image_tensor,
            features.BoundingBox: F.crop_bounding_box,
            features.Mask: F.crop_mask,
        },
    ),
    DispatcherInfo(
        F.resized_crop,
        kernels={
            features.Image: F.resized_crop_image_tensor,
            features.BoundingBox: F.resized_crop_bounding_box,
            features.Mask: F.resized_crop_mask,
        },
    ),
    DispatcherInfo(
        F.pad,
        kernels={
            features.Image: F.pad_image_tensor,
            features.BoundingBox: F.pad_bounding_box,
            features.Mask: F.pad_mask,
        },
    ),
    DispatcherInfo(
        F.perspective,
        kernels={
            features.Image: F.perspective_image_tensor,
            features.BoundingBox: F.perspective_bounding_box,
            features.Mask: F.perspective_mask,
        },
    ),
    DispatcherInfo(
        F.elastic,
        kernels={
            features.Image: F.elastic_image_tensor,
            features.BoundingBox: F.elastic_bounding_box,
            features.Mask: F.elastic_mask,
        },
    ),
    DispatcherInfo(
        F.center_crop,
        kernels={
            features.Image: F.center_crop_image_tensor,
            features.BoundingBox: F.center_crop_bounding_box,
            features.Mask: F.center_crop_mask,
        },
        skips=[
            skip_integer_size_jit("output_size"),
        ],
    ),
    DispatcherInfo(
        F.gaussian_blur,
        kernels={
            features.Image: F.gaussian_blur_image_tensor,
        },
        skips=[
            skip_python_scalar_arg_jit("kernel_size"),
            skip_python_scalar_arg_jit("sigma"),
        ],
    ),
    DispatcherInfo(
        F.equalize,
        kernels={
            features.Image: F.equalize_image_tensor,
        },
    ),
    DispatcherInfo(
        F.invert,
        kernels={
            features.Image: F.invert_image_tensor,
        },
    ),
    DispatcherInfo(
        F.posterize,
        kernels={
            features.Image: F.posterize_image_tensor,
        },
    ),
    DispatcherInfo(
        F.solarize,
        kernels={
            features.Image: F.solarize_image_tensor,
        },
    ),
    DispatcherInfo(
        F.autocontrast,
        kernels={
            features.Image: F.autocontrast_image_tensor,
        },
    ),
    DispatcherInfo(
        F.adjust_sharpness,
        kernels={
            features.Image: F.adjust_sharpness_image_tensor,
        },
    ),
    DispatcherInfo(
        F.erase,
        kernels={
            features.Image: F.erase_image_tensor,
        },
    ),
    DispatcherInfo(
        F.adjust_brightness,
        kernels={
            features.Image: F.adjust_brightness_image_tensor,
        },
    ),
    DispatcherInfo(
        F.adjust_contrast,
        kernels={
            features.Image: F.adjust_contrast_image_tensor,
        },
    ),
    DispatcherInfo(
        F.adjust_gamma,
        kernels={
            features.Image: F.adjust_gamma_image_tensor,
        },
    ),
    DispatcherInfo(
        F.adjust_hue,
        kernels={
            features.Image: F.adjust_hue_image_tensor,
        },
    ),
    DispatcherInfo(
        F.adjust_saturation,
        kernels={
            features.Image: F.adjust_saturation_image_tensor,
        },
    ),
    DispatcherInfo(
        F.five_crop,
        kernels={
            features.Image: F.five_crop_image_tensor,
        },
        skips=[
            skip_integer_size_jit(),
        ],
    ),
    DispatcherInfo(
        F.ten_crop,
        kernels={
            features.Image: F.ten_crop_image_tensor,
        },
        skips=[
            skip_integer_size_jit(),
        ],
    ),
    DispatcherInfo(
        F.normalize,
        kernels={
            features.Image: F.normalize_image_tensor,
        },
    ),
]
