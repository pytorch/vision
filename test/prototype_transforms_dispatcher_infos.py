import dataclasses
import functools
from typing import Callable, Dict, Type

import pytest
import torch
import torchvision.prototype.transforms.functional as F
from prototype_common_utils import ArgsKwargs
from prototype_transforms_kernel_infos import KERNEL_INFOS
from test_prototype_transforms_functional import FUNCTIONAL_INFOS
from torchvision.prototype import features

__all__ = ["DispatcherInfo", "DISPATCHER_INFOS"]

KERNEL_SAMPLE_INPUTS_FN_MAP = {info.kernel: info.sample_inputs_fn for info in KERNEL_INFOS}


# Helper class to use the infos from the old framework for now tests
class PreloadedArgsKwargs(ArgsKwargs):
    def load(self, device="cpu"):
        args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in self.args)
        kwargs = {
            keyword: arg.to(device) if isinstance(arg, torch.Tensor) else arg for keyword, arg in self.kwargs.items()
        }
        return args, kwargs


def preloaded_sample_inputs(args_kwargs):
    for args, kwargs in args_kwargs:
        yield PreloadedArgsKwargs(*args, **kwargs)


KERNEL_SAMPLE_INPUTS_FN_MAP.update(
    {info.functional: functools.partial(preloaded_sample_inputs, info.sample_inputs()) for info in FUNCTIONAL_INFOS}
)


@dataclasses.dataclass
class DispatcherInfo:
    dispatcher: Callable
    kernels: Dict[Type, Callable]

    def sample_inputs(self, *types):
        for type in types or self.kernels.keys():
            if type not in self.kernels:
                raise pytest.UsageError(f"There is no kernel registered for type {type.__name__}")

            yield from KERNEL_SAMPLE_INPUTS_FN_MAP[self.kernels[type]]()


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
    ),
    DispatcherInfo(
        F.affine,
        kernels={
            features.Image: F.affine_image_tensor,
            features.BoundingBox: F.affine_bounding_box,
            features.Mask: F.affine_mask,
        },
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
    # FIXME:
    # RuntimeError: perspective() is missing value for argument 'startpoints'.
    # Declaration: perspective(Tensor inpt, int[][] startpoints, int[][] endpoints,
    # Enum<__torch__.torchvision.transforms.functional.InterpolationMode> interpolation=Enum<InterpolationMode.BILINEAR>,
    # Union(float[], float, int, NoneType) fill=None) -> Tensor
    #
    # This is probably due to the fact that F.perspective does not have the same signature as F.perspective_image_tensor
    # DispatcherInfo(
    #     F.perspective,
    #     kernels={
    #         features.Image: F.perspective_image_tensor,
    #         features.BoundingBox: F.perspective_bounding_box,
    #         features.Mask: F.perspective_mask,
    #     },
    # ),
    DispatcherInfo(
        F.center_crop,
        kernels={
            features.Image: F.center_crop_image_tensor,
            features.BoundingBox: F.center_crop_bounding_box,
            features.Mask: F.center_crop_mask,
        },
    ),
    DispatcherInfo(
        F.gaussian_blur,
        kernels={
            features.Image: F.gaussian_blur_image_tensor,
        },
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
]
