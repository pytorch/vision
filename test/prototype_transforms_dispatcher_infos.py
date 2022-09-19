import dataclasses
from typing import Callable, Dict, Type

import pytest
import torchvision.prototype.transforms.functional as F

from prototype_transforms_kernel_infos import KERNEL_INFOS

from torchvision.prototype import features

__all__ = ["DispatcherInfo", "DISPATCHER_INFOS"]


KERNEL_SAMPLE_INPUTS_FN_MAP = {info.kernel: info.sample_inputs_fn for info in KERNEL_INFOS}


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
]
