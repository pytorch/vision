from typing import TypeVar

import torch

from ._bounding_boxes import BoundingBoxes, BoundingBoxFormat, is_rotated_bounding_format
from ._image import Image
from ._keypoints import KeyPoints
from ._mask import Mask
from ._torch_function_helpers import set_return_type
from ._tv_tensor import TVTensor
from torchvision.tv_tensors._tv_tensor import TVTensor

from ._video import Video


TVTensorType = TypeVar("TVTensorType", bound=TVTensor)


# TODO: Fix this. We skip this method as it leads to
# RecursionError: maximum recursion depth exceeded while calling a Python object
# Until `disable` is removed, there will be graph breaks after all calls to functional transforms
@torch.compiler.disable
def wrap(wrappee: torch.Tensor, *, like: TVTensorType, **kwargs) -> TVTensorType:
    """Convert a :class:`torch.Tensor` (``wrappee``) into the same :class:`~torchvision.tv_tensors.TVTensor` subclass as ``like``.

    Args:
        wrappee (Tensor): The tensor to convert.
        like (:class:`~torchvision.tv_tensors.TVTensor`): The reference.
            ``wrappee`` will be converted into the same subclass as ``like``
            maintaining the same metadata as ``like``.
        kwargs: Optional overrides for metadata. For BoundingBoxes: ``format``, ``canvas_size``, ``clamping_mode``.
            For KeyPoints: ``canvas_size``.
    """
    if not hasattr(like, "__wrap__"):
        raise TypeError(f"Expected `like` to have a `__wrap__` method, but got {type(like)}")

    return like.__wrap__(wrappee, **kwargs)
